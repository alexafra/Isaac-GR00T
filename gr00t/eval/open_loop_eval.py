# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import re
import time
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.eval._horizon_contract import PolicyHorizonSpec, migrate_deprecated_action_horizon_argv
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    execution_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        execution_horizon: Number of predicted-chunk steps executed per inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, execution_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_action_gr00t(action: dict[str, Any], batch_index: int = 0) -> dict[str, Any]:
    """Select one batch item and add the action prefix used by evaluation."""

    return {f"action.{key}": action[key][batch_index] for key in action}


def required_visual_frame_indices(
    *,
    trajectory_length: int,
    steps: int,
    execution_horizon: int,
    modality_configs: dict[str, Any],
) -> np.ndarray:
    """Return the video/mask rows needed by open-loop inference, in decode order."""

    if execution_horizon <= 0:
        raise ValueError(f"execution_horizon must be positive, got {execution_horizon}")
    actual_steps = min(steps, trajectory_length)
    indices: set[int] = set()
    for modality in ("video", "mask"):
        config = modality_configs.get(modality)
        if config is None:
            continue
        for step_count in range(0, actual_steps, execution_horizon):
            indices.update(step_count + int(delta) for delta in config.delta_indices)
    ordered = np.asarray(sorted(indices), dtype=np.int64)
    invalid = ordered[(ordered < 0) | (ordered >= trajectory_length)]
    if invalid.size:
        raise IndexError(
            f"Evaluation requires visual frame indices {invalid.tolist()} outside trajectory "
            f"range 0..{trajectory_length - 1}"
        )
    return ordered


def batch_policy_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Join already-batched single observations without changing their order."""

    if not observations:
        raise ValueError("Cannot batch an empty observation list")
    batched: dict[str, Any] = {}
    for modality, first_values in observations[0].items():
        batched[modality] = {}
        for key, first_value in first_values.items():
            values = [observation[modality][key] for observation in observations]
            if isinstance(first_value, np.ndarray):
                batched[modality][key] = np.concatenate(values, axis=0)
            elif isinstance(first_value, list):
                batched[modality][key] = [item for value in values for item in value]
            else:
                raise TypeError(
                    f"Cannot batch observation {modality}.{key} of type "
                    f"{type(first_value).__name__}"
                )
    return batched


def inference_noise_seed(base_seed: int, trajectory_id: int, step_count: int) -> int:
    """Derive a stable per-observation seed independent of inference batching."""

    payload = f"{int(base_seed)}:{int(trajectory_id)}:{int(step_count)}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63)


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    execution_horizon=16,
    save_plot_path=None,
    progress_label: str | None = None,
    trajectory: pd.DataFrame | None = None,
    inference_batch_size: int = 1,
    inference_seed: int | None = None,
):
    if inference_batch_size <= 0:
        raise ValueError(f"inference_batch_size must be positive, got {inference_batch_size}")
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id] if trajectory is None else trajectory
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )
    progress_label = progress_label or f"trajectory {traj_id}"
    inference_steps = list(range(0, actual_steps, execution_horizon))
    total_inferences = len(inference_steps)
    total_requests = (total_inferences + inference_batch_size - 1) // inference_batch_size
    trajectory_started_at = time.perf_counter()

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    # Fail fast if the open-loop stride doesn't fit the model's predicted chunk
    # (also rejects a non-contiguous action window, which the linear indexing
    # below would silently mis-execute).
    PolicyHorizonSpec.from_modality_config(
        loader.modality_configs, n_action_steps=execution_horizon
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    for request_index, batch_start in enumerate(
        range(0, total_inferences, inference_batch_size), start=1
    ):
        batch_steps = inference_steps[batch_start : batch_start + inference_batch_size]
        observations = []
        for step_count in batch_steps:
            data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
            obs = {}
            for k, v in data_point.states.items():
                obs[f"state.{k}"] = v  # (T, D)
            for k, v in data_point.images.items():
                obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
            for language_key in loader.modality_configs["language"].modality_keys:
                obs[language_key] = data_point.text
            observations.append(parse_observation_gr00t(obs, loader.modality_configs))

        parsed_obs = batch_policy_observations(observations)
        inference_started_at = time.perf_counter()
        if inference_seed is None:
            _action_chunk, _ = policy.get_action(parsed_obs)
        else:
            noise_seeds = [
                inference_noise_seed(inference_seed, traj_id, step_count)
                for step_count in batch_steps
            ]
            _action_chunk, _ = policy.get_action(
                parsed_obs,
                options={
                    "inference_mode": "synchronous",
                    "noise_seeds": noise_seeds,
                },
            )
        inference_seconds = time.perf_counter() - inference_started_at
        elapsed_seconds = time.perf_counter() - trajectory_started_at
        completed_frame = min(batch_steps[-1] + execution_horizon, actual_steps)
        remaining_requests = total_requests - request_index
        estimated_remaining_seconds = elapsed_seconds / request_index * remaining_requests
        if inference_batch_size == 1:
            logging.info(
                "[%s] inference %d/%d complete: frames %d-%d/%d (%.1f%%), "
                "request %.3fs, episode elapsed %.1fs, episode ETA %.1fs",
                progress_label,
                batch_start + 1,
                total_inferences,
                batch_steps[0] + 1,
                completed_frame,
                actual_steps,
                100.0 * completed_frame / actual_steps,
                inference_seconds,
                elapsed_seconds,
                estimated_remaining_seconds,
            )
        else:
            logging.info(
                "[%s] inference batch %d/%d complete: observations %d-%d/%d, "
                "frames through %d/%d (%.1f%%), request %.3fs, episode elapsed %.1fs, "
                "episode ETA %.1fs",
                progress_label,
                request_index,
                total_requests,
                batch_start + 1,
                batch_start + len(batch_steps),
                total_inferences,
                completed_frame,
                actual_steps,
                100.0 * completed_frame / actual_steps,
                inference_seconds,
                elapsed_seconds,
                estimated_remaining_seconds,
            )

        for batch_index, _step_count in enumerate(batch_steps):
            action_chunk = parse_action_gr00t(_action_chunk, batch_index=batch_index)
            for j in range(execution_horizon):
                # The np.atleast_1d handles scalar action groups.
                concat_pred_action = np.concatenate(
                    [
                        np.atleast_1d(action_chunk[f"action.{key}"][j])
                        for key in action_keys
                    ],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        execution_horizon=execution_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    execution_horizon: int = 16
    """How many steps of each predicted action chunk to execute before re-planning
    (must be <= the model's predicted chunk length)."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""


def main(args: ArgsConfig):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Apply --denoising-steps: the action head reads num_inference_timesteps
        # at sampling time.
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        logging.info(f"Using {args.denoising_steps} denoising steps")
    else:
        policy = PolicyClient(host=args.host, port=args.port)
        if args.denoising_steps != ArgsConfig.denoising_steps:
            logging.warning(
                "--denoising-steps=%d is ignored when running against a remote "
                "policy server; set the denoising steps on the server "
                "(run_gr00t_server.py) instead.",
                args.denoising_steps,
            )

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            execution_horizon=args.execution_horizon,
            save_plot_path=args.save_plot_path,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    if migrate_deprecated_action_horizon_argv():
        logging.warning("--action-horizon is deprecated; use --execution-horizon.")
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
