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

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro
import yaml


DEFAULT_MODEL_SERVER_PORT = 5555


def _load_deployment_dataset_contract(dataset_path: Path) -> dict:
    """Load the hardware-facing parts of a LeRobot dataset contract."""

    info_path = dataset_path / "meta" / "info.json"
    try:
        with open(info_path) as file:
            info = json.load(file)
        features = info["features"]
        state_names = features["observation.state"]["names"]
        action_names = features["action"]["names"]
        image_shape = features["observation.images.ego_view"]["shape"]
        if len(state_names) == 1 and isinstance(state_names[0], list):
            state_names = state_names[0]
        if len(action_names) == 1 and isinstance(action_names[0], list):
            action_names = action_names[0]
        contract = {
            "robot_type": str(info["robot_type"]),
            "fps": float(info["fps"]),
            "observation_state_names": list(state_names),
            "action_names": list(action_names),
            "ego_view_shape": list(image_shape),
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot build a deployment contract from {info_path}: {exc}") from exc
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    contract["sha256"] = hashlib.sha256(canonical).hexdigest()
    return contract


def _verify_checkpoint_dataset_path(
    model_path: Path, embodiment_tag: str, deployment_dataset_path: Path
) -> None:
    """Bind an asserted deployment dataset to the checkpoint's saved training config."""

    config_path = model_path / "experiment_cfg" / "config.yaml"
    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
        datasets = config["data"]["datasets"]
        recorded = [
            Path(path).resolve()
            for dataset in datasets
            if dataset.get("embodiment_tag") == embodiment_tag
            for path in dataset["dataset_paths"]
        ]
    except (FileNotFoundError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Cannot verify deployment data against checkpoint config {config_path}: {exc}"
        ) from exc
    requested = deployment_dataset_path.resolve()
    if recorded != [requested]:
        raise ValueError(
            "Deployment dataset must be the single training dataset recorded for "
            f"embodiment '{embodiment_tag}'. Checkpoint records {recorded}; got {requested}."
        )


def _load_checkpoint_action_output_contract(model_path: Path, embodiment_tag: str) -> dict:
    """Describe how ``Gr00tPolicy.get_action`` decodes checkpoint outputs."""

    candidates = [
        model_path / "processor_config.json",
        model_path / "processor" / "processor_config.json",
    ]
    config_path = next((path for path in candidates if path.is_file()), candidates[0])
    try:
        with open(config_path) as file:
            processor = json.load(file)["processor_kwargs"]
        use_relative_action = processor["use_relative_action"]
        if not isinstance(use_relative_action, bool):
            raise TypeError(
                "processor_kwargs.use_relative_action must be a boolean, got "
                f"{use_relative_action!r}"
            )
        action = processor["modality_configs"][embodiment_tag]["action"]
        keys = action["modality_keys"]
        configs = action["action_configs"]
        relative_keys = [
            key for key, config in zip(keys, configs, strict=True) if config["rep"] == "RELATIVE"
        ]
        representations = {config["rep"] for config in configs}
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot determine action output semantics from {config_path}: {exc}"
        ) from exc
    supported_representations = representations <= {"RELATIVE", "ABSOLUTE"}
    absolute_output = supported_representations and (use_relative_action or not relative_keys)
    return {
        "semantics": "absolute_joint_position" if absolute_output else "checkpoint_native",
        "use_relative_action": use_relative_action,
        "relative_keys_decoded_to_absolute": relative_keys if use_relative_action else [],
    }


def _load_json_modality_configs(config_path: Path) -> dict[str, ModalityConfig]:
    """Load a JSON file whose values are ModalityConfig field dicts.

    A dataset's ``meta/modality.json`` is a different (data-layout) schema and is
    not accepted here — point such users at a .py config instead of letting the
    ``ModalityConfig(**v)`` unpack raise a bare ``TypeError``.
    """
    with open(config_path, "r") as f:
        raw = json.load(f)
    try:
        return {k: ModalityConfig(**v) for k, v in raw.items()}
    except TypeError as exc:
        raise ValueError(
            f"{config_path} is not a ModalityConfig JSON: each value must hold ModalityConfig "
            f"fields (delta_indices, modality_keys, ...). A dataset's meta/modality.json uses a "
            f"different schema; pass a .py modality config (e.g. examples/SO100/so100_config.py) instead."
        ) from exc


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    deployment_dataset_path: str | None = None
    """Dataset root whose meta/info.json identifies the physical deployment contract"""


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                modality_configs = _load_json_modality_configs(config_path)
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    policy_metadata = {
        "embodiment_tag": config.embodiment_tag.value,
        "embodiment_name": config.embodiment_tag.name,
    }
    if config.deployment_dataset_path is not None:
        deployment_dataset_path = Path(config.deployment_dataset_path)
        if config.model_path is not None:
            _verify_checkpoint_dataset_path(
                Path(config.model_path), config.embodiment_tag.value, deployment_dataset_path
            )
            policy_metadata["action_output_contract"] = _load_checkpoint_action_output_contract(
                Path(config.model_path), config.embodiment_tag.value
            )
        elif Path(config.dataset_path).resolve() != deployment_dataset_path.resolve():
            raise ValueError("Replay policy --deployment-dataset-path must equal --dataset-path")
        policy_metadata["dataset_contract"] = _load_deployment_dataset_contract(
            deployment_dataset_path
        )

    with PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
        policy_metadata=policy_metadata,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
