# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
MONITOR_SCRIPT = (
    REPO_ROOT / "scripts/monitor_inspire_e2_ftp_colour_only_20k_20260912.sh"
)


def _monitor_environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GROOT_MONITOR_OUTPUT_DIR": str(tmp_path / "output"),
            "GROOT_MONITOR_TRAIN_LOG": str(tmp_path / "train.log"),
            "GROOT_MONITOR_LOG": str(tmp_path / "monitor.log"),
            "GROOT_MONITOR_PYTHON_BIN": os.environ.get("PYTHON", "python3"),
            "GROOT_MONITOR_LOW_GPU_SAMPLE_INTERVAL_SECONDS": "0",
        }
    )
    return environment


def _run_monitor_function(
    tmp_path: Path,
    body: str,
    *,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = _monitor_environment(tmp_path)
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        ["bash", "-c", f'source "{MONITOR_SCRIPT}"\n{body}'],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=30,
    )


def _write_complete_checkpoint(output_dir: Path, step: int) -> Path:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}})
    )
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"model")
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / filename).write_bytes(b"state")
    return checkpoint


def test_resume_quarantines_newer_incomplete_checkpoint_without_deleting_it(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    selected = _write_complete_checkpoint(output_dir, 5000)
    older_incomplete = output_dir / "checkpoint-3000"
    older_incomplete.mkdir()
    newer_incomplete = output_dir / "checkpoint-10000"
    newer_incomplete.mkdir()
    (newer_incomplete / "partial-shard.safetensors").write_bytes(b"preserve me")

    result = _run_monitor_function(
        tmp_path,
        'latest="$(latest_complete_checkpoint)"\n'
        '[[ "${latest}" == "${OUTPUT_DIR}/checkpoint-5000" ]]\n'
        'quarantine_incomplete_newer_checkpoints "${latest}"',
    )

    assert result.returncode == 0, result.stderr
    assert selected.is_dir()
    assert older_incomplete.is_dir(), "Only incomplete checkpoints newer than the resume point move"
    assert not newer_incomplete.exists()
    quarantined = list(
        Path(f"{output_dir}.checkpoint_quarantine").glob(
            "????????T??????Z-*/checkpoint-10000/partial-shard.safetensors"
        )
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"preserve me"
    assert "recoverable quarantine" in (tmp_path / "monitor.log").read_text()


def test_failure_scan_excludes_signature_from_previous_attempt(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "run_mode=fresh\n"
        "CUDA out of memory\n"
        "run_mode=resume\n"
        "ValueError: unrelated current failure\n"
    )

    result = _run_monitor_function(
        tmp_path,
        'offset="$(current_attempt_log_offset)"\n'
        'failure_tail_from_offset "${offset}"',
    )

    assert result.returncode == 0, result.stderr
    assert "run_mode=resume" in result.stdout
    assert "unrelated current failure" in result.stdout
    assert "CUDA out of memory" not in result.stdout


def test_explicit_retry_boundary_excludes_old_signature_without_new_run_marker(
    tmp_path: Path,
) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text("run_mode=fresh\nOutOfMemoryError\n")

    result = _run_monitor_function(
        tmp_path,
        'offset="$(log_size_bytes)"\n'
        'printf "%s\\n" "preflight failed before run marker" >>"${TRAIN_LOG}"\n'
        'failure_tail_from_offset "${offset}"',
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "preflight failed before run marker\n"


def test_stall_requires_old_files_and_repeated_low_gpu_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    checkpoint = output_dir / "checkpoint-5000"
    checkpoint.mkdir(parents=True)
    train_log = tmp_path / "train.log"
    train_log.write_text("progress\n")
    checkpoint_file = checkpoint / "partial-model.safetensors"
    checkpoint_file.write_bytes(b"partial")

    now = int(time.time())
    old = now - 1801
    os.utime(train_log, (old, old))
    os.utime(checkpoint_file, (old, old))
    os.utime(checkpoint, (old, old))
    os.utime(output_dir, (old, old))

    gpu_calls = tmp_path / "gpu-calls"
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_nvidia_smi.write_text(
        '#!/usr/bin/env bash\nprintf "call\\n" >>"${GPU_CALLS_FILE}"\nprintf "0\\n"\n'
    )
    fake_nvidia_smi.chmod(0o755)

    result = _run_monitor_function(
        tmp_path,
        f'training_looks_stalled "{now}"\n'
        'printf "%s|%s\\n" "${LAST_INACTIVITY_SECONDS}" "${LOW_GPU_SAMPLES}"',
        extra_environment={
            "GPU_CALLS_FILE": str(gpu_calls),
            "GROOT_MONITOR_NVIDIA_SMI_BIN": str(fake_nvidia_smi),
        },
    )

    assert result.returncode == 0, result.stderr
    inactivity, samples = result.stdout.strip().split("|")
    assert int(inactivity) >= 1800
    assert samples == "0%,0%,0%"
    assert gpu_calls.read_text().splitlines() == ["call", "call", "call"]

    os.utime(train_log, (now, now))
    recent_result = _run_monitor_function(
        tmp_path,
        f'training_looks_stalled "{now}"',
        extra_environment={
            "GPU_CALLS_FILE": str(gpu_calls),
            "GROOT_MONITOR_NVIDIA_SMI_BIN": str(fake_nvidia_smi),
        },
    )
    assert recent_result.returncode != 0
    assert gpu_calls.read_text().splitlines() == ["call", "call", "call"]


def test_old_files_do_not_trigger_stall_when_gpu_is_busy(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text("stale progress\n")
    now = int(time.time())
    old = now - 3600
    os.utime(train_log, (old, old))

    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_nvidia_smi.write_text('#!/usr/bin/env bash\nprintf "97\\n"\n')
    fake_nvidia_smi.chmod(0o755)

    result = _run_monitor_function(
        tmp_path,
        f'training_looks_stalled "{now}"',
        extra_environment={"GROOT_MONITOR_NVIDIA_SMI_BIN": str(fake_nvidia_smi)},
    )

    assert result.returncode != 0


def test_inactive_oom_recovery_quarantines_before_mocked_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_complete_checkpoint(output_dir, 5000)
    incomplete = output_dir / "checkpoint-10000"
    incomplete.mkdir()
    (incomplete / "partial").write_text("keep")
    (tmp_path / "train.log").write_text("run_mode=fresh\nCUDA out of memory\n")

    fake_systemctl = tmp_path / "systemctl"
    fake_systemctl.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"is-active"* ]]; then exit 1; fi\n'
        "exit 0\n"
    )
    fake_systemctl.chmod(0o755)
    systemd_run_call = tmp_path / "systemd-run-call"
    fake_systemd_run = tmp_path / "systemd-run"
    fake_systemd_run.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >"${SYSTEMD_RUN_CALL_FILE}"\n'
    )
    fake_systemd_run.chmod(0o755)
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_nvidia_smi.write_text('#!/usr/bin/env bash\nprintf "97\\n"\n')
    fake_nvidia_smi.chmod(0o755)

    result = _run_monitor_function(
        tmp_path,
        "main",
        extra_environment={
            "GROOT_MONITOR_CHECK_COUNT": "1",
            "GROOT_MONITOR_CHECK_INTERVAL_SECONDS": "0",
            "GROOT_MONITOR_RUNNER": "/safe/mock-runner.sh",
            "GROOT_MONITOR_SYSTEMCTL_BIN": str(fake_systemctl),
            "GROOT_MONITOR_SYSTEMD_RUN_BIN": str(fake_systemd_run),
            "GROOT_MONITOR_NVIDIA_SMI_BIN": str(fake_nvidia_smi),
            "SYSTEMD_RUN_CALL_FILE": str(systemd_run_call),
        },
    )

    assert result.returncode == 0, result.stderr
    assert not incomplete.exists()
    quarantined = list(
        Path(f"{output_dir}.checkpoint_quarantine").glob("*/checkpoint-10000/partial")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == "keep"
    launch = systemd_run_call.read_text()
    assert "GROOT_GLOBAL_BATCH_SIZE=16" in launch
    assert "GROOT_GRADIENT_ACCUMULATION_STEPS=2" in launch
    assert "/usr/bin/bash /safe/mock-runner.sh resume" in launch
    monitor_log = (tmp_path / "monitor.log").read_text()
    assert "Resume safety:" in monitor_log
    assert "Retry launched:" in monitor_log
