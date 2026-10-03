"""Pinned live-model aliases and opt-in, persistent operator blinding.

The private manifest is an audit record, not a security boundary against its owner.
Never print the manifest unless the operator deliberately asks to unblind.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
import uuid

DEFAULT_LIVE_ROOT = Path("/home/alex/Development/Models/inspire_live_test")
DEFAULT_DATASET = "/home/alex/Development/Datasets/lerobot2/inspire/final_1090eps_20260920_nondark_views/combined/train"
ALIASES = ("rgb", "separate", "fusion")
BLIND_ALIASES = ("rgb", "separate")
BLIND_LABELS = "AB"
DEFAULT_BLIND_STUDY = "live-ab-02"
# Keep the completed RGB/Turbo study usable with its original pinned identities.
BLIND_STUDY_ALIASES = {"live-ab-01": ("rgb", "fusion")}


@dataclass(frozen=True)
class LiveSelection:
    checkpoint: Path
    identity: dict
    private_directory: Path | None = None


def _checkpoint(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not re.fullmatch(r"checkpoint-\d+", path.name):
        raise ValueError("Alias must resolve to a checkpoint inside the live-model folder")
    for name in ("config.json", "processor_config.json"):
        if not (path / name).is_file():
            raise ValueError("Live checkpoint is missing configuration")
    weights = list(path.glob("*.safetensors"))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError("Live checkpoint is missing weights")
    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards <= {p.name for p in weights}:
            raise ValueError("Live checkpoint has missing weight shards")
    return path


def _fingerprint(path: Path) -> dict:
    """Small-file hashes plus shard sizes/mtimes; does not scan GBs at launch."""
    configs = sorted(set(path.glob("*.json")) | set((path / "processor").glob("*.json")))
    return {
        "config_sha256": {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in configs},
        "weights": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(path.glob("*.safetensors"))},
    }


def _registry(root: Path) -> dict:
    registry = json.loads((root / "live_models.json").read_text())
    if registry.get("schema_version") != 1 or set(registry["models"]) != set(ALIASES):
        raise ValueError("Expected a v1 registry with rgb, separate and fusion")
    return registry["models"]


def blind_manifest(root: Path, study: str) -> tuple[dict, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", study):
        raise ValueError("Study name must be 1..64 letters, digits, underscores or hyphens")
    aliases = BLIND_STUDY_ALIASES.get(study, BLIND_ALIASES)
    directory = root / ".blind_studies" / study
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    manifest_path = directory / "mapping.json"
    # Serialize concurrent launches; existing studies never reshuffle or track edits to aliases.
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not manifest_path.exists():
            registry = _registry(root)
            entries = []
            for alias in aliases:
                checkpoint = _checkpoint(root, registry[alias])
                entries.append({"alias": alias, "checkpoint": str(checkpoint.relative_to(root.resolve())),
                                "fingerprint": _fingerprint(checkpoint)})
            secrets.SystemRandom().shuffle(entries)
            manifest = {"schema_version": 2, "study": study, "study_id": uuid.uuid4().hex,
                        "models": dict(zip(BLIND_LABELS, entries))}
            fd, temporary = tempfile.mkstemp(prefix=".mapping-", dir=directory)
            try:
                with os.fdopen(fd, "w") as file:
                    json.dump(manifest, file, indent=2)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, manifest_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema_version") != 2 or manifest.get("study") != study
                or set(manifest["models"]) != set(BLIND_LABELS)
                or {item["alias"] for item in manifest["models"].values()} != set(aliases)):
            raise ValueError("Saved study does not match the requested A/B model pair; refusing to reshuffle")
    return manifest, directory


def resolve_live_model(*, model: str | None = None, blind: str | None = None,
                       study: str = DEFAULT_BLIND_STUDY, root: Path = DEFAULT_LIVE_ROOT) -> LiveSelection:
    root = root.expanduser().resolve(strict=True)
    if (model is None) == (blind is None):
        raise ValueError("Choose either --model or --blind, not both")
    if model is not None:
        alias = model.strip().lower().replace(" ", "-").replace("_", "-")
        if alias not in ALIASES:
            raise ValueError("Model alias must be rgb, separate or fusion")
        checkpoint = _checkpoint(root, _registry(root)[alias])
        return LiveSelection(checkpoint, {"model_name": checkpoint.parent.name,
                             "checkpoint": checkpoint.name, "model_path": str(checkpoint), "alias": alias})
    label = blind.upper()
    if label not in BLIND_LABELS or len(label) != 1:
        raise ValueError("Blind model must be A or B")
    manifest, directory = blind_manifest(root, study)
    entry = manifest["models"][label]
    checkpoint = _checkpoint(root, entry["checkpoint"])
    if _fingerprint(checkpoint) != entry["fingerprint"]:
        raise ValueError("Pinned blind checkpoint changed; restore it or explicitly create a new study")
    return LiveSelection(checkpoint, {"model_name": f"blind-{study}-{label}", "checkpoint": "blinded",
                         "blinded": True, "blind_label": label, "study": study,
                         "study_id": manifest["study_id"]}, directory)


@contextmanager
def private_server_output(directory: Path):
    """Capture Python/native loader output; return a public, identity-free announcer."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    fd, _ = tempfile.mkstemp(prefix="server-", suffix=".log", dir=directory)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        yield lambda message: os.write(saved[0], (message + "\n").encode())
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for descriptor in (*saved, fd):
            os.close(descriptor)
