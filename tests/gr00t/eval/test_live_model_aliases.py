"""CPU-only checks of pinned trial identities; no model loading or server bind."""
import json
from pathlib import Path

import pytest

from gr00t.eval.live_model_aliases import (
    ALIASES, BLIND_ALIASES, DEFAULT_BLIND_STUDY, blind_manifest, resolve_live_model,
)


def _root(tmp_path: Path) -> Path:
    models = {}
    for alias in ALIASES:
        relative = f"{alias}/checkpoint-30000"
        path = tmp_path / relative
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "processor_config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"fixture")
        models[alias] = relative
    (tmp_path / "live_models.json").write_text(json.dumps({"schema_version": 1, "models": models}))
    return tmp_path


def test_visible_aliases_and_blind_study_stay_pinned(tmp_path):
    root = _root(tmp_path)
    for alias in ALIASES:
        assert resolve_live_model(model=alias, root=root).checkpoint.parent.name == alias
    original, directory = blind_manifest(root, "trial-1")
    again, _ = blind_manifest(root, "trial-1")
    assert original == again
    assert set(original["models"]) == {"A", "B"}
    assert {item["alias"] for item in original["models"].values()} == set(BLIND_ALIASES)
    assert (directory / "mapping.json").stat().st_mode & 0o777 == 0o600
    for label in "AB":
        public = resolve_live_model(blind=label, root=root, study="trial-1").identity
        assert public["blinded"] and "model_path" not in public
        assert all(alias not in public["model_name"] for alias in ALIASES)
    with pytest.raises(ValueError, match="A or B"):
        resolve_live_model(blind="C", root=root, study="trial-1")


def test_blind_checkpoint_change_blocks_reuse(tmp_path):
    root = _root(tmp_path)
    manifest, _ = blind_manifest(root, "trial-1")
    selected = root / manifest["models"]["A"]["checkpoint"] / "config.json"
    selected.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed"):
        resolve_live_model(blind="A", root=root, study="trial-1")


def test_new_normals_study_preserves_completed_turbo_mapping(tmp_path):
    root = _root(tmp_path)
    old, old_directory = blind_manifest(root, "live-ab-01")
    original_bytes = (old_directory / "mapping.json").read_bytes()
    new, _ = blind_manifest(root, DEFAULT_BLIND_STUDY)
    assert {item["alias"] for item in old["models"].values()} == {"rgb", "fusion"}
    assert {item["alias"] for item in new["models"].values()} == {"rgb", "separate"}
    assert new["study_id"] != old["study_id"]
    for study, manifest in (("live-ab-01", old), (DEFAULT_BLIND_STUDY, new)):
        for label in "AB":
            selected = resolve_live_model(blind=label, study=study, root=root)
            assert selected.checkpoint == root / manifest["models"][label]["checkpoint"]
            assert selected.identity["study_id"] == manifest["study_id"]
        assert blind_manifest(root, study)[0] == manifest
    assert (old_directory / "mapping.json").read_bytes() == original_bytes
    assert resolve_live_model(blind="A", root=root).identity["study"] == DEFAULT_BLIND_STUDY


def test_alias_cannot_escape_live_folder(tmp_path):
    root = _root(tmp_path)
    registry = json.loads((root / "live_models.json").read_text())
    registry["models"]["rgb"] = "../other/checkpoint-30000"
    (root / "live_models.json").write_text(json.dumps(registry))
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_live_model(model="rgb", root=root)
