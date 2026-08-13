import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "examples" / "UnitreeG1" / "g1_dex3_head_3_channel_surface_normals_config.py"
)


def _load_surface_normals_config():
    spec = importlib.util.spec_from_file_location("unitree_surface_normals_config", CONFIG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.g1_dex3_head_3_channel_surface_normals_config


def test_surface_normals_config_is_rgb_plus_normals_with_horizon_32():
    config = _load_surface_normals_config()

    assert config["video"].delta_indices == [0]
    assert config["video"].modality_keys == ["ego_view", "surface_normals_view"]
    assert config["action"].delta_indices == list(range(32))


def test_unitree_modality_layout_maps_surface_normals_feature():
    with open(REPO_ROOT / "examples" / "UnitreeG1" / "modality.json") as file:
        modality = json.load(file)

    assert modality["video"]["surface_normals_view"] == {
        "original_key": "observation.images.surface_normals_view"
    }
