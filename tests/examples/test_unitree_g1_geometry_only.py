import importlib.util
from pathlib import Path

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_config(filename: str, variable: str):
    path = REPO_ROOT / "examples" / "UnitreeG1" / filename
    spec = importlib.util.spec_from_file_location(variable, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    previous = MODALITY_CONFIGS.pop(tag, None)
    try:
        spec.loader.exec_module(module)
        return getattr(module, variable)
    finally:
        MODALITY_CONFIGS.pop(tag, None)
        if previous is not None:
            MODALITY_CONFIGS[tag] = previous


def test_depth_gray_only_config_has_no_colour_input():
    config = _load_config(
        "g1_dex3_head_depth_gray_only_config.py",
        "g1_dex3_head_depth_gray_only_config",
    )

    assert config["video"].delta_indices == [0]
    assert config["video"].modality_keys == ["depth_gray_view"]
    assert config["video"].channel_fusion is None
    assert config["action"].delta_indices == list(range(32))


def test_surface_normals_only_config_has_no_colour_input():
    config = _load_config(
        "g1_dex3_head_surface_normals_only_config.py",
        "g1_dex3_head_surface_normals_only_config",
    )

    assert config["video"].delta_indices == [0]
    assert config["video"].modality_keys == ["surface_normals_view"]
    assert config["video"].channel_fusion is None
    assert config["action"].delta_indices == list(range(32))
