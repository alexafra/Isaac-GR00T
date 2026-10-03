"""The opt-in recipe binds one explicit RGB/geometry contract without legacy drift."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.depth_colormap import FIXED_TURBO_DEPTH_COLORMAP
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, VideoChannelSource
from gr00t.experiment.launch_finetune import (
    configure_vision_input_contract,
    configure_vision_modality_dropout_contract,
)
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import pytest


def _ft(**kwargs):
    return FinetuneConfig("unused-base", "unused-data", "NEW_EMBODIMENT", **kwargs)


def _config(video):
    return SimpleNamespace(
        model=Gr00tN1d7Config(),
        data=SimpleNamespace(
            modality_configs={
                "new": {"video": video},
                "old": {"video": ModalityConfig([0], ["ego_view"])},
            }
        ),
    )


@pytest.mark.parametrize("key", ["depth_gray_view", "surface_normals_view"])
@pytest.mark.parametrize("mode", ["early", "pre", "post", "separate"])
def test_dropout_maps_all_geometry_representations_and_fusion_stages(key, mode):
    kwargs = {}
    if mode == "early":
        kwargs["channel_fusion"] = [
            VideoChannelSource("ego_view", (0, 1, 2)),
            VideoChannelSource(key, (0,) if key == "depth_gray_view" else (0, 1, 2)),
        ]
    elif mode in {"pre", "post"}:
        kwargs.update(
            post_vision_fusion=True,
            post_vision_fusion_stage=f"{mode}_vision_language_adapter",
        )
    video = ModalityConfig([0, -1], ["ego_view", key], **kwargs)
    config = _config(video)
    configure_vision_input_contract(config, "new", video, None)
    configure_vision_modality_dropout_contract(
        config,
        "new",
        video,
        _ft(vision_modality_dropout_rgb_prob=0.05, vision_modality_dropout_geometry_prob=0.05),
    )
    assert list(config.data.modality_configs) == ["new"]
    assert config.model.vision_modality_dropout_source_keys == ["ego_view", key]
    assert config.model.vision_modality_dropout_rgb_prob == 0.05
    assert config.model.vision_modality_dropout_geometry_prob == 0.05
    assert config.model.vision_modality_dropout_state_policy == "independent"
    restored = Gr00tN1d7Config.from_dict(config.model.to_dict())
    assert restored.vision_modality_dropout_source_keys == ["ego_view", key]
    assert restored.vision_modality_dropout_rgb_prob == 0.05


@pytest.mark.parametrize("late", [False, True])
def test_turbo_dropout_retains_colormap_contract(late):
    video = ModalityConfig(
        [0],
        ["ego_view", "depth_gray_view"],
        post_vision_fusion=late,
        post_vision_fusion_stage="pre_vision_language_adapter" if late else None,
        depth_colormap=FIXED_TURBO_DEPTH_COLORMAP,
    )
    config = _config(video)
    configure_vision_input_contract(config, "new", video, None)
    before = config.model.vision_depth_colormap_lut_sha256
    configure_vision_modality_dropout_contract(
        config, "new", video, _ft(vision_modality_dropout_rgb_prob=0.05)
    )
    assert config.model.vision_depth_colormap == FIXED_TURBO_DEPTH_COLORMAP
    assert config.model.vision_depth_colormap_lut_sha256 == before


def test_turbo_early_fusion_recipe_constructs_dropout_processor(monkeypatch, tmp_path):
    config_name = "g1_inspire_head_6_channel_turbo_depth_fusion_config"
    config_dir = Path(__file__).resolve().parents[3] / "examples" / "UnitreeG1"
    monkeypatch.syspath_prepend(str(config_dir))
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    previous = MODALITY_CONFIGS.pop(tag, None)
    try:
        spec = importlib.util.spec_from_file_location(config_name, config_dir / f"{config_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modalities = {tag: getattr(module, config_name)}
    finally:
        MODALITY_CONFIGS.pop(tag, None)
        if previous is not None:
            MODALITY_CONFIGS[tag] = previous
    video = modalities[tag]["video"]
    config = SimpleNamespace(
        model=Gr00tN1d7Config(), data=SimpleNamespace(modality_configs=modalities)
    )
    configure_vision_input_contract(config, tag, video, "rgb_mean")
    configure_vision_modality_dropout_contract(
        config,
        tag,
        video,
        _ft(
            vision_modality_dropout_rgb_prob=0.05,
            vision_modality_dropout_geometry_prob=0.05,
            vision_modality_dropout_state_policy="independent",
        ),
    )
    assert config.model.vision_input_channels == 6
    assert config.model.vision_patch_embed_init == "rgb_mean"
    assert config.model.vision_depth_colormap == FIXED_TURBO_DEPTH_COLORMAP
    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor", return_value=MagicMock()
    ):
        processor = Gr00tN1d7Processor(
            modality_configs=config.data.modality_configs,
            max_state_dim=26,
            max_action_dim=26,
            max_action_horizon=32,
            image_crop_size=[8, 8],
            image_target_size=[8, 8],
            use_albumentations=True,
            state_dropout_prob=0.2,
            vision_modality_dropout_rgb_prob=config.model.vision_modality_dropout_rgb_prob,
            vision_modality_dropout_geometry_prob=config.model.vision_modality_dropout_geometry_prob,
            vision_modality_dropout_state_policy=config.model.vision_modality_dropout_state_policy,
            vision_modality_dropout_source_keys=config.model.vision_modality_dropout_source_keys,
        )
        processor.save_pretrained(tmp_path)
        restored = Gr00tN1d7Processor.from_pretrained(tmp_path)
    assert restored.vision_modality_dropout_rgb_prob == 0.05
    assert restored.vision_modality_dropout_geometry_prob == 0.05
    assert restored.vision_modality_dropout_state_policy == "independent"
    assert restored.vision_modality_dropout_source_keys == ["ego_view", "depth_gray_view"]
    assert restored.modality_configs[tag]["video"].vision_input_channels == 6
    assert restored.modality_configs[tag]["video"].depth_colormap == FIXED_TURBO_DEPTH_COLORMAP
    assert restored.collator.vision_input_channels == 6


@pytest.mark.parametrize("keys", [["ego_view"], ["ego_view", "depth_gray_view"]])
def test_zero_default_retains_pretrained_embodiments_and_no_source_contract(keys):
    video = ModalityConfig([0], keys)
    config = _config(video)
    configure_vision_input_contract(config, "new", video, None)
    configure_vision_modality_dropout_contract(config, "new", video, _ft())
    assert list(config.data.modality_configs) == ["new", "old"]
    assert config.model.vision_modality_dropout_source_keys == []
    assert config.model.vision_modality_dropout_rgb_prob == 0.0
    assert config.model.vision_modality_dropout_geometry_prob == 0.0


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["rgb", "geometry"])
def test_cli_rejects_nonfinite_or_out_of_range_probabilities(field, value):
    with pytest.raises(ValueError, match="finite and in"):
        _ft(**{f"vision_modality_dropout_{field}_prob": value})


def test_cli_global_state_first_probability_bound_and_independent_alternative():
    with pytest.raises(ValueError, match="eligible probability"):
        _ft(
            vision_modality_dropout_rgb_prob=0.65,
            vision_modality_dropout_state_policy="state_first",
        )
    _ft(
        vision_modality_dropout_rgb_prob=0.32,
        vision_modality_dropout_geometry_prob=0.32,
        vision_modality_dropout_state_policy="state_first",
    )
    _ft(vision_modality_dropout_rgb_prob=0.7, vision_modality_dropout_state_policy="independent")
    with pytest.raises(ValueError, match="sum to"):
        _ft(vision_modality_dropout_rgb_prob=0.6, vision_modality_dropout_geometry_prob=0.5)
    with pytest.raises(ValueError, match="state_first or independent"):
        _ft(vision_modality_dropout_state_policy="typo")


@pytest.mark.parametrize(
    "keys",
    [
        ["ego_view"],
        ["depth_gray_view", "ego_view"],
        ["ego_view", "wrist"],
        ["ego_view", "depth_gray_view", "surface_normals_view"],
    ],
)
def test_active_dropout_rejects_ambiguous_or_rgb_only_inputs(keys):
    video = ModalityConfig([0], keys)
    with pytest.raises(ValueError, match="exactly ordered paired"):
        configure_vision_modality_dropout_contract(
            _config(video), "new", video, _ft(vision_modality_dropout_rgb_prob=0.05)
        )


def test_active_dropout_rejects_empty_frames_and_wrong_depth_channel():
    video = ModalityConfig([], ["ego_view", "depth_gray_view"])
    with pytest.raises(ValueError, match="positive frame"):
        configure_vision_modality_dropout_contract(
            _config(video), "new", video, _ft(vision_modality_dropout_rgb_prob=0.05)
        )
    video = ModalityConfig(
        [0],
        ["ego_view", "depth_gray_view"],
        channel_fusion=[
            VideoChannelSource("ego_view", (0, 1, 2)),
            VideoChannelSource("depth_gray_view", (1,)),
        ],
    )
    with pytest.raises(ValueError, match="RGB channels"):
        configure_vision_modality_dropout_contract(
            _config(video), "new", video, _ft(vision_modality_dropout_rgb_prob=0.05)
        )
