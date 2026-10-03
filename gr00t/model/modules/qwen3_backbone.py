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

import logging
import math

from huggingface_hub.errors import GatedRepoError
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


_GATED_BACKBONE_HINT = (
    "Cannot download the VLM backbone '{model_name}', which is a gated Hugging "
    "Face repo. Every GR00T checkpoint (including the base nvidia/GR00T-N1.7-3B) "
    "loads this backbone, so both zero-shot inference and finetuning require "
    "access to it. Request access at https://huggingface.co/{model_name} and "
    "authenticate with `hf auth login` (or set the HF_TOKEN environment variable) "
    "before loading a GR00T model."
)


_GATED_MARKERS = ("gated repo", "is restricted", "access to model", "401 client error")

_PRE_VISION_LANGUAGE_ADAPTER = "pre_vision_language_adapter"
_POST_VISION_LANGUAGE_ADAPTER = "post_vision_language_adapter"
_VISION_FUSION_STAGES = {
    _PRE_VISION_LANGUAGE_ADAPTER,
    _POST_VISION_LANGUAGE_ADAPTER,
}


def _is_gated_repo_error(exc: BaseException) -> bool:
    """True if ``exc`` (or any exception it wraps) is a gated/forbidden HF download.

    transformers may raise ``GatedRepoError`` directly or wrap it (as
    ``__cause__``/``__context__``) inside an ``OSError`` whose own message omits
    the gated markers, so we walk the chain.
    """
    cur: BaseException | None = exc
    for _ in range(10):  # bounded walk guards against cyclic exception chains
        if cur is None:
            break
        if isinstance(cur, GatedRepoError) or any(m in str(cur).lower() for m in _GATED_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _real_inference_device(module: torch.nn.Module) -> torch.device:
    """Best-effort real (non-meta) device backing a module, defaulting to CPU.

    RoPE buffers must be rebuilt on the device holding the loaded weights, never
    on the ``meta`` device that may be the active default during nested load.
    """
    for tensor in module.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers():
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def recompute_vision_rotary_inv_freq(
    rotary: torch.nn.Module, head_dim_half: int, device: torch.device
) -> torch.Tensor:
    """Re-derive Qwen3-VL's vision RoPE ``inv_freq`` via the module's own class.

    Reusing ``type(rotary)(...)`` keeps the analytic formula owned by Transformers;
    the real-device context makes it correct even under a ``meta`` default device.
    """
    with torch.device(device):
        fresh = type(rotary)(head_dim_half)
    return fresh.inv_freq.detach().to(device=device, dtype=torch.float32)


def recompute_text_rotary_inv_freq(
    rotary: torch.nn.Module, config, device: torch.device
) -> tuple[torch.Tensor, float]:
    """Re-derive Qwen3-VL's text RoPE ``inv_freq`` via the module's own class.

    Delegates to the module's constructor so the configured ``rope_init_fn``
    (default / scaled / dynamic) stays the single source of truth; returns the
    FP32 ``inv_freq`` and its ``attention_scaling``.
    """
    with torch.device(device):
        fresh = type(rotary)(config=config, device=device)
    inv_freq = fresh.inv_freq.detach().to(device=device, dtype=torch.float32)
    attention_scaling = float(getattr(fresh, "attention_scaling", 1.0))
    return inv_freq, attention_scaling


def _assign_inv_freq(
    rotary: torch.nn.Module, name: str, value: torch.Tensor, *, persistent: bool
) -> bool:
    """Write ``value`` onto ``rotary.<name>`` if it differs; return whether it changed.

    Returns ``True`` when the buffer/attribute was (re)written, ``False`` when the
    existing value already matched (idempotent no-op). This boolean lets tests
    assert that a corrupted buffer is actually repaired rather than silently
    skipped.
    """
    current = getattr(rotary, name, None)
    if (
        isinstance(current, torch.Tensor)
        and current.device.type != "meta"
        and current.device == value.device
        and current.shape == value.shape
        and current.dtype == value.dtype
        and torch.equal(current, value)
    ):
        return False
    if name in rotary._buffers:
        rotary.register_buffer(name, value, persistent=persistent)
    else:
        setattr(rotary, name, value)
    return True


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        tune_vision_patch_embed: bool = False,  # earlyfusion
        vision_input_channels: int = 3,  # earlyfusion
        vision_patch_embed_init: str = "original_rgb",  # earlyfusion
        vision_fusion_mode: str = "none",
        vision_fusion_version: int = 1,
        vision_fusion_stage: str = _PRE_VISION_LANGUAGE_ADAPTER,
        late_fusion_language_readout: str = "post_final_rmsnorm",
        vision_fusion_adapter_type: str = "concat_linear",
        vision_fusion_adapter_bias: bool = True,
        vision_fusion_adapter_bias_init: str = "zeros",
        vision_fusion_adapter_count: int = 4,
        vision_fusion_deepstack_layers: list[int] | None = None,
        vision_fusion_adapter_input_dim: int = 2048,
        vision_fusion_adapter_output_dim: int = 1024,
        vision_fusion_source_keys: list[str] | None = None,
        vision_fusion_adapter_init: str = "rgb50_geo50",
        vision_fusion_rgb_init_scale: float = 0.5,
        vision_fusion_geometry_init_scale: float = 0.5,
        tune_vision_fusion_adapters: bool = True,
        vision_modality_dropout_rgb_prob: float = 0.0,
        vision_modality_dropout_geometry_prob: float = 0.0,
        vision_modality_dropout_source_keys: list[str] | None = None,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()
        if late_fusion_language_readout not in ("post_final_rmsnorm", "pre_final_rmsnorm"):
            raise ValueError(f"Unsupported late-fusion language readout: {late_fusion_language_readout!r}")
        self.late_fusion_language_readout = late_fusion_language_readout
        self.vision_modality_dropout_rgb_prob = vision_modality_dropout_rgb_prob
        self.vision_modality_dropout_geometry_prob = vision_modality_dropout_geometry_prob
        self.vision_modality_dropout_source_keys = list(vision_modality_dropout_source_keys or [])
        probabilities = (vision_modality_dropout_rgb_prob, vision_modality_dropout_geometry_prob)
        if (
            any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
            or sum(probabilities) > 1
        ):
            raise ValueError(
                "Vision modality dropout probabilities must be finite, nonnegative and sum to <= 1"
            )
        if any(probabilities) and (
            len(self.vision_modality_dropout_source_keys) != 2
            or self.vision_modality_dropout_source_keys[0] != "ego_view"
            or self.vision_modality_dropout_source_keys[1]
            not in ("depth_gray_view", "surface_normals_view")
        ):
            raise ValueError(
                "Vision modality dropout requires ordered [ego_view, geometry] source keys"
            )
        self.trainable_params_fp32 = trainable_params_fp32
        if vision_fusion_mode != "none":
            incompatible_trainability = {
                "tune_llm": tune_llm,
                "tune_visual": tune_visual,
                "tune_vision_patch_embed": tune_vision_patch_embed,
                "tune_top_llm_layers": tune_top_llm_layers,
                "tune_vision_fusion_adapters": tune_vision_fusion_adapters,
            }
            if tune_llm or tune_visual or tune_vision_patch_embed or tune_top_llm_layers != 0:
                raise ValueError(
                    "Late fusion requires the shared Qwen patch stem, ViT, pretrained "
                    "vision-language projection, and LLM to remain frozen; got "
                    f"{incompatible_trainability!r}"
                )
            if not tune_vision_fusion_adapters:
                raise ValueError("Late-fusion training requires tune_vision_fusion_adapters=True")

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                **extra_kwargs,
                **transformers_loading_kwargs,
            ).eval()
        except Exception as exc:
            if _is_gated_repo_error(exc):
                raise RuntimeError(_GATED_BACKBONE_HINT.format(model_name=model_name)) from exc
            raise

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.expand_vision_input_channels(vision_input_channels, vision_patch_embed_init)  # fmt: skip  # earlyfusion
        self.vision_fusion_mode = "none"
        self.vision_fusion_version = 1
        self.vision_fusion_stage = _PRE_VISION_LANGUAGE_ADAPTER
        self.vision_fusion_adapter_type = "concat_linear"
        self.vision_fusion_adapter_bias = True
        self.vision_fusion_adapter_bias_init = "zeros"
        self.vision_fusion_adapter_count = 4
        self.vision_fusion_deepstack_layers = [5, 11, 17]
        self.vision_fusion_adapter_input_dim = 2048
        self.vision_fusion_adapter_output_dim = 1024
        self.vision_fusion_source_keys: list[str] = []
        self.vision_fusion_adapter_init = "rgb50_geo50"
        self.vision_fusion_rgb_init_scale = 0.5
        self.vision_fusion_geometry_init_scale = 0.5
        self.vision_fusion_adapters: torch.nn.ModuleDict | None = None
        if vision_fusion_mode != "none":
            self.enable_post_vision_late_fusion(
                mode=vision_fusion_mode,
                version=vision_fusion_version,
                stage=vision_fusion_stage,
                adapter_type=vision_fusion_adapter_type,
                adapter_bias=vision_fusion_adapter_bias,
                adapter_bias_init=vision_fusion_adapter_bias_init,
                adapter_count=vision_fusion_adapter_count,
                deepstack_layers=vision_fusion_deepstack_layers or [],
                adapter_input_dim=vision_fusion_adapter_input_dim,
                adapter_output_dim=vision_fusion_adapter_output_dim,
                source_keys=vision_fusion_source_keys or [],
                adapter_init=vision_fusion_adapter_init,
                rgb_init_scale=vision_fusion_rgb_init_scale,
                geometry_init_scale=vision_fusion_geometry_init_scale,
                tune_adapters=tune_vision_fusion_adapters,
            )
        self.select_layer = select_layer
        self.set_trainable_parameters(
            tune_llm,
            tune_visual,
            tune_top_llm_layers,
            tune_vision_patch_embed,
            tune_vision_fusion_adapters,
        )
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

        # Repair Qwen3-VL's non-persistent RoPE buffers once, after weights are
        # loaded and trainable dtypes are finalized. See _reset_rotary_inv_freq.
        self._reset_rotary_inv_freq()

        # Restore the fast vision patch-embed kernel on torch>=2.9. See method docstring.
        self._apply_vision_patch_embed_channels_last()

    @staticmethod
    def validate_post_vision_fusion_contract(
        *,
        mode: str,
        version: int,
        stage: str,
        adapter_type: str,
        adapter_bias: bool,
        adapter_bias_init: str,
        adapter_count: int,
        deepstack_layers: list[int],
        adapter_input_dim: int,
        adapter_output_dim: int,
        source_keys: list[str],
        adapter_init: str,
        rgb_init_scale: float,
        geometry_init_scale: float,
        vision_input_channels: int,
    ) -> None:
        """Fail closed on a late-fusion checkpoint/input contract mismatch."""
        if mode != "post_vision_late_fusion":
            raise ValueError(
                "vision_fusion_mode must be 'post_vision_late_fusion' when late fusion "
                f"is enabled; got {mode!r}"
            )
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError(f"Unsupported vision_fusion_version {version}; expected 1")
        if stage not in _VISION_FUSION_STAGES:
            raise ValueError(
                "vision_fusion_stage must be 'pre_vision_language_adapter' or "
                f"'post_vision_language_adapter'; got {stage!r}"
            )
        if adapter_type != "concat_linear":
            raise ValueError(
                f"vision_fusion_adapter_type must be 'concat_linear'; got {adapter_type!r}"
            )
        if adapter_bias is not True:
            raise ValueError(
                "vision_fusion_adapter_bias must be True for the version-1 contract; "
                f"got {adapter_bias!r}"
            )
        if adapter_bias_init != "zeros":
            raise ValueError(
                f"vision_fusion_adapter_bias_init must be 'zeros'; got {adapter_bias_init!r}"
            )
        if (
            isinstance(adapter_count, bool)
            or not isinstance(adapter_count, int)
            or adapter_count != 4
        ):
            raise ValueError(
                "Late fusion requires four independent adapters "
                f"(final + three DeepStack); got {adapter_count}"
            )
        if deepstack_layers != [5, 11, 17]:
            raise ValueError(
                "vision_fusion_deepstack_layers must be exactly [5, 11, 17]; "
                f"got {deepstack_layers!r}"
            )
        expected_dimensions = {
            _PRE_VISION_LANGUAGE_ADAPTER: (2048, 1024),
            _POST_VISION_LANGUAGE_ADAPTER: (4096, 2048),
        }[stage]
        dimensions = (adapter_input_dim, adapter_output_dim)
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions)
            or dimensions != expected_dimensions
        ):
            raise ValueError(
                f"vision fusion stage {stage!r} requires serialized adapter dimensions "
                f"Linear{expected_dimensions}; got Linear{dimensions}"
            )
        if (
            not isinstance(source_keys, list)
            or len(source_keys) != 2
            or not all(isinstance(key, str) and key for key in source_keys)
        ):
            raise ValueError(
                "vision_fusion_source_keys must contain exactly two non-empty ordered keys "
                "(primary RGB, auxiliary geometry)"
            )
        if source_keys[0] == source_keys[1]:
            raise ValueError("vision_fusion_source_keys must identify two distinct sources")
        if adapter_init != "rgb50_geo50":
            raise ValueError(
                f"vision_fusion_adapter_init must be 'rgb50_geo50'; got {adapter_init!r}"
            )
        scales = (rgb_init_scale, geometry_init_scale)
        if any(
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            for scale in scales
        ) or (float(rgb_init_scale), float(geometry_init_scale)) != (0.5, 0.5):
            raise ValueError(
                "The rgb50_geo50 adapter initialization requires exact serialized scales "
                "vision_fusion_rgb_init_scale=0.5 and "
                "vision_fusion_geometry_init_scale=0.5; "
                f"got {scales!r}"
            )
        if vision_input_channels != 3:
            raise ValueError(
                "Late fusion requires the unchanged three-channel Qwen vision "
                f"stem; got vision_input_channels={vision_input_channels}"
            )

    def _validate_qwen_deepstack_contract(
        self,
        *,
        stage: str,
        deepstack_layers: list[int],
        adapter_input_dim: int,
        adapter_output_dim: int,
    ) -> None:
        visual = self.model.visual
        indexes = list(getattr(visual, "deepstack_visual_indexes", []))
        mergers = getattr(visual, "deepstack_merger_list", None)
        merger_count = len(mergers) if mergers is not None else 0
        if indexes != deepstack_layers or merger_count != 3:
            raise RuntimeError(
                "Version-1 late fusion requires Qwen3-VL DeepStack layers "
                f"{deepstack_layers!r} and three outputs; found indexes={indexes!r}, "
                f"mergers={merger_count}"
            )
        vision_config = getattr(self.model.config, "vision_config", None)
        text_config = getattr(self.model.config, "text_config", None)
        vision_hidden = getattr(vision_config, "hidden_size", None)
        vision_output = getattr(vision_config, "out_hidden_size", None)
        text_hidden = getattr(text_config, "hidden_size", None)
        if (vision_hidden, vision_output, text_hidden) != (1024, 2048, 2048):
            raise RuntimeError(
                "Version-1 late fusion requires Qwen dimensions "
                "vision_hidden=1024, vision_output=2048, text_hidden=2048; got "
                f"{(vision_hidden, vision_output, text_hidden)!r}"
            )
        expected_dimensions = (
            (2 * vision_hidden, vision_hidden)
            if stage == _PRE_VISION_LANGUAGE_ADAPTER
            else (2 * vision_output, vision_output)
        )
        if (adapter_input_dim, adapter_output_dim) != expected_dimensions:
            raise RuntimeError(
                "Serialized late-fusion dimensions do not match the loaded Qwen layout: "
                f"serialized=Linear({adapter_input_dim}, {adapter_output_dim}), "
                f"loaded=Linear{expected_dimensions}"
            )

    def enable_post_vision_late_fusion(
        self,
        *,
        mode: str,
        version: int,
        stage: str,
        adapter_type: str,
        adapter_bias: bool,
        adapter_bias_init: str,
        adapter_count: int,
        deepstack_layers: list[int],
        adapter_input_dim: int,
        adapter_output_dim: int,
        source_keys: list[str],
        adapter_init: str,
        rgb_init_scale: float,
        geometry_init_scale: float,
        tune_adapters: bool,
    ) -> None:
        """Add deterministic late-fusion adapters to a loaded RGB base backbone.

        This method is also the explicit base-checkpoint migration path. Saved
        late-fusion checkpoints instantiate the same modules in ``__init__`` and
        then restore their learned weights normally.
        """
        vision_input_channels = self.model.visual.patch_embed.proj.in_channels
        self.validate_post_vision_fusion_contract(
            mode=mode,
            version=version,
            stage=stage,
            adapter_type=adapter_type,
            adapter_bias=adapter_bias,
            adapter_bias_init=adapter_bias_init,
            adapter_count=adapter_count,
            deepstack_layers=deepstack_layers,
            adapter_input_dim=adapter_input_dim,
            adapter_output_dim=adapter_output_dim,
            source_keys=source_keys,
            adapter_init=adapter_init,
            rgb_init_scale=rgb_init_scale,
            geometry_init_scale=geometry_init_scale,
            vision_input_channels=vision_input_channels,
        )
        self._validate_qwen_deepstack_contract(
            stage=stage,
            deepstack_layers=deepstack_layers,
            adapter_input_dim=adapter_input_dim,
            adapter_output_dim=adapter_output_dim,
        )

        if self.vision_fusion_adapters is not None:
            existing = (
                self.vision_fusion_mode,
                self.vision_fusion_version,
                self.vision_fusion_stage,
                self.vision_fusion_adapter_type,
                self.vision_fusion_adapter_bias,
                self.vision_fusion_adapter_bias_init,
                self.vision_fusion_adapter_count,
                self.vision_fusion_deepstack_layers,
                self.vision_fusion_adapter_input_dim,
                self.vision_fusion_adapter_output_dim,
                self.vision_fusion_source_keys,
                self.vision_fusion_adapter_init,
                self.vision_fusion_rgb_init_scale,
                self.vision_fusion_geometry_init_scale,
            )
            requested = (
                mode,
                version,
                stage,
                adapter_type,
                adapter_bias,
                adapter_bias_init,
                adapter_count,
                deepstack_layers,
                adapter_input_dim,
                adapter_output_dim,
                source_keys,
                adapter_init,
                rgb_init_scale,
                geometry_init_scale,
            )
            if existing != requested:
                raise RuntimeError(
                    f"Late-fusion backbone is already configured as {existing!r}, not {requested!r}"
                )
            if hasattr(self, "tune_llm"):
                self.set_trainable_parameters(
                    self.tune_llm,
                    self.tune_visual,
                    self.tune_top_llm_layers,
                    self.tune_vision_patch_embed,
                    tune_adapters,
                )
            else:
                self.vision_fusion_adapters.requires_grad_(tune_adapters)
            return

        vision_config = self.model.config.vision_config
        if stage == _PRE_VISION_LANGUAGE_ADAPTER:
            adapter_hidden_size = int(vision_config.hidden_size)
        else:
            adapter_hidden_size = int(vision_config.out_hidden_size)
        if (2 * adapter_hidden_size, adapter_hidden_size) != (
            adapter_input_dim,
            adapter_output_dim,
        ):
            raise RuntimeError(
                "Serialized late-fusion adapter dimensions do not match the loaded Qwen "
                f"stage: serialized=Linear({adapter_input_dim}, {adapter_output_dim}), "
                f"loaded=Linear({2 * adapter_hidden_size}, {adapter_hidden_size})"
            )
        device = _real_inference_device(self.model)
        model_dtype = next(
            (
                parameter.dtype
                for parameter in self.model.parameters()
                if parameter.device.type != "meta"
            ),
            torch.float32,
        )
        adapter_dtype = torch.float32 if self.trainable_params_fp32 else model_dtype
        names = ("final", "deepstack_0", "deepstack_1", "deepstack_2")
        adapters = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(
                    2 * adapter_hidden_size,
                    adapter_hidden_size,
                    bias=adapter_bias,
                    device=device,
                    dtype=adapter_dtype,
                )
                for name in names
            }
        )
        with torch.no_grad():
            for adapter in adapters.values():
                adapter.weight.zero_()
                adapter.weight[:, :adapter_hidden_size].diagonal().fill_(float(rgb_init_scale))
                adapter.weight[:, adapter_hidden_size:].diagonal().fill_(float(geometry_init_scale))
                if adapter.bias is not None:
                    adapter.bias.zero_()
        adapters.requires_grad_(tune_adapters)

        self.vision_fusion_adapters = adapters
        self.vision_fusion_mode = mode
        self.vision_fusion_version = version
        self.vision_fusion_stage = stage
        self.vision_fusion_adapter_type = adapter_type
        self.vision_fusion_adapter_bias = adapter_bias
        self.vision_fusion_adapter_bias_init = adapter_bias_init
        self.vision_fusion_adapter_count = adapter_count
        self.vision_fusion_deepstack_layers = list(deepstack_layers)
        self.vision_fusion_adapter_input_dim = adapter_input_dim
        self.vision_fusion_adapter_output_dim = adapter_output_dim
        self.vision_fusion_source_keys = list(source_keys)
        self.vision_fusion_adapter_init = adapter_init
        self.vision_fusion_rgb_init_scale = float(rgb_init_scale)
        self.vision_fusion_geometry_init_scale = float(geometry_init_scale)
        if hasattr(self, "tune_llm"):
            self.set_trainable_parameters(
                self.tune_llm,
                self.tune_visual,
                self.tune_top_llm_layers,
                self.tune_vision_patch_embed,
                tune_adapters,
            )

    def expand_vision_input_channels(self, input_channels: int, initialization: str) -> None:  # fmt: skip  # earlyfusion
        if input_channels not in (3, 4, 6):  # earlyfusion
            raise ValueError(f"vision_input_channels must be 3, 4, or 6; got {input_channels}")  # fmt: skip  # earlyfusion
        patch_embed = self.model.visual.patch_embed  # earlyfusion
        old = patch_embed.proj  # earlyfusion
        if old.in_channels == input_channels:  # earlyfusion
            return  # earlyfusion
        if old.in_channels != 3:  # earlyfusion
            raise ValueError(f"Cannot expand vision patch embedding from {old.in_channels} channels")  # fmt: skip  # earlyfusion
        allowed = {4: ("rgb_mean",), 6: ("zeros", "rgb_mean")}[input_channels]  # earlyfusion
        if initialization not in allowed:  # earlyfusion
            raise ValueError(f"vision_patch_embed_init must be one of {allowed!r} for {input_channels} channels")  # fmt: skip  # earlyfusion
        new = torch.nn.Conv3d(input_channels, old.out_channels, old.kernel_size, old.stride, old.padding, old.dilation, old.groups, old.bias is not None, old.padding_mode, device=old.weight.device, dtype=old.weight.dtype)  # fmt: skip  # earlyfusion
        with torch.no_grad():  # earlyfusion
            new.weight.zero_()  # earlyfusion
            new.weight[:, :3].copy_(old.weight)  # earlyfusion
            if initialization == "rgb_mean":  # earlyfusion
                rgb_mean = old.weight.mean(dim=1, keepdim=True)  # earlyfusion
                new.weight[:, 3:].copy_(rgb_mean.expand(-1, input_channels - 3, -1, -1, -1))  # fmt: skip  # earlyfusion
            if old.bias is not None:  # earlyfusion
                new.bias.copy_(old.bias)  # earlyfusion
        new.weight.requires_grad_(old.weight.requires_grad)  # earlyfusion
        if old.bias is not None:  # earlyfusion
            new.bias.requires_grad_(old.bias.requires_grad)  # earlyfusion
        patch_embed.proj = new  # earlyfusion
        patch_embed.in_channels = input_channels  # earlyfusion
        self.model.config.vision_config.in_channels = input_channels  # earlyfusion
        self.model.visual.config.in_channels = input_channels  # earlyfusion

    def _apply_vision_patch_embed_channels_last(self) -> None:
        """Force the Qwen3-VL vision patch-embed ``Conv3d`` to ``channels_last_3d``.

        On torch>=2.9 (cuDNN 9.10.x) the contiguous bf16 patch-embed ``Conv3d``
        dispatches to a pathologically slow kernel on sm_80; ``channels_last_3d``
        selects the fast one and is numerically equivalent (harmless elsewhere).
        """
        visual = getattr(self.model, "visual", None)
        if visual is None:
            return

        def _to_channels_last_3d(_module, inputs):
            # Skip while tracing: the legacy TorchScript ONNX exporter cannot lower a
            # memory_format on .contiguous() ("onnx memory_format support is not
            # implemented"). This is a runtime-only perf hint, so the plain contiguous
            # graph is numerically equivalent for export.
            if not inputs or torch.jit.is_tracing():
                return inputs
            x = inputs[0]
            if isinstance(x, torch.Tensor) and x.dim() == 5:
                return (x.contiguous(memory_format=torch.channels_last_3d), *inputs[1:])
            return inputs

        patched = 0
        for module in visual.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.to(memory_format=torch.channels_last_3d)
                module.register_forward_pre_hook(_to_channels_last_3d)
                patched += 1
        if patched:
            logger.debug(
                "Applied channels_last_3d to %d vision patch-embed Conv3d module(s) "
                "(torch>=2.9 cuDNN Conv3d perf workaround).",
                patched,
            )

    def set_trainable_parameters(
        self,
        tune_llm: bool,
        tune_visual: bool,
        tune_top_llm_layers: int,
        tune_vision_patch_embed: bool = False,
        tune_vision_fusion_adapters: bool = True,
    ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.tune_vision_patch_embed = tune_vision_patch_embed  # earlyfusion
        self.tune_top_llm_layers = tune_top_llm_layers
        if getattr(self, "vision_fusion_mode", "none") == "post_vision_late_fusion":
            if tune_llm or tune_visual or tune_vision_patch_embed or tune_top_llm_layers != 0:
                raise ValueError(
                    "Late fusion requires the shared Qwen patch stem, ViT, pretrained "
                    "vision-language projection, and LLM to remain frozen"
                )
            if not tune_vision_fusion_adapters:
                raise ValueError("Late-fusion training requires tune_vision_fusion_adapters=True")
            self.model.requires_grad_(False)
            if self.vision_fusion_adapters is None:
                raise RuntimeError("Late-fusion adapters are not initialized")
            self.vision_fusion_adapters.requires_grad_(True)
            if any(parameter.requires_grad for parameter in self.model.parameters()):
                raise RuntimeError("Late-fusion Qwen freeze contract was not applied")
            if not all(
                parameter.requires_grad for parameter in self.vision_fusion_adapters.parameters()
            ):
                raise RuntimeError("Late-fusion adapters must remain trainable")
        else:
            self._set_legacy_trainable_parameters(
                tune_llm=tune_llm,
                tune_visual=tune_visual,
                tune_top_llm_layers=tune_top_llm_layers,
                tune_vision_patch_embed=tune_vision_patch_embed,
            )

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def _set_legacy_trainable_parameters(
        self,
        *,
        tune_llm: bool,
        tune_visual: bool,
        tune_top_llm_layers: int,
        tune_vision_patch_embed: bool,
    ) -> None:
        """Preserve the original trainability behavior for every non-late mode."""
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)
        if tune_vision_patch_embed:  # earlyfusion
            self.model.visual.patch_embed.proj.requires_grad_(True)  # earlyfusion
        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()

    def _reset_rotary_inv_freq(self) -> None:
        """Re-derive Qwen3-VL's non-persistent RoPE ``inv_freq`` buffers once at load.

        These ``persistent=False`` buffers are not restored from a checkpoint and
        can be left uninitialized under the nested ``no_init_weights`` load, so we
        rebuild the analytic FP32 frequencies here (load time only, no per-forward
        cost). Fails closed (raises) when the layout/config needed to rebuild is
        missing, so a drifted transformers version cannot silently serve
        uninitialized buffers.
        """
        config = getattr(self.model, "config", None)
        vision_changed = self._reset_vision_rotary_inv_freq(config)
        text_changed = self._reset_language_rotary_inv_freq()
        logger.debug(
            "Qwen3-VL RoPE inv_freq reset (vision_rewritten=%s, text_rewritten=%s).",
            vision_changed,
            text_changed,
        )

    def _reset_vision_rotary_inv_freq(self, config) -> bool:
        visual = getattr(self.model, "visual", None)
        rotary = getattr(visual, "rotary_pos_emb", None)
        if rotary is None or not hasattr(rotary, "inv_freq"):
            raise RuntimeError(
                "Qwen3-VL visual rotary_pos_emb/inv_freq not found; cannot rebuild the "
                "non-persistent vision RoPE inv_freq. Continuing would leave it "
                "uninitialized and silently corrupt inference, so we fail the load. The "
                "transformers Qwen3-VL layout may have changed; pin a compatible version."
            )

        vision_config = getattr(config, "vision_config", None)
        if vision_config is None or not all(
            hasattr(vision_config, attr) for attr in ("hidden_size", "num_heads")
        ):
            raise RuntimeError(
                "Qwen3-VL vision_config missing hidden_size/num_heads; cannot rebuild the "
                "non-persistent vision RoPE inv_freq (continuing would leave it "
                "uninitialized and silently corrupt inference)."
            )

        head_dim = vision_config.hidden_size // vision_config.num_heads
        device = _real_inference_device(visual)
        inv_freq = recompute_vision_rotary_inv_freq(rotary, head_dim // 2, device)
        return _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)

    def _reset_language_rotary_inv_freq(self) -> bool:
        language_model = getattr(self.model, "language_model", None)
        rotary = getattr(language_model, "rotary_emb", None)
        text_config = getattr(rotary, "config", None) or getattr(language_model, "config", None)
        if rotary is None or not hasattr(rotary, "inv_freq") or text_config is None:
            raise RuntimeError(
                "Qwen3-VL language rotary_emb/inv_freq/config not found; cannot rebuild the "
                "non-persistent text RoPE inv_freq. Continuing would leave it uninitialized "
                "and silently corrupt inference, so we fail the load. The transformers "
                "Qwen3-VL layout may have changed; pin a compatible version."
            )

        device = _real_inference_device(language_model)
        inv_freq, _attention_scaling = recompute_text_rotary_inv_freq(rotary, text_config, device)
        changed = _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)
        # ``original_inv_freq`` is a plain attribute (not a buffer) that dynamic
        # RoPE updates restore from; keep it consistent with inv_freq.
        if hasattr(rotary, "original_inv_freq"):
            changed = (
                _assign_inv_freq(rotary, "original_inv_freq", inv_freq.clone(), persistent=False)
                or changed
            )
        return changed

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _vision_modality_dropout_active(self) -> bool:
        return self.training and (
            getattr(self, "vision_modality_dropout_rgb_prob", 0.0) > 0
            or getattr(self, "vision_modality_dropout_geometry_prob", 0.0) > 0
        )

    def _modality_token_keep(
        self, vl_input: dict[str, torch.Tensor], *, separate_views: bool, merged: bool
    ) -> torch.Tensor:
        """Expand processor-owned example masks in packed image/token order, without RNG."""
        if "vision_modality_keep" not in vl_input or "vision_modality_num_frames" not in vl_input:
            raise ValueError(
                "Enabled training modality dropout requires processor keep/frame metadata"
            )
        keep = vl_input["vision_modality_keep"]
        frames = vl_input["vision_modality_num_frames"]
        batch_size = vl_input["input_ids"].shape[0]
        if keep.shape != (batch_size, 2) or not bool(((keep == 0) | (keep == 1)).all()):
            raise ValueError("vision_modality_keep must be binary [batch, 2]")
        if not bool((keep.sum(dim=-1) >= 1).all()):
            raise ValueError("Vision modality dropout cannot remove RGB and geometry together")
        if frames.shape != (batch_size,) or frames.dtype not in (torch.int32, torch.int64):
            raise ValueError("vision_modality_num_frames must be integer [batch]")
        if not bool((frames > 0).all()):
            raise ValueError("vision_modality_num_frames must be positive")
        grid = vl_input["image_grid_thw"]
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.dtype not in (torch.int32, torch.int64):
            raise ValueError("Modality dropout requires integer image_grid_thw [images, 3]")
        if not bool((grid > 0).all()):
            raise ValueError("Modality dropout image grids must be positive")
        image_keep = torch.repeat_interleave(keep, frames.to(keep.device), dim=0)
        if separate_views:
            # Processor order is sample -> time -> [RGB, geometry].
            image_keep = image_keep.reshape(-1, 1)
        if image_keep.shape[0] != grid.shape[0]:
            raise ValueError("Modality dropout frame counts do not match packed image grids")
        token_counts = grid.prod(dim=-1)
        if merged:
            merge_area = self.model.model.visual.spatial_merge_size**2
            if bool((token_counts.remainder(merge_area) != 0).any()):
                raise ValueError("Image grids must contain complete spatial-merge groups")
            token_counts = token_counts // merge_area
        return torch.repeat_interleave(image_keep, token_counts.to(keep.device), dim=0)

    def _mask_early_fusion_pixels(self, vl_input: dict[str, torch.Tensor]) -> torch.Tensor:
        """Zero normalized channel groups before the shared patch stem (not black RGB)."""
        pixels = vl_input["pixel_values"]
        channels = self.model.visual.patch_embed.proj.in_channels
        if channels not in (4, 6) or pixels.ndim != 2 or pixels.shape[1] % channels:
            raise ValueError("Early-fusion modality dropout requires packed 4/6-channel patches")
        keep = self._modality_token_keep(vl_input, separate_views=False, merged=False)
        if keep.shape[0] != pixels.shape[0]:
            raise ValueError("Early-fusion modality mask does not match raw patch rows")
        channel_keep = torch.cat(
            (keep[:, :1].expand(-1, 3), keep[:, 1:].expand(-1, channels - 3)), dim=-1
        ).to(device=pixels.device, dtype=pixels.dtype)
        return (pixels.reshape(pixels.shape[0], channels, -1) * channel_keep[..., None]).reshape_as(
            pixels
        )

    def _forward_separate_view_modality_dropout(
        self, vl_input: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Remove a view's content at every Qwen vision-to-language entry, retaining slots."""
        qwen = self.model.model
        keep = self._modality_token_keep(vl_input, separate_views=True, merged=True)
        chunks, deepstack = qwen.get_image_features(
            vl_input["pixel_values"], vl_input["image_grid_thw"]
        )
        if (
            not isinstance(chunks, (tuple, list))
            or len(chunks) != vl_input["image_grid_thw"].shape[0]
        ):
            raise RuntimeError("Separate-view Qwen features did not preserve image order")
        final = torch.cat(tuple(chunks), dim=0)
        if final.ndim != 2 or keep.shape[0] != final.shape[0]:
            raise RuntimeError("Separate-view modality mask does not match Qwen token rows")
        if not isinstance(deepstack, (tuple, list)) or len(deepstack) != 3:
            raise RuntimeError("Separate-view dropout requires all three Qwen DeepStack streams")
        if any(stream.shape != final.shape for stream in deepstack):
            raise RuntimeError("Separate-view final/DeepStack token alignment mismatch")
        return self._forward_fused_vision_language(
            input_ids=vl_input["input_ids"],
            attention_mask=vl_input["attention_mask"],
            primary_grid=vl_input["image_grid_thw"],
            fused_final=final * keep.to(device=final.device, dtype=final.dtype),
            fused_deepstack=[
                stream * keep.to(device=stream.device, dtype=stream.dtype) for stream in deepstack
            ],
            preserve_standard_hidden_state=True,
        )

    def _fuse_vision_streams(
        self,
        adapter_name: str,
        primary: torch.Tensor,
        auxiliary: torch.Tensor,
        modality_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.vision_fusion_adapters is None:
            raise RuntimeError("Late-fusion adapters are not enabled")
        if primary.ndim != 2 or auxiliary.ndim != 2 or primary.shape != auxiliary.shape:
            raise ValueError(
                f"Late-fusion {adapter_name} streams must have identical rank-2 shapes; "
                f"got primary={tuple(primary.shape)}, auxiliary={tuple(auxiliary.shape)}"
            )
        adapter = self.vision_fusion_adapters[adapter_name]
        expected_width = adapter.out_features
        if primary.shape[-1] != expected_width:
            raise ValueError(
                f"Late-fusion {adapter_name} stream width must be {expected_width} for "
                f"stage {self.vision_fusion_stage!r}; "
                f"got {primary.shape[-1]}"
            )
        if (
            adapter.in_features != self.vision_fusion_adapter_input_dim
            or adapter.out_features != self.vision_fusion_adapter_output_dim
            or (adapter.bias is not None) != self.vision_fusion_adapter_bias
        ):
            raise RuntimeError(
                f"Late-fusion {adapter_name} does not match its serialized adapter contract: "
                f"Linear({adapter.in_features}, {adapter.out_features}, "
                f"bias={adapter.bias is not None})"
            )
        if modality_keep is not None:
            if modality_keep.shape != (primary.shape[0], 2):
                raise ValueError("Late-fusion modality mask must match packed token rows")
            primary = primary * modality_keep[:, :1].to(device=primary.device, dtype=primary.dtype)
            auxiliary = auxiliary * modality_keep[:, 1:].to(
                device=auxiliary.device, dtype=auxiliary.dtype
            )
        adapter_input = torch.cat((primary, auxiliary), dim=-1).to(
            device=adapter.weight.device, dtype=adapter.weight.dtype
        )
        fused = adapter(adapter_input)
        return fused.to(device=primary.device, dtype=primary.dtype)

    @staticmethod
    def _attention_mask_for_rope(
        attention_mask: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask["full_attention"]
        if attention_mask.ndim != 4:
            return attention_mask
        attention_mask = torch.diagonal(attention_mask[:, 0], dim1=1, dim2=2)
        if attention_mask.dtype.is_floating_point:
            attention_mask = attention_mask / torch.finfo(attention_mask.dtype).min
            attention_mask = (1.0 - attention_mask).int()
        return attention_mask

    def _encode_pre_merger_late_fusion(
        self,
        combined_pixels: torch.Tensor,
        combined_grid: torch.Tensor,
        primary_raw_token_count: int,
        modality_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the pinned Qwen vision path and fuse raw ViT features before mergers.

        RGB images occupy the first half of ``combined_grid`` and auxiliary
        images the second. Qwen's ``cu_seqlens`` keeps every image independent
        through the shared ViT. At layers 5/11/17 and the final output, aligned
        raw 1024-wide source tokens are fused before invoking Qwen's existing
        pretrained DeepStack/final mergers. The returned tensors therefore keep
        the normal 2048-wide language interface and one destination image span.
        """
        visual = self.model.model.visual
        hidden_states = visual.patch_embed(combined_pixels.type(visual.dtype))
        expected_raw_tokens = 2 * primary_raw_token_count
        if hidden_states.ndim != 2 or hidden_states.shape[0] != expected_raw_tokens:
            raise RuntimeError(
                "Qwen patch embedding did not preserve the paired raw-token layout: "
                f"expected ({expected_raw_tokens}, 1024), got {tuple(hidden_states.shape)}"
            )

        pos_embeds = visual.fast_pos_embed_interpolate(combined_grid)
        if pos_embeds.shape != hidden_states.shape:
            raise RuntimeError(
                "Qwen position embeddings do not match paired patch embeddings: "
                f"patches={tuple(hidden_states.shape)}, positions={tuple(pos_embeds.shape)}"
            )
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = visual.rot_pos_emb(combined_grid)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        rotary_pair = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (rotary_pair.cos(), rotary_pair.sin())

        cu_seqlens = torch.repeat_interleave(
            combined_grid[:, 1] * combined_grid[:, 2], combined_grid[:, 0]
        ).cumsum(
            dim=0,
            dtype=combined_grid.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_indexes = list(visual.deepstack_visual_indexes)
        if deepstack_indexes != self.vision_fusion_deepstack_layers:
            raise RuntimeError(
                "Loaded Qwen DeepStack layers no longer match the serialized late-fusion "
                f"contract: loaded={deepstack_indexes!r}, "
                f"serialized={self.vision_fusion_deepstack_layers!r}"
            )
        deepstack_lookup = {layer: index for index, layer in enumerate(deepstack_indexes)}
        fused_deepstack = []
        for layer_num, block in enumerate(visual.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in deepstack_lookup:
                stream_index = deepstack_lookup[layer_num]
                fused_raw = self._fuse_vision_streams(
                    f"deepstack_{stream_index}",
                    hidden_states[:primary_raw_token_count],
                    hidden_states[primary_raw_token_count:],
                    modality_keep=modality_keep,
                )
                fused_deepstack.append(visual.deepstack_merger_list[stream_index](fused_raw))

        fused_final_raw = self._fuse_vision_streams(
            "final",
            hidden_states[:primary_raw_token_count],
            hidden_states[primary_raw_token_count:],
            modality_keep=modality_keep,
        )
        fused_final = visual.merger(fused_final_raw)
        if len(fused_deepstack) != 3:
            raise RuntimeError(
                "Pre-merger late fusion requires outputs from exactly three DeepStack sites; "
                f"got {len(fused_deepstack)}"
            )
        expected_shape = fused_final.shape
        if fused_final.ndim != 2 or any(
            stream.shape != expected_shape for stream in fused_deepstack
        ):
            raise RuntimeError(
                "Qwen final and DeepStack mergers must return matching rank-2 token streams; "
                f"final={tuple(expected_shape)}, "
                f"deepstack={[tuple(stream.shape) for stream in fused_deepstack]}"
            )
        expected_width = int(self.model.config.text_config.hidden_size)
        if fused_final.shape[-1] != expected_width:
            raise RuntimeError(
                "Qwen vision-to-language projection returned an unexpected width: "
                f"expected {expected_width}, got {fused_final.shape[-1]}"
            )
        return fused_final, fused_deepstack

    @staticmethod
    def _validate_late_fusion_inputs(
        vl_input: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        required = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "late_fusion_pixel_values",
            "late_fusion_image_grid_thw",
        }
        missing = sorted(required - vl_input.keys())
        if missing:
            raise KeyError(f"Late-fusion input is missing required keys: {missing}")

        input_ids = vl_input["input_ids"]
        attention_mask = vl_input["attention_mask"]
        primary_pixels = vl_input["pixel_values"]
        auxiliary_pixels = vl_input["late_fusion_pixel_values"]
        primary_grid = vl_input["image_grid_thw"]
        auxiliary_grid = vl_input["late_fusion_image_grid_thw"]

        if primary_grid.ndim != 2 or primary_grid.shape[-1] != 3 or primary_grid.numel() == 0:
            raise ValueError(
                "image_grid_thw must be a non-empty [num_images, 3] tensor for late fusion; "
                f"got {tuple(primary_grid.shape)}"
            )
        if primary_grid.shape != auxiliary_grid.shape or not torch.equal(
            primary_grid, auxiliary_grid
        ):
            raise ValueError(
                "Late-fusion primary and auxiliary image_grid_thw tensors must be exactly "
                "equal so tokens can be paired spatially"
            )
        if primary_pixels.shape != auxiliary_pixels.shape:
            raise ValueError(
                "Late-fusion primary and auxiliary pixel tensors must have identical shapes; "
                f"got primary={tuple(primary_pixels.shape)}, "
                f"auxiliary={tuple(auxiliary_pixels.shape)}"
            )
        return (
            input_ids,
            attention_mask,
            primary_pixels,
            auxiliary_pixels,
            primary_grid,
            auxiliary_grid,
        )

    def _forward_fused_vision_language(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        primary_grid: torch.Tensor,
        fused_final: torch.Tensor,
        fused_deepstack: list[torch.Tensor],
        preserve_standard_hidden_state: bool = False,
    ) -> torch.Tensor:
        if len(fused_deepstack) != 3:
            raise RuntimeError(
                "Late fusion requires exactly three DeepStack visual streams; "
                f"got {len(fused_deepstack)}"
            )
        qwen = self.model.model
        inputs_embeds = qwen.get_input_embeddings()(input_ids)
        fused_final = fused_final.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask, _ = qwen.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=fused_final,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, fused_final)
        visual_pos_masks = image_mask[..., 0]
        fused_deepstack = [
            stream.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            for stream in fused_deepstack
        ]

        position_ids, rope_deltas = qwen.get_rope_index(
            input_ids,
            primary_grid,
            None,
            attention_mask=self._attention_mask_for_rope(attention_mask),
        )
        qwen.rope_deltas = rope_deltas
        if preserve_standard_hidden_state or (
            getattr(self, "vision_fusion_mode", "none") == "post_vision_late_fusion"
            and getattr(self, "late_fusion_language_readout", "post_final_rmsnorm")
            == "pre_final_rmsnorm"
        ):
            # The ordinary conditional-model output recorder returns the last
            # decoder tensor BEFORE final RMSNorm. Corrected late fusion and
            # separate-view dropout preserve that exact tensor and its gradients.
            # Historical late-fusion checkpoints still return last_hidden_state.
            layers = getattr(qwen.language_model, "layers", None)
            if layers is None or len(layers) == 0:
                raise RuntimeError("Pre-final-RMSNorm readout requires Qwen decoder output capture")
            captured = []

            def capture_last_decoder(_module, _inputs, output):
                captured.append(output[0] if isinstance(output, tuple) else output)

            handle = layers[-1].register_forward_hook(capture_last_decoder)
            try:
                qwen.language_model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=fused_deepstack,
                )
            finally:
                handle.remove()
            if len(captured) != 1 or not isinstance(captured[0], torch.Tensor):
                raise RuntimeError(
                    "Qwen last decoder did not produce exactly one hidden-state tensor"
                )
            return captured[0]
        outputs = qwen.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=fused_deepstack,
        )
        return outputs.last_hidden_state

    def _forward_pre_adapter_late_fusion(self, vl_input: dict[str, torch.Tensor]) -> torch.Tensor:
        (
            input_ids,
            attention_mask,
            primary_pixels,
            auxiliary_pixels,
            primary_grid,
            auxiliary_grid,
        ) = self._validate_late_fusion_inputs(vl_input)

        primary_raw_token_count = int(primary_grid.prod(dim=-1).sum().item())
        if primary_pixels.ndim != 2 or primary_pixels.shape[0] != primary_raw_token_count:
            raise ValueError(
                "Late-fusion primary pixel_values must contain exactly one flattened patch "
                f"per raw grid token; pixels={tuple(primary_pixels.shape)}, "
                f"grid_tokens={primary_raw_token_count}"
            )
        combined_pixels = torch.cat((primary_pixels, auxiliary_pixels), dim=0)
        combined_grid = torch.cat((primary_grid, auxiliary_grid), dim=0)
        fused_final, fused_deepstack = self._encode_pre_merger_late_fusion(
            combined_pixels,
            combined_grid,
            primary_raw_token_count,
            **(
                {
                    "modality_keep": self._modality_token_keep(
                        vl_input, separate_views=False, merged=False
                    )
                }
                if self._vision_modality_dropout_active()
                else {}
            ),
        )
        return self._forward_fused_vision_language(
            input_ids=input_ids,
            attention_mask=attention_mask,
            primary_grid=primary_grid,
            fused_final=fused_final,
            fused_deepstack=fused_deepstack,
        )

    def _forward_post_adapter_late_fusion(self, vl_input: dict[str, torch.Tensor]) -> torch.Tensor:
        (
            input_ids,
            attention_mask,
            primary_pixels,
            auxiliary_pixels,
            primary_grid,
            auxiliary_grid,
        ) = self._validate_late_fusion_inputs(vl_input)

        modality_keep = (
            self._modality_token_keep(vl_input, separate_views=False, merged=True)
            if self._vision_modality_dropout_active()
            else None
        )
        qwen = self.model.model
        combined_pixels = torch.cat((primary_pixels, auxiliary_pixels), dim=0)
        combined_grid = torch.cat((primary_grid, auxiliary_grid), dim=0)
        image_chunks, deepstack_streams = qwen.get_image_features(
            combined_pixels,
            combined_grid,
        )

        primary_image_count = primary_grid.shape[0]
        if not isinstance(image_chunks, (tuple, list)) or len(image_chunks) != (
            2 * primary_image_count
        ):
            raise RuntimeError(
                "Qwen vision output did not preserve the paired image layout: "
                f"expected {2 * primary_image_count} chunks, got "
                f"{len(image_chunks) if isinstance(image_chunks, (tuple, list)) else type(image_chunks)!r}"
            )
        primary_chunks = image_chunks[:primary_image_count]
        auxiliary_chunks = image_chunks[primary_image_count:]
        mismatched_chunks = [
            index
            for index, (primary, auxiliary) in enumerate(
                zip(primary_chunks, auxiliary_chunks, strict=True)
            )
            if primary.shape != auxiliary.shape
        ]
        if mismatched_chunks:
            raise RuntimeError(
                "Qwen produced different token shapes for aligned late-fusion image pairs "
                f"at indexes {mismatched_chunks}"
            )
        primary_final = torch.cat(tuple(primary_chunks), dim=0)
        auxiliary_final = torch.cat(tuple(auxiliary_chunks), dim=0)
        fused_final = self._fuse_vision_streams(
            "final", primary_final, auxiliary_final, modality_keep=modality_keep
        )

        if not isinstance(deepstack_streams, (tuple, list)) or len(deepstack_streams) != 3:
            raise RuntimeError(
                "Qwen vision output must contain exactly three DeepStack streams; got "
                f"{len(deepstack_streams) if isinstance(deepstack_streams, (tuple, list)) else type(deepstack_streams)!r}"
            )
        primary_token_count = primary_final.shape[0]
        fused_deepstack = []
        for index, stream in enumerate(deepstack_streams):
            if stream.ndim != 2 or stream.shape[0] != 2 * primary_token_count:
                raise RuntimeError(
                    f"Qwen DeepStack stream {index} did not preserve paired token order: "
                    f"expected {2 * primary_token_count} rows, got {tuple(stream.shape)}"
                )
            fused_deepstack.append(
                self._fuse_vision_streams(
                    f"deepstack_{index}",
                    stream[:primary_token_count],
                    stream[primary_token_count:],
                    modality_keep=modality_keep,
                )
            )

        return self._forward_fused_vision_language(
            input_ids=input_ids,
            attention_mask=attention_mask,
            primary_grid=primary_grid,
            fused_final=fused_final,
            fused_deepstack=fused_deepstack,
        )

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        modality_dropout = self._vision_modality_dropout_active()
        if modality_dropout:
            keys_to_use.extend(["vision_modality_keep", "vision_modality_num_frames"])
        if self.vision_fusion_mode == "post_vision_late_fusion":
            keys_to_use.extend(["late_fusion_pixel_values", "late_fusion_image_grid_thw"])
            selected_input = {k: vl_input[k] for k in keys_to_use if k in vl_input}
            if self.vision_fusion_stage == _PRE_VISION_LANGUAGE_ADAPTER:
                outputs = self._forward_pre_adapter_late_fusion(selected_input)
            elif self.vision_fusion_stage == _POST_VISION_LANGUAGE_ADAPTER:
                outputs = self._forward_post_adapter_late_fusion(selected_input)
            else:
                raise RuntimeError(
                    "Late-fusion backbone has unsupported serialized stage "
                    f"{self.vision_fusion_stage!r}"
                )
        else:
            late_keys = {"late_fusion_pixel_values", "late_fusion_image_grid_thw"}
            unexpected_late_keys = sorted(late_keys.intersection(vl_input.keys()))
            if unexpected_late_keys:
                raise ValueError(
                    "Received late-fusion inputs for a non-late-fusion checkpoint: "
                    f"{unexpected_late_keys}"
                )
            selected_input = {k: vl_input[k] for k in keys_to_use}
            if modality_dropout and self.model.visual.patch_embed.proj.in_channels == 3:
                outputs = self._forward_separate_view_modality_dropout(selected_input)
            else:
                model_input = selected_input
                if modality_dropout:
                    model_input = {
                        k: v
                        for k, v in selected_input.items()
                        if not k.startswith("vision_modality_")
                    }
                    model_input["pixel_values"] = self._mask_early_fusion_pixels(selected_input)
                model_outputs = self.model(**model_input, output_hidden_states=True)
                outputs = model_outputs.hidden_states[-1]
        image_mask = selected_input["input_ids"] == self.model.config.image_token_id
        attention_mask = selected_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
