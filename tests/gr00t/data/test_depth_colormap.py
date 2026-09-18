# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

from gr00t.data.depth_colormap import (
    FIXED_TURBO_DEPTH_COLORMAP,
    FIXED_TURBO_DEPTH_LUT,
    FIXED_TURBO_DEPTH_LUT_SHA256,
    apply_fixed_turbo_depth_colormap,
    apply_fixed_turbo_depth_colormap_array,
    fixed_turbo_depth_colormap_contract,
)
import numpy as np
from PIL import Image
import pytest


def test_fixed_turbo_depth_lut_is_pinned_and_reserves_invalid_black():
    digest = "sha256:" + hashlib.sha256(FIXED_TURBO_DEPTH_LUT.tobytes()).hexdigest()

    assert FIXED_TURBO_DEPTH_LUT.shape == (256, 3)
    assert FIXED_TURBO_DEPTH_LUT.dtype == np.uint8
    assert FIXED_TURBO_DEPTH_LUT.flags.writeable is False
    assert digest == FIXED_TURBO_DEPTH_LUT_SHA256
    np.testing.assert_array_equal(FIXED_TURBO_DEPTH_LUT[0], [0, 0, 0])
    np.testing.assert_array_equal(FIXED_TURBO_DEPTH_LUT[1], [48, 18, 59])
    np.testing.assert_array_equal(FIXED_TURBO_DEPTH_LUT[255], [122, 4, 2])

    assert fixed_turbo_depth_colormap_contract() == {
        "name": FIXED_TURBO_DEPTH_COLORMAP,
        "lut_sha256": FIXED_TURBO_DEPTH_LUT_SHA256,
        "input_view": "depth_gray_view",
        "input_encoding": "linear_grayscale_replicated_rgb",
        "invalid_input_value": 0,
        "invalid_output_rgb": [0, 0, 0],
        "valid_input_range": [1, 255],
    }


def test_fixed_turbo_depth_colormap_is_an_exact_gray_code_lookup():
    gray = np.array([[0, 1, 2], [127, 254, 255]], dtype=np.uint8)
    replicated = np.repeat(gray[..., None], 3, axis=-1)

    output = np.asarray(apply_fixed_turbo_depth_colormap(Image.fromarray(replicated)))

    np.testing.assert_array_equal(output, FIXED_TURBO_DEPTH_LUT[gray])
    np.testing.assert_array_equal(output[0, 0], [0, 0, 0])

    batch = np.repeat(replicated[None, None], 2, axis=0)
    batch_output = apply_fixed_turbo_depth_colormap_array(batch)
    np.testing.assert_array_equal(batch_output, FIXED_TURBO_DEPTH_LUT[batch[..., 0]])


def test_fixed_turbo_depth_colormap_rejects_non_replicated_rgb():
    invalid = np.zeros((2, 3, 3), dtype=np.uint8)
    invalid[..., 1] = 1

    with pytest.raises(ValueError, match="three identical channels"):
        apply_fixed_turbo_depth_colormap(invalid)
