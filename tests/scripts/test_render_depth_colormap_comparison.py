import numpy as np
from scripts.analysis_tools.render_depth_colormap_comparison import colourize_depth


def _render(depth, colour_map):
    return colourize_depth(
        np.asarray(depth, dtype=np.uint16),
        scale_m_per_unit=0.001,
        near_m=0.25,
        far_m=1.0,
        colour_map=colour_map,
    )


def test_current_gray_matches_fixed_metric_encoding_and_preserves_invalid_black():
    rendered = _render([[0, 250, 625, 1000]], "gray")

    np.testing.assert_array_equal(rendered[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(rendered[0, 1], [1, 1, 1])
    np.testing.assert_array_equal(rendered[0, 2], [128, 128, 128])
    np.testing.assert_array_equal(rendered[0, 3], [255, 255, 255])


def test_standard_colour_maps_preserve_invalid_black_and_add_chromatic_channels():
    for colour_map in ("turbo", "viridis", "cividis", "plasma", "inferno", "magma", "jet"):
        rendered = _render([[0, 250, 625, 1000]], colour_map)

        assert rendered.dtype == np.uint8
        assert rendered.shape == (1, 4, 3)
        np.testing.assert_array_equal(rendered[0, 0], [0, 0, 0])
        assert any(len(set(pixel.tolist())) > 1 for pixel in rendered[0, 1:])
