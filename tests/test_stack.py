"""The layer stack: what each mask does to the pixels, and in which order."""

import numpy as np
import pytest

from pixelsb.domain.models import (
    ArnoldMask,
    BitChoice,
    BitsMask,
    CropMask,
    FftMask,
    GrayscaleMask,
    InvertMask,
    Layer,
    LoadedImage,
    RegionMask,
    SampleOrigin,
    SamplePlane,
    ThresholdMask,
    XorMask,
    level_ceiling,
)
from pixelsb.domain.selection import all_bits, channel_members, lsb_bits
from pixelsb.domain.stack import arnold_image, arnold_indices, match_span, resolve
from tests.support import layers, make_image, planes_rgb, planes_rgba, raster


def _image(samples: list, planes: tuple[SamplePlane, ...] | None = None) -> LoadedImage:
    return make_image(np.array(samples, dtype=np.uint16), planes or planes_rgb())


def test_the_base_raster_is_the_image_itself() -> None:
    image = _image([[[10, 20, 30]]])
    base = resolve(image, ())
    assert base.samples is image.samples
    assert base.selection == all_bits(image.planes)
    assert base.live is None
    assert not base.cropped
    assert base.failures == ()


def test_a_disabled_mask_leaves_the_pixels_alone() -> None:
    image = _image([[[10, 20, 30]]])
    off = resolve(image, layers(InvertMask(), enabled=False))
    assert off.samples is image.samples
    assert resolve(image, layers(InvertMask())).samples[0, 0].tolist() == [245, 235, 225]


def test_a_bits_mask_replaces_the_selection_and_drops_what_is_gone() -> None:
    image = _image([[[1, 2, 3]]])
    stack = layers(BitsMask(frozenset({BitChoice("R", 0), BitChoice("R", 9), BitChoice("X", 0)})))
    assert resolve(image, stack).selection == frozenset({BitChoice("R", 0)})


def test_a_region_mask_marks_the_pixels_it_keeps() -> None:
    image = _image([[[0, 0, 0], [9, 9, 9]], [[9, 9, 9], [0, 0, 0]]])
    kept = resolve(image, layers(RegionMask("R > 0"))).live
    assert kept is not None
    assert kept.tolist() == [[False, True], [True, False]]


def test_region_masks_keep_only_what_every_one_of_them_passes() -> None:
    image = _image([[[5, 0, 0], [9, 0, 0]], [[0, 0, 0], [7, 0, 0]]])
    stack = layers(RegionMask("R >= 5"), RegionMask("top == 0"))
    kept = resolve(image, stack).live
    assert kept is not None
    assert kept.tolist() == [[True, True], [False, False]]


def test_a_region_mask_reads_what_the_masks_below_it_left() -> None:
    """The same expression, on either side of a threshold, keeps other pixels."""
    image = _image([[[200, 0, 0], [40, 0, 0]]])
    below = layers(ThresholdMask(level=100), RegionMask("R == 255"))
    above = layers(RegionMask("R == 255"), ThresholdMask(level=100))
    lower = resolve(image, below).live
    upper = resolve(image, above).live
    assert lower is not None
    assert upper is not None
    assert lower.tolist() == [[True, False]]  # pushed to 255 first
    assert upper.tolist() == [[False, False]]  # 200 is not 255 yet


def test_a_region_mask_that_will_not_compile_is_reported_and_keeps_everything() -> None:
    image = _image([[[1, 0, 0]]])
    stack = layers(RegionMask("R >= "), InvertMask())
    broken = resolve(image, stack)
    assert broken.live is None
    assert broken.samples[0, 0].tolist() == [254, 255, 255]  # the stack carried on
    (failure,) = broken.failures
    assert failure.index == 0
    assert "语法错误" in failure.message


def test_a_region_mask_that_will_not_run_is_reported_too() -> None:
    image = _image([[[1, 0, 0]]])
    broken = resolve(image, layers(RegionMask("R == " + "9" * 30)))
    assert broken.live is None
    assert "超出范围" in broken.failures[0].message


def test_inverting_reads_every_channel_upside_down_at_its_own_maximum() -> None:
    planes = (SamplePlane("L", 0, 16, SampleOrigin.RAW),)
    image = make_image(np.array([[[1000]]], dtype=np.uint16), planes)
    inverted = raster(image, InvertMask())
    assert inverted.samples[0, 0].tolist() == [64535]


def test_grayscale_sets_every_color_channel_to_the_brightness() -> None:
    image = _image([[[255, 0, 0], [0, 255, 0]]])
    gray = raster(image, GrayscaleMask())
    assert gray.samples[0, 0].tolist() == [54, 54, 54]  # 2126/10000 of 255
    assert gray.samples[0, 1].tolist() == [182, 182, 182]


def test_grayscale_leaves_an_alpha_channel_and_a_gray_picture_alone() -> None:
    rgba = make_image(np.full((1, 1, 4), 200, dtype=np.uint16), planes_rgba())
    assert raster(rgba, GrayscaleMask()).samples[0, 0].tolist() == [200, 200, 200, 200]
    planes = (SamplePlane("L", 0, 8, SampleOrigin.RAW),)
    gray = make_image(np.array([[[7]]], dtype=np.uint16), planes)
    assert raster(gray, GrayscaleMask()).samples is gray.samples


def test_threshold_pushes_each_channel_to_zero_or_its_maximum() -> None:
    image = _image([[[100, 101, 0]]])
    pushed = raster(image, ThresholdMask(level=100))
    assert pushed.samples[0, 0].tolist() == [0, 255, 0]


def test_xor_flips_the_bits_the_constant_sets() -> None:
    image = _image([[[0b1010, 0b0101, 0]]])
    folded = raster(image, XorMask(0b1100))
    assert folded.samples[0, 0].tolist() == [0b0110, 0b1001, 0b1100]


def test_cropping_crops_to_the_pixels_that_are_left() -> None:
    image = _image(
        [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        ]
    )
    cropped = raster(image, RegionMask("left >= 1 and left <= 2"), CropMask())
    assert (cropped.width, cropped.height) == (2, 2)
    assert cropped.columns is not None
    assert cropped.rows is not None
    assert cropped.columns.tolist() == [1, 2]
    assert cropped.rows.tolist() == [0, 1]
    assert cropped.samples[0].tolist() == [[1, 0, 0], [2, 0, 0]]
    assert cropped.cropped


def test_cropping_twice_crops_again_from_what_is_already_there() -> None:
    image = _image([[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]]])
    stack = layers(
        RegionMask("left >= 1"),
        CropMask(),
        RegionMask("left <= 3"),
        CropMask(),
    )
    cropped = resolve(image, stack)
    assert cropped.columns is not None
    assert cropped.columns.tolist() == [1, 2, 3]
    assert cropped.source_at(0, 0).x == 1


def test_cropping_with_nothing_left_keeps_no_cells_at_all() -> None:
    image = _image([[[1, 0, 0]]])
    empty = raster(image, RegionMask("R > 200"), CropMask())
    assert (empty.width, empty.height) == (0, 0)
    assert empty.live_count == 0
    assert empty.cropped


def test_cropping_without_a_region_mask_has_nothing_to_crop_to() -> None:
    image = _image([[[1, 0, 0]]])
    alone = raster(image, CropMask())
    assert alone.samples is image.samples
    assert not alone.cropped


def test_match_span_covers_the_rows_and_columns_a_mask_touches() -> None:
    live = np.zeros((4, 5), dtype=bool)
    live[1, 3] = True
    live[3, 0] = True
    span = match_span(live)
    assert span is not None
    rows, columns = span
    assert rows.tolist() == [1, 3]
    assert columns.tolist() == [0, 3]
    assert match_span(np.zeros((2, 2), dtype=bool)) is None


def test_a_value_mask_never_writes_into_the_image() -> None:
    image = _image([[[10, 20, 30]]])
    raster(image, InvertMask(), ThresholdMask(level=1), GrayscaleMask(), XorMask(0xFF))
    assert image.samples[0, 0].tolist() == [10, 20, 30]


@pytest.mark.parametrize("level", [-1, 65536])
def test_a_threshold_outside_the_range_is_refused(level: int) -> None:
    with pytest.raises(ValueError, match=r"0\.\.65535"):
        ThresholdMask(level=level)


@pytest.mark.parametrize("value", [-1, 65536])
def test_an_xor_value_outside_the_range_is_refused(value: int) -> None:
    with pytest.raises(ValueError, match=r"0\.\.65535"):
        XorMask(value=value)


def test_a_threshold_splits_a_sixteen_bit_plane_at_its_own_scale() -> None:
    planes = (SamplePlane("L", 0, 16, SampleOrigin.RAW),)
    image = make_image(np.array([[[1000], [40000]]], dtype=np.uint16), planes)
    pushed = raster(image, ThresholdMask(level=20000))
    assert pushed.samples[0, :, 0].tolist() == [0, 65535]
    assert level_ceiling(planes) == 65535
    assert level_ceiling(planes_rgb()) == 255
    assert level_ceiling(()) == 255


def test_a_bits_mask_naming_a_plane_the_image_lacks_selects_nothing() -> None:
    image = _image([[[1, 0, 0]]])
    stack = layers(BitsMask(frozenset({BitChoice("A", 0)})))
    assert resolve(image, stack).selection == frozenset()


def test_the_stack_starts_from_every_lowest_bit_when_it_is_asked_to() -> None:
    image = _image([[[1, 2, 3]]])
    stack = (Layer(BitsMask(lsb_bits(image.planes))),)
    assert resolve(image, stack).selection == frozenset(
        {BitChoice("R", 0), BitChoice("G", 0), BitChoice("B", 0)}
    )


def _encode(samples: np.ndarray, times: int, a: int, b: int) -> np.ndarray:
    """The forward cat map, the way a challenge would have scrambled the picture."""
    size = samples.shape[0]
    columns = np.arange(size, dtype=np.int64)[None, :]
    rows = np.arange(size, dtype=np.int64)[:, None]
    out = samples
    for _ in range(times):
        new_x = (columns + b * rows) % size
        new_y = (a * columns + (a * b + 1) * rows) % size
        moved = np.empty_like(out)
        moved[new_y, new_x] = out
        out = moved
    return out


def test_the_cat_map_moves_pixels_and_undoes_the_forward_scramble() -> None:
    """The operation is the recovery direction: same parameters as the encoder undo it."""
    samples = np.arange(75, dtype=np.uint16).reshape(5, 5, 3)
    scrambled = _encode(samples, 2, 1, 3)
    assert sorted(scrambled.ravel().tolist()) == sorted(samples.ravel().tolist())
    assert np.array_equal(arnold_image(scrambled, 2, 1, 3), samples)


def test_the_cat_maps_grids_name_every_cell_exactly_once() -> None:
    source_y, source_x = arnold_indices(6, 3, 2, 1)
    landed = {(int(y), int(x)) for y, x in zip(source_y.ravel(), source_x.ravel(), strict=True)}
    assert len(landed) == 36


def test_the_cat_maps_grids_match_composing_the_map_step_by_step() -> None:
    """Repeated squaring raises the map's matrix; the answer is the one composed."""
    rng = np.random.default_rng(7)
    for _case in range(8):
        size = int(rng.integers(2, 12))
        times = int(rng.integers(1, 40))
        a, b = int(rng.integers(-100, 100)), int(rng.integers(-100, 100))
        columns = np.arange(size, dtype=np.int64)[None, :]
        rows = np.arange(size, dtype=np.int64)[:, None]
        # The recovery map [[ab+1, -b], [-a, 1]], applied one step at a time.
        x, y = (a * b + 1) * columns - b * rows, -a * columns + rows
        x, y = x % size, y % size
        for _step in range(times - 1):
            x, y = ((a * b + 1) * x - b * y) % size, (-a * x + y) % size
        source_y, source_x = arnold_indices(size, times, a, b)
        assert np.array_equal(source_x, x)
        assert np.array_equal(source_y, y)


def test_the_cat_map_mask_needs_the_whole_square() -> None:
    image = _image(np.zeros((3, 3, 3), dtype=np.uint16).tolist())
    assert resolve(image, (Layer(ArnoldMask(1, 1, 1)),)).failures == ()
    tall = _image(np.zeros((4, 3, 3), dtype=np.uint16).tolist())
    (failure,) = resolve(tall, (Layer(ArnoldMask(1, 1, 1)),)).failures
    assert "方图" in failure.message
    stack = (Layer(RegionMask("left < 2")), Layer(CropMask()), Layer(ArnoldMask(1, 1, 1)))
    (failure,) = resolve(image, stack).failures
    assert "完整画幅" in failure.message


def test_the_cat_map_carries_the_live_mask_with_the_pixels() -> None:
    image = _image([[[255, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])  # 2x2
    stack = (Layer(RegionMask("left == 0 and top == 0")), Layer(ArnoldMask(1, 1, 1)))
    kept = resolve(image, stack)
    assert kept.live is not None
    assert kept.live.sum() == 1  # the same pixel, wherever the map put it


def test_the_spectrum_of_a_constant_channel_sits_at_the_center() -> None:
    image = _image([[[9, 9, 9]] * 4] * 4)  # 4x4, flat
    shown = raster(image, FftMask())
    assert int(shown.samples[2, 2, 0]) == 255  # fftshift puts DC in the middle
    assert int(shown.samples[0, 0, 0]) < 10  # a flat picture has nothing else to say


def test_the_spectrum_of_an_impulse_is_flat() -> None:
    flat = [[[0, 0, 0]] * 4 for _ in range(4)]
    flat[1][2] = [200, 200, 200]
    shown = raster(_image(flat), FftMask())
    assert shown.samples[:, :, 0].min() == shown.samples[:, :, 0].max()


def test_the_spectrum_takes_the_channels_the_selection_carries() -> None:
    image = _image([[[9, 9, 9, 40]] * 4 for _ in range(4)], planes_rgba())
    keep = all_bits(image.planes) - channel_members(image.planes, "A")
    shown = raster(image, BitsMask(keep), FftMask())
    assert shown.samples[0, 0, 3] == 40  # alpha had no bit selected: it keeps its value
    assert int(shown.samples[2, 2, 0]) == 255  # the colors still take their spectrum


def test_a_spectrum_over_no_selection_changes_nothing() -> None:
    image = _image([[[9, 9, 9]] * 4 for _ in range(4)])
    shown = raster(image, BitsMask(frozenset()), FftMask())
    assert shown.samples[2, 2, 0] == 9  # nothing takes part, so nothing is transformed


def test_with_every_bit_selected_alpha_takes_a_spectrum_too() -> None:
    image = _image([[[9, 9, 9, 40]] * 4 for _ in range(4)], planes_rgba())
    shown = raster(image, FftMask())
    assert int(shown.samples[2, 2, 3]) == 255  # the default selection carries alpha
