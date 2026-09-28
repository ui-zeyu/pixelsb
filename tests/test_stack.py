"""The layer stack: what each mask does to the pixels, and in which order."""

from pathlib import Path

import numpy as np
import pytest

from pixelsb.domain.models import (
    ArnoldMask,
    BitChoice,
    BitsMask,
    CombineMask,
    CombineOp,
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
from pixelsb.domain.selection import channel_members
from pixelsb.domain.stack import arnold_image, arnold_indices, match_span, resolve
from tests.support import layers, make_image, planes_rgb, planes_rgba, raster


def _image(samples: list, planes: tuple[SamplePlane, ...] | None = None) -> LoadedImage:
    return make_image(np.array(samples, dtype=np.uint16), planes or planes_rgb())


def test_the_base_raster_is_the_image_itself() -> None:
    image = _image([[[10, 20, 30]]])
    base = resolve(image, ())
    assert base.samples is image.samples
    assert base.planes == image.planes
    assert base.live is None
    assert not base.cropped
    assert base.failures == ()


def test_a_disabled_mask_leaves_the_pixels_alone() -> None:
    image = _image([[[10, 20, 30]]])
    off = resolve(image, layers(InvertMask(), enabled=False))
    assert off.samples is image.samples
    assert resolve(image, layers(InvertMask())).samples[0, 0].tolist() == [245, 235, 225]


def test_a_projection_packs_the_chosen_bits_and_shrinks_the_depth() -> None:
    image = _image([[[0b0000_0101, 0, 0]]])
    one = raster(image, BitsMask(frozenset({BitChoice("R", 0)})))
    assert one.samples[0, 0, 0] == 1
    assert one.planes[0].bit_depth == 1  # a one-bit projection is a one-bit channel
    two = raster(image, BitsMask(frozenset({BitChoice("R", 2), BitChoice("R", 0)})))
    assert two.samples[0, 0, 0] == 0b11  # bits 0 and 2 packed onto the low end
    assert two.planes[0].bit_depth == 2


def test_a_projection_keeps_only_the_channels_it_names() -> None:
    image = _image([[[1, 2, 3]]])
    shown = raster(image, BitsMask(frozenset({BitChoice("B", 0)})))
    assert tuple(plane.name for plane in shown.planes) == ("B",)
    assert shown.samples.shape == (1, 1, 1)


def test_an_operation_above_a_projection_reads_the_projected_picture() -> None:
    """The pipeline rule: every mask takes the one below it as its input."""
    image = _image([[[0b101, 0, 0], [0b001, 0, 0]]])
    pushed = raster(image, BitsMask(frozenset({BitChoice("R", 0)})), ThresholdMask(level=0))
    assert pushed.samples[:, :, 0].tolist() == [[1, 1]]  # the 1-bit plane's own maximum
    assert pushed.planes[0].bit_depth == 1


def test_a_region_above_a_projection_filters_on_the_packed_values() -> None:
    image = _image([[[0b100, 0, 0], [0b101, 0, 0]]])
    kept = raster(
        image,
        BitsMask(frozenset({BitChoice("R", 2), BitChoice("R", 0)})),
        RegionMask("R == 2"),
    ).live
    assert kept is not None
    assert kept.tolist() == [[True, False]]  # packed values 2 and 3


def test_a_predicate_above_a_projection_sees_the_shrunken_depth() -> None:
    image = _image([[[0b101, 0, 0]]])
    broken = raster(image, BitsMask(frozenset({BitChoice("R", 0)})), RegionMask("R.3 == 1"))
    (failure,) = broken.failures
    assert "只有 1 位" in failure.message
    assert broken.live is None


def test_a_projection_naming_a_channel_this_image_lacks_fails_the_layer() -> None:
    image = _image([[[1, 2, 3]]])
    broken = raster(image, BitsMask(frozenset({BitChoice("R", 0), BitChoice("X", 0)})))
    (failure,) = broken.failures
    assert "位选择的通道这张图没有" in failure.message
    assert broken.samples[0, 0].tolist() == [1, 2, 3]  # the refused layer leaves the pixels


def test_a_projection_of_no_bits_fails_the_layer() -> None:
    image = _image([[[1, 2, 3]]])
    broken = raster(image, BitsMask(frozenset()))
    (failure,) = broken.failures
    assert "位选择是空的" in failure.message


def test_a_projection_over_a_projection_projects_the_packed_picture() -> None:
    image = _image([[[0b110, 0, 0]]])
    shown = raster(
        image,
        BitsMask(channel_members(image.planes, "R")),  # the whole channel, unchanged
        BitsMask(frozenset({BitChoice("R", 0)})),  # bit 0 of it
    )
    assert shown.samples[0, 0, 0] == 0
    assert shown.planes[0].bit_depth == 1


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


def test_combining_xors_the_planes_two_pictures_share() -> None:
    base = _image([[[10, 20, 30], [40, 50, 60]]])
    other = make_image(np.full((1, 2, 3), 15, dtype=np.uint16), planes_rgb(), path=Path("b.png"))
    combined = raster(base, CombineMask(other))
    assert combined.samples[0, 0].tolist() == [10 ^ 15, 20 ^ 15, 30 ^ 15]
    assert combined.samples[0, 1].tolist() == [40 ^ 15, 50 ^ 15, 60 ^ 15]
    assert combined.planes == base.planes


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (CombineOp.XOR, [10 ^ 99, 200 ^ 99]),
        (CombineOp.AND, [10 & 99, 200 & 99]),
        (CombineOp.OR, [10 | 99, 200 | 99]),
        (CombineOp.MIN, [10, 99]),
        (CombineOp.MAX, [99, 200]),
        (CombineOp.ADD, [(10 + 99) // 2, (200 + 99) // 2]),
        # (10 - 99) / 2 truncates toward zero: -44, where flooring would say -45.
        (CombineOp.SUB, [(10 - 99) // 2 + 128 + 1, (200 - 99) // 2 + 128]),
    ],
)
def test_each_combine_operation_reads_as_stegsolve_does(op: CombineOp, expected: list[int]) -> None:
    base = _image([[[10, 200, 30]]])
    other = make_image(np.full((1, 1, 3), 99, dtype=np.uint16), planes_rgb(), path=Path("b.png"))
    combined = raster(base, CombineMask(other, op))
    assert combined.samples[0, 0, :2].tolist() == expected


def test_a_combine_subtracting_equal_planes_reads_flat() -> None:
    base = _image([[[70, 70, 70]]])
    other = make_image(np.full((1, 1, 3), 70, dtype=np.uint16), planes_rgb(), path=Path("b.png"))
    combined = raster(base, CombineMask(other, CombineOp.SUB))
    assert combined.samples[0, 0, 0] == 128


def test_a_combine_keeps_the_planes_only_one_side_carries() -> None:
    base = make_image(np.array([[[10, 20, 30, 5]]], dtype=np.uint16), planes_rgba())
    other = make_image(np.array([[[15, 15, 15]]], dtype=np.uint16), planes_rgb())
    combined = raster(base, CombineMask(other))
    assert combined.samples[0, 0].tolist() == [10 ^ 15, 20 ^ 15, 30 ^ 15, 5]
    assert tuple(plane.name for plane in combined.planes) == ("R", "G", "B", "A")


def test_a_combine_drops_the_planes_only_the_other_picture_has() -> None:
    base = _image([[[10, 20, 30]]])
    other = make_image(np.array([[[15, 15, 15, 250]]], dtype=np.uint16), planes_rgba())
    combined = raster(base, CombineMask(other))
    assert combined.samples[0, 0].tolist() == [10 ^ 15, 20 ^ 15, 30 ^ 15]
    assert tuple(plane.name for plane in combined.planes) == ("R", "G", "B")


def test_a_combine_meets_a_deeper_plane_at_its_own_depth() -> None:
    base = make_image(np.array([[[200]]], dtype=np.uint16), planes_rgb()[:1])
    deep = (SamplePlane("R", 0, 16, SampleOrigin.RAW),)
    other = make_image(np.array([[[300]]], dtype=np.uint16), deep)
    combined = raster(base, CombineMask(other))
    assert combined.samples[0, 0, 0] == 200 ^ 300
    assert combined.planes[0].bit_depth == 16


def test_a_combine_subtracts_around_the_widened_plane_s_midpoint() -> None:
    """A 16-bit meeting lifts the flat point to 32768, the depth's own middle."""
    base = make_image(np.array([[[200]]], dtype=np.uint16), planes_rgb()[:1])
    deep = (SamplePlane("R", 0, 16, SampleOrigin.RAW),)
    other = make_image(np.array([[[300]]], dtype=np.uint16), deep)
    combined = raster(base, CombineMask(other, CombineOp.SUB))
    assert combined.samples[0, 0, 0] == 32768 - 50
    assert combined.planes[0].bit_depth == 16


def test_two_combines_of_one_picture_are_alike_and_others_are_not() -> None:
    first = _image([[[10, 20, 30]]])
    second = _image([[[1, 2, 3]]])
    assert CombineMask(first) == CombineMask(first)
    assert CombineMask(first, CombineOp.XOR) == CombineMask(first)
    assert CombineMask(first) != CombineMask(second)


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


def _encode(samples: np.ndarray, times: int, a: int, b: int) -> np.ndarray:
    """The forward cat map as the common arnold_encode scripts write it.

    The matrix acts on ``(row, column)`` — ``a`` multiplies the row on the
    second line, ``b`` the column on the first — and the app's recovery has to
    read the same parameters in the same order to undo it.
    """
    size = samples.shape[0]
    rows = np.arange(size, dtype=np.int64)[:, None]
    columns = np.arange(size, dtype=np.int64)[None, :]
    out = samples
    for _ in range(times):
        new_row = (rows + b * columns) % size
        new_column = (a * rows + (a * b + 1) * columns) % size
        moved = np.empty_like(out)
        moved[new_row, new_column] = out
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


def test_the_cat_map_matches_the_arnold_decode_scripts() -> None:
    """The parameters a challenge's own script takes decode the same picture here.

    Transcribed straight from the loop every writeup ships: the new coordinates
    are computed over ``(row, column)`` and the destination cell takes the
    source's pixel. Huge int32 coefficients included.
    """
    rng = np.random.default_rng(9)
    samples = rng.integers(0, 4, (8, 8, 3)).astype(np.uint16)
    size = 8
    rows = np.arange(size, dtype=np.int64)[:, None]
    columns = np.arange(size, dtype=np.int64)[None, :]
    for a, b in ((1, 2), (-3, 5), (0x729E, 0x6F6C53)):
        script = np.zeros_like(samples)
        script[((a * b + 1) * rows - b * columns) % size, (-a * rows + columns) % size] = samples
        assert np.array_equal(arnold_image(samples, 1, a, b), script)


def test_the_cat_maps_grids_match_composing_the_map_step_by_step() -> None:
    """Repeated squaring raises the map's matrix; the answer is the one composed."""
    rng = np.random.default_rng(7)
    for _case in range(8):
        size = int(rng.integers(2, 12))
        times = int(rng.integers(1, 40))
        a, b = int(rng.integers(-100, 100)), int(rng.integers(-100, 100))
        rows = np.arange(size, dtype=np.int64)[:, None]
        columns = np.arange(size, dtype=np.int64)[None, :]
        # The recovery map [[ab+1, -b], [-a, 1]], applied to (row, column) one
        # step at a time, which is the arnold_decode script's own order.
        row, column = (a * b + 1) * rows - b * columns, -a * rows + columns
        row, column = row % size, column % size
        for _step in range(times - 1):
            row, column = ((a * b + 1) * row - b * column) % size, (-a * row + column) % size
        source_row, source_column = arnold_indices(size, times, a, b)
        assert np.array_equal(source_row, row)
        assert np.array_equal(source_column, column)


def test_the_cat_map_needs_a_square_block_of_samples() -> None:
    image = _image(np.zeros((3, 3, 3), dtype=np.uint16).tolist())
    plain = resolve(image, (Layer(ArnoldMask(1, 1, 1)),))
    assert plain.failures == ()
    assert plain.rows is None
    assert plain.columns is None
    assert np.array_equal(plain.samples, arnold_image(image.samples, 1, 1, 1))
    tall = _image(np.zeros((4, 3, 3), dtype=np.uint16).tolist())
    (failure,) = resolve(tall, (Layer(ArnoldMask(1, 1, 1)),)).failures
    assert "方图" in failure.message
    # A crop that left a square is still a whole block of cells to mix.
    cropped = resolve(image, layers(RegionMask("left < 2 and top < 2"), CropMask()))
    mapped = resolve(
        image,
        layers(RegionMask("left < 2 and top < 2"), CropMask(), ArnoldMask(1, 1, 1)),
    )
    assert mapped.failures == ()
    assert mapped.rows is None
    assert mapped.columns is None
    assert np.array_equal(mapped.samples, arnold_image(cropped.samples, 1, 1, 1))
    # A crop that left a rectangle is not.
    stack = (Layer(RegionMask("left < 2")), Layer(CropMask()), Layer(ArnoldMask(1, 1, 1)))
    (failure,) = resolve(image, stack).failures
    assert "方图" in failure.message


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


def test_the_spectrum_takes_the_channels_it_names() -> None:
    image = _image([[[9, 9, 9, 40]] * 4 for _ in range(4)], planes_rgba())
    shown = raster(image, FftMask(("R", "G", "B")))
    assert shown.samples[0, 0, 3] == 40  # alpha was not named: it keeps its value
    assert int(shown.samples[2, 2, 0]) == 255  # the colors take their spectrum


def test_a_spectrum_naming_alpha_does_alpha_too() -> None:
    image = _image([[[9, 9, 9, 40]] * 4 for _ in range(4)], planes_rgba())
    shown = raster(image, FftMask(("A",)))
    assert int(shown.samples[2, 2, 3]) == 255  # alpha takes its spectrum
    assert shown.samples[2, 2, 0] == 9  # the unnamed colors keep their values


def test_a_spectrum_naming_no_channel_changes_nothing() -> None:
    image = _image([[[9, 9, 9]] * 4 for _ in range(4)])
    shown = raster(image, FftMask(()))
    assert shown.samples[2, 2, 0] == 9  # nothing takes part, so nothing is transformed


def test_a_spectrum_naming_a_channel_the_image_lacks_fails_the_layer() -> None:
    image = _image([[[9, 9, 9]] * 4 for _ in range(4)])
    broken = raster(image, FftMask(("R", "X")))
    (failure,) = broken.failures
    assert "频谱的通道这张图没有" in failure.message
    assert broken.samples[2, 2, 0] == 9  # the refused layer leaves the pixels alone


def test_the_spectrum_by_default_leaves_alpha_alone() -> None:
    image = _image([[[9, 9, 9, 40]] * 4 for _ in range(4)], planes_rgba())
    shown = raster(image, FftMask())
    assert int(shown.samples[2, 2, 0]) == 255  # the colors take their spectrum
    assert shown.samples[0, 0, 3] == 40  # alpha does not: its spectrum is a dark picture
