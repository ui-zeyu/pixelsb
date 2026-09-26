from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixelsb.domain.models import SampleOrigin
from pixelsb.io.loading import (
    ImageLoadError,
    frame_geometries,
    image_from_pixels,
    load_frame,
    load_image,
)


def test_image_from_pixels_keeps_a_fourth_channel() -> None:
    """A stream rendered with alpha becomes an image with an alpha plane."""
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = 200
    image = image_from_pixels(Path("rendered.png"), rgba)
    assert image.source_mode == "RGBA"
    assert [plane.name for plane in image.planes] == ["R", "G", "B", "A"]
    assert image.samples[0, 0].tolist() == [0, 0, 0, 200]
    plain = image_from_pixels(Path("plain.png"), np.zeros((2, 3, 3), dtype=np.uint8))
    assert plain.source_mode == "RGB"
    assert [plane.name for plane in plain.planes] == ["R", "G", "B"]
    with pytest.raises(ValueError, match="HxWx3 or HxWx4"):
        image_from_pixels(Path("bad.png"), np.zeros((2, 3, 2), dtype=np.uint8))


def test_a_color_key_png_gains_an_alpha_plane(tmp_path: Path) -> None:
    """A PNG can name one transparent sample instead of carrying a channel."""
    rgb = np.array([[[10, 20, 30], [1, 2, 3]]], dtype=np.uint8)
    path = tmp_path / "keyed.png"
    Image.fromarray(rgb, "RGB").save(path, transparency=(10, 20, 30))
    image = load_image(path)
    assert [plane.name for plane in image.planes] == ["R", "G", "B", "A"]
    assert image.samples[0, 0].tolist() == [10, 20, 30, 0]  # the key is the clear one
    assert image.samples[0, 1].tolist() == [1, 2, 3, 255]


def test_a_gray_color_key_png_gains_an_alpha_plane(tmp_path: Path) -> None:
    path = tmp_path / "keyed-gray.png"
    Image.fromarray(np.array([[7, 9]], dtype=np.uint8), "L").save(path, transparency=7)
    image = load_image(path)
    assert [plane.name for plane in image.planes] == ["L", "A"]
    assert image.samples[0, 0].tolist() == [7, 0]
    assert image.samples[0, 1].tolist() == [9, 255]


def test_rgb_and_rgba_keep_raw_channels(tmp_path: Path) -> None:
    rgb = Image.new("RGB", (1, 1), (1, 2, 3))
    rgb_path = tmp_path / "rgb.png"
    rgb.save(rgb_path)
    loaded = load_image(rgb_path)
    assert [plane.name for plane in loaded.planes] == ["R", "G", "B"]
    assert loaded.samples[0, 0].tolist() == [1, 2, 3]
    assert not loaded.samples.flags.writeable

    rgba = Image.new("RGBA", (1, 1), (1, 2, 3, 4))
    rgba_path = tmp_path / "rgba.png"
    rgba.save(rgba_path)
    loaded_rgba = load_image(rgba_path)
    assert loaded_rgba.samples[0, 0].tolist() == [1, 2, 3, 4]
    assert loaded_rgba.plane("A").origin is SampleOrigin.RAW


def test_mode_one_uses_a_single_bit(tmp_path: Path) -> None:
    image = Image.new("1", (2, 1))
    image.putpixel((0, 0), 1)
    image.putpixel((1, 0), 0)
    path = tmp_path / "bit.png"
    image.save(path)
    loaded = load_image(path)
    assert loaded.planes[0] == loaded.plane("L")
    assert loaded.planes[0].bit_depth == 1
    assert int(loaded.samples[0, 0, 0]) == 1
    assert int(loaded.samples[0, 1, 0]) == 0


def test_la_keeps_luminance_and_alpha(tmp_path: Path) -> None:
    image = Image.new("LA", (1, 1), (128, 64))
    path = tmp_path / "la.png"
    image.save(path)
    loaded = load_image(path)
    assert loaded.samples[0, 0].tolist() == [128, 64]
    assert [plane.origin for plane in loaded.planes] == [SampleOrigin.RAW, SampleOrigin.RAW]


def test_palette_index_is_separate_from_the_looked_up_color(tmp_path: Path) -> None:
    image = Image.new("P", (2, 1))
    palette = [0] * 768
    palette[0:6] = [255, 0, 0, 0, 0, 255]
    image.putpalette(palette)
    image.putpixel((0, 0), 1)
    image.putpixel((1, 0), 0)
    image.info["transparency"] = 0
    path = tmp_path / "palette.png"
    image.save(path)
    loaded = load_image(path)
    assert loaded.plane("Index").origin is SampleOrigin.RAW
    assert loaded.plane("R").origin is SampleOrigin.PALETTE
    assert int(loaded.samples[0, 0, loaded.plane("Index").index]) == 1
    assert int(loaded.samples[0, 0, loaded.plane("R").index]) == 0
    assert int(loaded.samples[0, 0, loaded.plane("B").index]) == 255
    assert int(loaded.samples[0, 1, loaded.plane("A").index]) == 0


def test_sixteen_bit_png_and_big_endian_tiff_keep_numeric_samples(tmp_path: Path) -> None:
    png = Image.frombytes("I;16", (2, 1), np.array([[0x1234, 0x0100]], dtype=np.uint16).tobytes())
    png_path = tmp_path / "gray.png"
    png.save(png_path)
    loaded_png = load_image(png_path)
    assert loaded_png.source_mode == "I;16"
    assert loaded_png.planes[0].bit_depth == 16
    assert int(loaded_png.samples[0, 0, 0]) == 0x1234
    assert int(loaded_png.samples[0, 1, 0]) == 0x0100

    tiff = Image.new("I;16B", (2, 1))
    tiff.putpixel((0, 0), 0x1234)
    tiff.putpixel((1, 0), 0x00FF)
    tiff_path = tmp_path / "gray.tif"
    tiff.save(tiff_path)
    loaded_tiff = load_image(tiff_path)
    assert loaded_tiff.source_mode == "I;16B"
    assert int(loaded_tiff.samples[0, 0, 0]) == 0x1234
    assert int(loaded_tiff.samples[0, 1, 0]) == 0x00FF


def test_cmyk_is_marked_converted(tmp_path: Path) -> None:
    image = Image.new("CMYK", (1, 1), (10, 20, 30, 40))
    path = tmp_path / "cmyk.tif"
    image.save(path)
    loaded = load_image(path)
    assert loaded.source_mode == "CMYK"
    assert [plane.name for plane in loaded.planes] == ["R", "G", "B", "A"]
    assert all(plane.origin is SampleOrigin.CONVERTED for plane in loaded.planes)


def test_gif_reports_every_frame_and_decodes_the_first(tmp_path: Path) -> None:
    _animated_gif(tmp_path)
    loaded = load_image(tmp_path / "anim.gif")
    assert loaded.frame_count == 3
    assert loaded.frame_index == 0
    assert loaded.frame_delays == (40, 50, 60)
    assert int(loaded.samples[0, 0, loaded.plane("Index").index]) == 0


def test_load_frame_decodes_the_frame_it_is_asked_for(tmp_path: Path) -> None:
    _animated_gif(tmp_path)
    second = load_frame(tmp_path / "anim.gif", 1)
    assert second.frame_index == 1
    assert second.frame_delays == (40, 50, 60)
    # PIL hands later GIF frames out in their own mode; what matters is the
    # pixel: palette color 1 is the red entry.
    assert second.samples[0, 0].tolist() == [255, 0, 0]
    first = load_frame(tmp_path / "anim.gif", 0)
    assert int(first.samples[0, 0, first.planes[0].index]) == 0


def test_load_frame_refuses_indexes_outside_the_image(tmp_path: Path) -> None:
    _animated_gif(tmp_path)
    with pytest.raises(ImageLoadError):
        load_frame(tmp_path / "anim.gif", 3)
    with pytest.raises(ImageLoadError):
        load_frame(tmp_path / "anim.gif", -1)


def test_a_single_frame_image_reports_no_delays(tmp_path: Path) -> None:
    image = Image.new("RGB", (1, 1))
    path = tmp_path / "still.png"
    image.save(path)
    loaded = load_image(path)
    assert loaded.frame_delays == ()
    with pytest.raises(ImageLoadError):
        load_frame(path, 1)


def _animated_gif(tmp_path: Path) -> list[Image.Image]:
    frames = []
    for index in range(3):
        frame = Image.new("P", (1, 1), color=index)
        frame.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * 759)
        frames.append(frame)
    path = tmp_path / "anim.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[40, 50, 60], loop=0)
    return frames


def test_missing_and_undecodable_files_raise(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "missing.png")
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    with pytest.raises(ImageLoadError):
        load_image(notes)


def test_frame_geometries_read_offset_frames(tmp_path: Path) -> None:
    """A GIF places each frame on the canvas: the table gives its rectangle and delay."""
    base = Image.new("P", (8, 8), 0)
    base.putpalette([0, 0, 0, 255, 255, 255, 255, 0, 0] + [0] * 759)
    moved = base.copy()
    for x in range(2, 5):
        for y in range(2, 5):
            moved.putpixel((x, y), 1)
    path = tmp_path / "moved.gif"
    base.save(path, save_all=True, append_images=[moved], duration=[50, 70], loop=0)
    geometries = frame_geometries(path)
    assert [(g.width, g.height, g.x, g.y) for g in geometries] == [(8, 8, 0, 0), (3, 3, 2, 2)]
    assert [g.delay for g in geometries] == [50, 70]


def test_frame_geometries_of_a_still_image_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "still.png"
    Image.new("RGB", (4, 4)).save(path)
    assert frame_geometries(path) == ()


def test_a_doctored_ihdr_opens_under_the_geometry_its_data_fills(tmp_path: Path) -> None:
    """Declared 10x10, data for 5x1: the loader repairs in memory, the file stays put."""
    path = tmp_path / "doctored.png"
    pixels = np.zeros((1, 5, 3), dtype=np.uint8)
    pixels[0, :, 0] = np.arange(5, dtype=np.uint8) * 40
    Image.fromarray(pixels).save(path)
    data = bytearray(path.read_bytes())
    at = data.find(b"IHDR") + 4
    data[at : at + 8] = (10).to_bytes(4, "big") + (10).to_bytes(4, "big")
    path.write_bytes(bytes(data))
    image = load_image(path)
    assert (image.width, image.height) == (5, 1)
    assert path.read_bytes() == bytes(data)  # the file on disk is exactly as it was


def test_a_small_declared_size_still_loads_as_declared(tmp_path: Path) -> None:
    """Extra data behind a small IHDR loads as declared; the census tells the rest."""
    path = tmp_path / "small.png"
    pixels = np.zeros((1, 5, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    image = load_image(path)
    assert (image.width, image.height) == (5, 1)
