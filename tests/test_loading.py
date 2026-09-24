from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixelsb.domain.models import SampleOrigin
from pixelsb.io.loading import ImageLoadError, load_image


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
    frames = []
    for index in range(3):
        frame = Image.new("P", (1, 1), color=index)
        frame.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * 759)
        frames.append(frame)
    path = tmp_path / "anim.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    loaded = load_image(path)
    assert loaded.frame_count == 3
    assert loaded.frame_index == 0
    assert int(loaded.samples[0, 0, loaded.plane("Index").index]) == 0


def test_missing_and_undecodable_files_raise(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "missing.png")
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    with pytest.raises(ImageLoadError):
        load_image(notes)
