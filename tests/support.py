"""Small in-memory images for domain tests."""

from pathlib import Path

import numpy as np

from pixelsb.domain.models import LoadedImage, SampleOrigin, SamplePlane


def planes_rgb() -> tuple[SamplePlane, ...]:
    return (
        SamplePlane("R", 0, 8, SampleOrigin.RAW),
        SamplePlane("G", 1, 8, SampleOrigin.RAW),
        SamplePlane("B", 2, 8, SampleOrigin.RAW),
    )


def make_image(
    samples: np.ndarray,
    plane_list: tuple[SamplePlane, ...],
    *,
    preview: np.ndarray | None = None,
    source_mode: str = "RGB",
    path: Path | None = None,
    frame_count: int = 1,
) -> LoadedImage:
    array = np.asarray(samples, dtype=np.uint16)
    height, width, _channels = array.shape
    if preview is None:
        gray = np.ascontiguousarray(array[:, :, 0] & 0xFF, dtype=np.uint8)
        alpha = np.full((height, width), 255, dtype=np.uint8)
        preview_array = np.dstack([gray, gray, gray, alpha])
    else:
        preview_array = np.asarray(preview, dtype=np.uint8)
    return LoadedImage(
        path=path or Path("memory.png"),
        source_mode=source_mode,
        width=width,
        height=height,
        samples=array,
        planes=plane_list,
        preview_rgba=preview_array,
        frame_count=frame_count,
        frame_index=0,
    )
