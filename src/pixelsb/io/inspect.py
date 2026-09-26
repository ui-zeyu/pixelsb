"""Container checks: what the file's own bytes say beyond the pixels."""

from pathlib import Path

from PIL import Image
from PIL.ExifTags import GPSTAGS, IFD, TAGS

from pixelsb.domain.container import ContainerReport, scan_container
from pixelsb.domain.formatting import clip

_EXIF_TEXT_LIMIT = 96
# The sub-IFD pointers sit in the base IFD; their tables are read separately.
_IFD_POINTERS = frozenset({IFD.Exif, IFD.GPSInfo, IFD.Interop, IFD.MakerNote, IFD.IFD1})


def inspect_container(path: Path) -> ContainerReport:
    """Census a file's structure: every block, and everything suspicious."""
    return scan_container(Path(path).read_bytes())


def exif_entries(path: Path) -> tuple[tuple[str, str], ...]:
    """The container's EXIF tags, base IFD plus the EXIF and GPS sub-IFDs.

    A file too broken for PIL to open still gets its page, so an unreadable
    container reads as no EXIF rather than an error.
    """
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            base = ((tag, value) for tag, value in exif.items() if tag not in _IFD_POINTERS)
            groups = (
                (base, TAGS),
                (exif.get_ifd(IFD.Exif).items(), TAGS),
                (exif.get_ifd(IFD.GPSInfo).items(), GPSTAGS),
            )
            return tuple(
                (table.get(tag, str(tag)), _exif_text(value))
                for items, table in groups
                for tag, value in items
            )
    except Exception:
        return ()


def _exif_text(value: object) -> str:
    if isinstance(value, bytes):
        text = repr(value)
    elif isinstance(value, tuple):
        text = ", ".join(_exif_text(item) for item in value)
    else:
        text = str(value)
    return clip(text, _EXIF_TEXT_LIMIT)
