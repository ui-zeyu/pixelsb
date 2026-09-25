"""What an extracted byte stream is: one record, read by whichever engine asks."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Classification:
    """A whole-stream type as one engine read it.

    ``label`` names the format for people ("png"), ``mime`` carries the
    standard spelling, and ``engine`` says who read it ("magika", "filetype").
    """

    label: str
    mime: str
    engine: str


type StreamClassifier = Callable[[bytes], Classification | None]
