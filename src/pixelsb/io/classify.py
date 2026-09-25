"""The stream classifier: magika's model first, filetype's signatures as fallback.

Magika reads whole files well but wants more than a bare header — truncated
streams come back ``unknown``, and byte soups led by a signature sometimes come
back as text. That is exactly where filetype shines: its leading-signature match
is objective evidence, so whenever the model is speechless or merely reading
text, the signature gets the final word. The model's text subtypes share that
fate: on the short strings CTF flags are, ``text/x-twig`` and ``text/css`` are
noise, so a text read is kept as plain ``text/plain`` — knowing a stream is
text is worth saying, the subtype is not.
"""

from functools import cache

import filetype
from magika import Magika

from pixelsb.domain.classify import Classification, StreamClassifier

# The labels magika lands on when the model has nothing to say.
_UNINFORMATIVE = frozenset({"unknown", "data"})
# The model's text read: objective signatures outrank it, and a signature-less
# printable stream is still worth naming as text.
_TEXT_GUESS = "txt"
_TEXT_MIME = "text/plain"


@cache
def stream_classifier() -> StreamClassifier:
    """The reader of extracted streams.

    Loading the model is the first call's cost; the cache keeps that a
    once-per-process event and every caller shares the answer.
    """
    engine = Magika()

    def classify(data: bytes) -> Classification | None:
        if not data:
            return None
        result = engine.identify_bytes(data)
        label, mime = result.output.label, result.output.mime_type
        if mime.startswith("text/"):
            label, mime = _TEXT_GUESS, _TEXT_MIME
        if label not in _UNINFORMATIVE and label != _TEXT_GUESS:
            return Classification(label, mime, "magika")
        signature = filetype.match(data)
        if signature is not None:
            return Classification(signature.extension, signature.mime, "filetype")
        if label == _TEXT_GUESS:
            return Classification(label, mime, "magika")
        return None

    return classify
