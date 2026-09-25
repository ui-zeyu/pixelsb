from PIL import Image

from pixelsb.domain.classify import Classification
from pixelsb.io.classify import stream_classifier

# Resolved once: the factory loads the model, and every case shares the answer.
classify = stream_classifier()


def _real_png() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, "PNG")
    return buffer.getvalue()


def test_the_model_reads_a_whole_stream() -> None:
    found = classify(_real_png())
    assert found == Classification("png", "image/png", "magika")


def test_a_signature_answers_where_the_model_is_speechless() -> None:
    """A truncated PNG header is below the model's radar; filetype catches it."""
    found = classify(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR")
    assert found == Classification("png", "image/png", "filetype")


def test_a_signature_outranks_the_model_s_text_guess() -> None:
    """``%PDF-`` is printable, so the model says txt; the signature is objective."""
    found = classify(b"%PDF-1.4\n")
    assert found is not None
    assert found.mime == "application/pdf"
    assert found.engine == "filetype"


def test_a_printable_stream_without_a_signature_stays_text() -> None:
    found = classify(b"flag{just_text}\n")
    assert found == Classification("txt", "text/plain", "magika")


def test_the_model_s_text_subtypes_are_normalized_to_plain_text() -> None:
    """A short flag reads as a template to the model; what survives is "text"."""
    assert classify(b"flag{ok}") == Classification("txt", "text/plain", "magika")
    assert classify(b"flag{b3d7bed5-e8da-4d9c-848c-e5d332d63bcd}") == Classification(
        "txt", "text/plain", "magika"
    )


def test_no_signal_answers_none() -> None:
    assert classify(b"") is None
    assert classify(bytes(range(256))) is None
