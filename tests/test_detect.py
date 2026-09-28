from pixelsb.domain.detect import MAX_DETECTIONS, Detection, detect_patterns, keyword_hits


def test_magic_headers_are_found_at_their_offsets() -> None:
    stream = b"\x89PNG\r\n\x1a\n" + b"\x00" * 5 + b"PK\x03\x04" + b"tail"
    assert detect_patterns(stream) == (
        Detection(0, "PNG", 8),
        Detection(13, "ZIP", 4),
    )


def test_flag_and_ctf_keywords_are_flagged_at_their_offsets() -> None:
    """The keywords are the finding; the body after `{` is the user's to read."""
    stream = b"prefix flag{ab_1} mid DASCTF{d0ne}\x00"
    detections = detect_patterns(stream)
    assert [(finding.label, finding.offset) for finding in detections] == [
        ("flag", 7),
        ("CTF", 25),
    ]
    assert all(finding.flagged for finding in detections)


def test_every_keyword_is_sought_case_insensitively() -> None:
    stream = b"a KEY b Secret c password d FLAG"
    assert [(finding.label, finding.offset) for finding in detect_patterns(stream)] == [
        ("KEY", 2),
        ("Secret", 8),
        ("password", 17),
        ("FLAG", 28),
    ]


def test_every_repeated_hit_is_reported_in_stream_order() -> None:
    stream = b"%PDF-" + b"\x00" * 4 + b"%PDF-"
    assert [finding.offset for finding in detect_patterns(stream)] == [0, 9]


def test_empty_and_clean_streams_find_nothing() -> None:
    assert detect_patterns(b"") == ()
    assert detect_patterns(b"nothing interesting \x00\xff here") == ()


def test_a_pathological_stream_is_capped() -> None:
    stream = b"GIF8" * (MAX_DETECTIONS * 4)
    assert len(detect_patterns(stream)) == MAX_DETECTIONS


def test_keyword_hits_report_the_flags_without_the_signature_sweep() -> None:
    """The sweep judges candidates by keywords alone, so that is all it asks for."""
    stream = b"\x89PNG\r\n\x1a\n" + b"\x00" * 5 + b"prefix flag{ab} "
    assert [(hit.label, hit.offset) for hit in keyword_hits(stream)] == [("flag", 20)]
    assert keyword_hits(b"no findings here \x00\xff") == ()
    assert keyword_hits(b"") == ()


def test_keyword_hits_are_ordered_by_offset_and_capped() -> None:
    stream = b"secret key " * (MAX_DETECTIONS * 4)
    assert len(keyword_hits(stream)) == MAX_DETECTIONS
    offsets = [hit.offset for hit in keyword_hits(stream)]
    assert offsets == sorted(offsets)


def test_a_stream_of_one_word_character_is_searched_instantly() -> None:
    """A constant bit plane extracts to one byte repeated — the quadratic trap.

    The old whole-stream regex backtracked through every split of the run, about
    nine seconds for forty thousand letters; ``find`` is linear, and this whole
    module runs in well under a second because of it.
    """
    assert detect_patterns(b"U" * 200_000) == ()
    assert detect_patterns(b"a" * 200_000) == ()


def test_a_keyword_past_a_long_word_run_is_still_found() -> None:
    detections = detect_patterns(b"U" * 500 + b"flag{in_the_noise}")
    assert [(finding.label, finding.offset) for finding in detections] == [("flag", 500)]
