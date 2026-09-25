from pixelsb.domain.detect import Detection, detect_patterns


def test_magic_headers_are_found_at_their_offsets() -> None:
    stream = b"\x89PNG\r\n\x1a\n" + b"\x00" * 5 + b"PK\x03\x04" + b"tail"
    assert detect_patterns(stream) == (
        Detection(0, "PNG", 8),
        Detection(13, "ZIP", 4),
    )


def test_flag_shaped_runs_are_flagged_including_competition_prefixes() -> None:
    stream = b"prefix flag{ab_1} mid DASCTF{d0ne}\x00"
    detections = detect_patterns(stream)
    assert [finding.label for finding in detections] == ["flag{ab_1}", "DASCTF{d0ne}"]
    assert all(finding.flagged for finding in detections)
    assert [finding.offset for finding in detections] == [7, 22]


def test_every_repeated_hit_is_reported_in_stream_order() -> None:
    stream = b"%PDF-" + b"\x00" * 4 + b"%PDF-"
    assert [finding.offset for finding in detect_patterns(stream)] == [0, 9]


def test_empty_and_clean_streams_find_nothing() -> None:
    assert detect_patterns(b"") == ()
    assert detect_patterns(b"nothing interesting \x00\xff here") == ()


def test_a_pathological_stream_is_capped() -> None:
    from pixelsb.domain.detect import MAX_DETECTIONS

    stream = b"GIF8" * (MAX_DETECTIONS * 4)
    assert len(detect_patterns(stream)) == MAX_DETECTIONS
