from app.core.segmentation.segmenter import build_segments


def test_short_segment_stays_single():
    segs = build_segments(
        [{"start": 0.0, "end": 2.0, "text": "صداع شديد مع دوخة"}], max_chars=220
    )
    assert len(segs) == 1
    assert segs[0].timestamp_precision == "segment"
    assert segs[0].order_index == 0
    assert segs[0].source_segment_index == 0


def test_empty_segment_skipped():
    segs = build_segments([{"start": 0.0, "end": 1.0, "text": "   "}])
    assert segs == []


def test_long_segment_resplit_by_words_within_bounds():
    # One long Whisper segment (> max_chars) with word timestamps and a clear pause.
    words = []
    t = 0.0
    for i in range(20):
        w = {"word": f" كلمة{i}", "start": t, "end": t + 0.3}
        words.append(w)
        # inject a long pause in the middle to force a clause boundary
        t += 0.3 + (0.9 if i == 9 else 0.05)
    text = "".join(w["word"] for w in words).strip()
    seg = {"start": 0.0, "end": words[-1]["end"], "text": text, "words": words}

    segs = build_segments([seg], max_chars=40, pause_gap_sec=0.5)

    assert len(segs) >= 2, "long segment should be re-split"
    assert all(s.timestamp_precision == "word" for s in segs)
    # order indices are globally increasing and contiguous
    assert [s.order_index for s in segs] == list(range(len(segs)))
    # timestamps monotonic and nested inside the parent [0, end]
    prev_end = -1.0
    for s in segs:
        assert s.start_sec <= s.end_sec
        assert s.start_sec >= 0.0 - 1e-9
        assert s.end_sec <= seg["end"] + 1e-9
        assert s.start_sec >= prev_end - 1e-9
        prev_end = s.start_sec


def test_long_segment_without_words_uses_interpolation():
    text = "كلمة " * 100  # long, no word timestamps
    segs = build_segments(
        [{"start": 10.0, "end": 20.0, "text": text.strip()}], max_chars=50
    )
    assert len(segs) >= 2
    assert all(s.timestamp_precision == "interpolated" for s in segs)
    assert segs[0].start_sec >= 10.0 - 1e-9
    assert segs[-1].end_sec <= 20.0 + 1e-9
