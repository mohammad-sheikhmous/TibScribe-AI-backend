"""Per-sentence audio clips (IMPLEMENTATION.md P2-05).

ffmpeg is an OS dependency, not a Python one, so the endpoint is designed to degrade
loudly: it says what is missing and hands back a working alternative rather than
crashing with a FileNotFoundError from inside subprocess. Both paths are tested here —
the command builder is a pure function, and the missing-binary branch is forced with a
monkeypatch so the suite behaves the same on machines with and without ffmpeg.
"""
import shutil

import pytest

from app.api.audio import CLIP_PAD_SEC, build_ffmpeg_cmd


def _upload(client, wav_bytes):
    return client.post(
        "/jobs", files={"file": ("note.wav", wav_bytes, "audio/wav")}
    ).json()["job_id"]


def _first_item(client, job_id):
    report = client.get(f"/jobs/{job_id}/report").json()
    return next(item for section in report["soap"].values() for item in section["items"])


# --- command construction (pure) ----------------------------------------------------

def test_ffmpeg_command_seeks_before_input_and_copies_the_stream():
    cmd = build_ffmpeg_cmd("in.mp3", 12.5, 3.25)
    assert cmd[cmd.index("-ss") + 1] == "12.500"
    assert cmd[cmd.index("-t") + 1] == "3.250"
    # -ss BEFORE -i is the fast input seek; -c copy avoids a lossy re-encode.
    assert cmd.index("-ss") < cmd.index("-i")

def test_ffmpeg_command_outputs_browser_playable_mp3():
    cmd = build_ffmpeg_cmd("in.mp3", 12.5, 3.25)

    assert cmd[cmd.index("-ss") + 1] == "12.500"
    assert cmd[cmd.index("-t") + 1] == "3.250"

    assert cmd.index("-ss") < cmd.index("-i")

    assert cmd[cmd.index("-map") + 1] == "0:a:0"
    assert "-vn" in cmd

    assert cmd[cmd.index("-c:a") + 1] == "libmp3lame"
    assert cmd[cmd.index("-f") + 1] == "mp3"

def test_command_never_produces_a_negative_or_zero_window():
    cmd = build_ffmpeg_cmd("in.mp3", -5.0, 0.0)
    assert float(cmd[cmd.index("-ss") + 1]) == 0.0
    assert float(cmd[cmd.index("-t") + 1]) > 0


# --- endpoint -----------------------------------------------------------------------

def test_missing_ffmpeg_returns_503_with_a_usable_alternative(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    job_id = _upload(client, wav_bytes)
    item = _first_item(client, job_id)

    resp = client.get(f"/jobs/{job_id}/items/{item['item_id']}/audio")

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "ffmpeg" in detail["error"]
    # the zero-dependency fallback: a media fragment on the full-file endpoint
    assert detail["alternative"].startswith(f"/jobs/{job_id}/audio#t=")
    assert resp.headers["x-clip-start-sec"] is not None


def test_clip_window_is_padded_around_the_sentence(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    job_id = _upload(client, wav_bytes)
    item = _first_item(client, job_id)

    resp = client.get(f"/jobs/{job_id}/items/{item['item_id']}/audio")
    start = float(resp.headers["x-clip-start-sec"])
    duration = float(resp.headers["x-clip-duration-sec"])

    expected_start = max(0.0, item["start_sec"] - CLIP_PAD_SEC)
    assert start == pytest.approx(expected_start, abs=0.01)
    assert duration == pytest.approx(
        (item["end_sec"] - item["start_sec"]) + 2 * CLIP_PAD_SEC, abs=0.01
    )


def test_unknown_item_is_404(client, wav_bytes):
    job_id = _upload(client, wav_bytes)
    assert client.get(f"/jobs/{job_id}/items/nope/audio").status_code == 404


def test_item_from_another_job_is_not_reachable(client, wav_bytes):
    job_a = _upload(client, wav_bytes)
    job_b = _upload(client, wav_bytes + b"different")
    item_a = _first_item(client, job_a)
    # the item exists, but not under job B
    assert client.get(f"/jobs/{job_b}/items/{item_a['item_id']}/audio").status_code == 404
