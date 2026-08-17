"""Audio playback + Range support (IMPLEMENTATION.md P1-10).

Range is what makes the per-sentence timestamps usable: the player seeks to
`start_sec` instead of downloading the whole recording first.
"""
import pytest

from app.api.audio import parse_range


# --- header parsing (pure) ----------------------------------------------------------

@pytest.mark.parametrize(
    "header,size,expected",
    [
        ("bytes=0-99", 1000, (0, 99)),
        ("bytes=100-", 1000, (100, 999)),      # open-ended
        ("bytes=-100", 1000, (900, 999)),      # suffix
        ("bytes=0-99999", 1000, (0, 999)),     # clamped to the file
        ("bytes=0-0", 1000, (0, 0)),           # single byte
    ],
)
def test_parse_range_variants(header, size, expected):
    assert parse_range(header, size) == expected


@pytest.mark.parametrize("header", [None, "", "items=0-10", "bytes=abc"])
def test_unusable_headers_mean_serve_the_whole_file(header):
    assert parse_range(header, 1000) is None


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=500-499", "bytes=-0"])
def test_unsatisfiable_ranges_raise(header):
    with pytest.raises(ValueError):
        parse_range(header, 1000)


# --- endpoint behaviour -------------------------------------------------------------

def _upload(client, wav_bytes):
    return client.post(
        "/jobs", files={"file": ("note.wav", wav_bytes, "audio/wav")}
    ).json()["job_id"]


def test_full_download_advertises_range_support(client, wav_bytes):
    job_id = _upload(client, wav_bytes)
    resp = client.get(f"/jobs/{job_id}/audio")

    assert resp.status_code == 200
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.headers["content-type"].startswith("audio/")
    assert int(resp.headers["content-length"]) == len(wav_bytes)
    assert resp.content == wav_bytes


def test_range_request_returns_206_with_the_exact_slice(client, wav_bytes):
    job_id = _upload(client, wav_bytes)
    resp = client.get(f"/jobs/{job_id}/audio", headers={"Range": "bytes=100-199"})

    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes 100-199/{len(wav_bytes)}"
    assert resp.headers["content-length"] == "100"
    assert resp.content == wav_bytes[100:200]


def test_suffix_range_returns_the_tail(client, wav_bytes):
    job_id = _upload(client, wav_bytes)
    resp = client.get(f"/jobs/{job_id}/audio", headers={"Range": "bytes=-64"})
    assert resp.status_code == 206
    assert resp.content == wav_bytes[-64:]


def test_unsatisfiable_range_returns_416(client, wav_bytes):
    job_id = _upload(client, wav_bytes)
    resp = client.get(f"/jobs/{job_id}/audio", headers={"Range": "bytes=999999-"})
    assert resp.status_code == 416
    assert resp.headers["content-range"] == f"bytes */{len(wav_bytes)}"


def test_audio_for_unknown_job_is_404(client):
    assert client.get("/jobs/nope/audio").status_code == 404


def test_timestamps_in_the_report_address_real_audio(client, wav_bytes):
    """The traceability contract: an item's timestamps + a Range request = playback."""
    job_id = _upload(client, wav_bytes)
    item = client.get(f"/jobs/{job_id}/report").json()["soap"]["subjective"]["items"][0]
    assert item["start_sec"] is not None and item["end_sec"] > item["start_sec"]
    assert client.get(f"/jobs/{job_id}/audio", headers={"Range": "bytes=0-9"}).status_code == 206
