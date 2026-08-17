"""Audio archive contract (IMPLEMENTATION.md P1-05).

The headline test is `test_store_exposes_no_delete_operation`: the retention promise is
enforced by the *shape of the interface*, not by remembering not to call something.
"""
import inspect

import pytest

from app.storage import audio_store as audio_store_module
from app.storage.audio_store import AudioStore, LocalAudioStore, sha256_of


def _write(tmp_path, name: str, data: bytes):
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --- the retention guarantee ------------------------------------------------------

def test_store_exposes_no_delete_operation():
    """No delete/remove/prune/purge anywhere in the store's public surface."""
    forbidden = ("delete", "remove", "unlink", "purge", "prune", "evict", "rmtree")
    for cls in (LocalAudioStore, AudioStore):
        for name in dir(cls):
            if name.startswith("_"):
                continue
            assert not any(word in name.lower() for word in forbidden), (
                f"{cls.__name__}.{name} looks like a deletion API — audio is permanent"
            )


def test_module_never_calls_a_filesystem_delete():
    """Guards against a helper that deletes internally while the API stays clean."""
    source = inspect.getsource(audio_store_module)
    for call in ("os.remove(", "os.unlink(", ".unlink(", "shutil.rmtree(", "os.rmdir("):
        assert call not in source, f"audio_store must not call {call}"


# --- content addressing -----------------------------------------------------------

def test_put_archives_under_the_content_hash(tmp_path):
    store = LocalAudioStore(tmp_path / "archive")
    src = _write(tmp_path, "note.wav", b"RIFF-audio-bytes")

    stored = store.put(src, ".wav")

    assert stored.sha256 == sha256_of(src)
    assert stored.storage_key == f"{stored.sha256[:2]}/{stored.sha256}.wav"
    assert stored.was_new is True
    assert store.exists(stored.storage_key)
    assert store.path_for(stored.storage_key).read_bytes() == b"RIFF-audio-bytes"


def test_identical_content_is_stored_once(tmp_path):
    store = LocalAudioStore(tmp_path / "archive")
    first = store.put(_write(tmp_path, "a.wav", b"same-bytes"), ".wav")
    second = store.put(_write(tmp_path, "b.wav", b"same-bytes"), ".wav")

    assert second.was_new is False
    assert first.storage_key == second.storage_key
    assert store.object_count() == 1


def test_different_content_gets_different_objects(tmp_path):
    store = LocalAudioStore(tmp_path / "archive")
    store.put(_write(tmp_path, "a.wav", b"one"), ".wav")
    store.put(_write(tmp_path, "b.wav", b"two"), ".wav")
    assert store.object_count() == 2


def test_archived_object_is_read_only(tmp_path):
    import os
    import stat

    store = LocalAudioStore(tmp_path / "archive")
    stored = store.put(_write(tmp_path, "a.wav", b"protected"), ".wav")
    path = store.path_for(stored.storage_key)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0

    # root/CAP_DAC_OVERRIDE can bypass chmod(0444), so the portable contract is
    # the absence of write permission bits. Ordinary users must also be rejected.
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        with pytest.raises(PermissionError):
            open(path, "wb")


def test_no_partial_object_is_left_behind(tmp_path):
    store = LocalAudioStore(tmp_path / "archive")
    store.put(_write(tmp_path, "a.wav", b"complete"), ".wav")
    assert not list((tmp_path / "archive").rglob("*.part"))


# --- reading ----------------------------------------------------------------------

def test_read_range_returns_the_inclusive_slice(tmp_path):
    store = LocalAudioStore(tmp_path / "archive")
    payload = bytes(range(100))
    stored = store.put(_write(tmp_path, "a.wav", payload), ".wav")

    chunks = b"".join(store.read_range(stored.storage_key, 10, 19))
    assert chunks == payload[10:20]
    assert store.size_of(stored.storage_key) == 100


def test_path_traversal_is_rejected(tmp_path):
    store = LocalAudioStore(tmp_path / "archive")
    with pytest.raises(ValueError):
        store.path_for("../../etc/passwd")


# --- integrity --------------------------------------------------------------------

def test_verify_detects_corruption(tmp_path):
    store = LocalAudioStore(tmp_path / "archive")
    stored = store.put(_write(tmp_path, "a.wav", b"original"), ".wav")
    assert store.verify(stored.storage_key, stored.sha256) is True

    target = store.path_for(stored.storage_key)
    target.chmod(0o600)          # simulate bit-rot on an otherwise read-only object
    target.write_bytes(b"rotted")
    assert store.verify(stored.storage_key, stored.sha256) is False


def test_verify_reports_false_for_a_missing_object(tmp_path):
    store = LocalAudioStore(tmp_path / "archive")
    assert store.verify("ab/does-not-exist.wav", "0" * 64) is False
