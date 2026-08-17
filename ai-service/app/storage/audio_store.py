"""Permanent, content-addressed audio storage (IMPLEMENTATION.md P1-05).

**There is no delete method — anywhere in this module, by design.** Recordings are
clinical source evidence: every report sentence carries a timestamp that only means
something while the audio still exists, and the archive doubles as the unlabelled
corpus that the self-supervised layer trains on (PLAN_V2 §5.1). "Free up space" is
therefore not a capability this interface offers; growth is handled by warning and by
cold-tiering, never by removal (PLAN_V2 §7).

Content addressing (`sha256`) gives de-duplication for free: the same recording
uploaded twice is stored once and referenced by two jobs.

Files are made read-only after writing, so a later bug cannot truncate an archived
recording in place.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Protocol

logger = logging.getLogger(__name__)

CHUNK = 1024 * 1024


@dataclass(frozen=True)
class StoredAudio:
    sha256: str
    storage_key: str          # store-relative, e.g. "ab/abcd….m4a"
    size_bytes: int
    extension: str
    was_new: bool             # False = identical content already archived


class AudioStore(Protocol):
    """Deliberately minimal: put, open, iterate, verify. No delete, no overwrite."""

    def put(self, source: Path, extension: str) -> StoredAudio: ...
    def path_for(self, storage_key: str) -> Path: ...
    def open(self, storage_key: str) -> BinaryIO: ...
    def verify(self, storage_key: str, sha256: str) -> bool: ...


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalAudioStore:
    """Filesystem implementation: <root>/<first two hex chars>/<sha256><ext>.

    The two-character fan-out keeps directory listings small once the archive holds
    tens of thousands of recordings.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- writing ---------------------------------------------------------------
    def put(self, source: Path, extension: str) -> StoredAudio:
        """Archive `source` under its content hash. Never overwrites an existing object."""
        source = Path(source)
        digest = sha256_of(source)
        extension = extension if extension.startswith(".") or not extension else f".{extension}"
        key = f"{digest[:2]}/{digest}{extension.lower()}"
        dest = self.root / key

        if dest.exists():
            logger.info("audio already archived (sha256=%s) — reusing", digest[:12])
            return StoredAudio(digest, key, dest.stat().st_size, extension, was_new=False)

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)          # atomic: no half-written object is ever visible
        self._make_read_only(dest)

        size = dest.stat().st_size
        logger.info("archived audio sha256=%s (%d bytes) -> %s", digest[:12], size, key)
        return StoredAudio(digest, key, size, extension, was_new=True)

    @staticmethod
    def _make_read_only(path: Path) -> None:
        try:
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        except OSError:  # pragma: no cover - platform/filesystem dependent
            logger.debug("could not set read-only on %s", path)

    # --- reading ---------------------------------------------------------------
    def path_for(self, storage_key: str) -> Path:
        path = (self.root / storage_key).resolve()
        root = self.root.resolve()
        # Defence in depth: a crafted key must not escape the store.
        if root != path and root not in path.parents:
            raise ValueError(f"storage key escapes the audio store: {storage_key!r}")
        return path

    def exists(self, storage_key: str) -> bool:
        return self.path_for(storage_key).is_file()

    def open(self, storage_key: str) -> BinaryIO:
        return open(self.path_for(storage_key), "rb")

    def read_range(self, storage_key: str, start: int, end: int) -> Iterator[bytes]:
        """Yield bytes [start, end] inclusive — the HTTP Range semantics (P1-10)."""
        remaining = end - start + 1
        with self.open(storage_key) as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    def size_of(self, storage_key: str) -> int:
        return self.path_for(storage_key).stat().st_size

    # --- integrity -------------------------------------------------------------
    def verify(self, storage_key: str, sha256: str) -> bool:
        """Re-hash an archived object. Silent bit-rot is the one way a 'never delete'
        policy can still lose data, so this is checked periodically (P1-12)."""
        path = self.path_for(storage_key)
        if not path.is_file():
            return False
        return sha256_of(path) == sha256

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def object_count(self) -> int:
        return sum(1 for p in self.root.rglob("*") if p.is_file())
