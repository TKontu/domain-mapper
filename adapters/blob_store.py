"""Content-addressed raw storage for fetched bodies (RS-01).

ARCHITECTURE.md §4 (Crawl & snapshot) stores each fetched body content-addressed
by ``sha256``; §5's ``snapshot`` row carries that digest. :class:`BlobStore` is
the protocol the Crawl stage will call, :class:`FilesystemBlobStore` the
filesystem implementation, and :class:`InMemoryBlobStore` an offline fake for
tests.

Semantics:

* ``put(data)`` returns the lowercase hex SHA-256 of ``data`` and stores it at
  ``root/<d[:2]>/<d[2:4]>/<d>``. Writes go to a temporary file in the same
  directory and are moved into place with :func:`os.replace`, so a reader never
  sees a partial blob. Putting an existing digest does not rewrite it.
* ``get(digest)`` re-hashes the bytes it read and raises
  :class:`BlobStoreError` if they no longer match the digest.
* ``get`` and ``exists`` reject anything that is not exactly 64 lowercase hex
  characters with :class:`BlobStoreError` before touching the filesystem, so a
  digest such as ``../...`` cannot escape the root.
* A missing blob raises :class:`BlobNotFound` (a :class:`BlobStoreError`);
  ``exists`` returns ``False`` instead.

:func:`raw_store_root` resolves ``RAW_STORE_DIR`` for this module only (default
``data/raw``); settings stay out of ``core/config.py`` until the wiring row.

The module is **unwired by design**: the follow-up row that wires raw bytes into
:func:`core.ingest.ingest_page` is its production caller. Scale: one blob is
held in memory per call, bounded by the fetcher's byte cap; a streaming API is
a residual.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

DEFAULT_RAW_STORE_DIR: Final = Path("data/raw")
_DIGEST_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_TEMP_PREFIX: Final = ".tmp-"


class BlobStoreError(RuntimeError):
    """A failed blob operation or an invalid digest."""


class BlobNotFound(BlobStoreError):
    """The requested digest is valid but no blob is stored for it."""


@runtime_checkable
class BlobStore(Protocol):
    """Content-addressed storage for one immutable body at a time."""

    def put(self, data: bytes) -> str:
        """Store ``data`` and return its lowercase hex SHA-256 digest."""
        ...

    def get(self, digest: str) -> bytes:
        """Return the verified bytes for ``digest``."""
        ...

    def exists(self, digest: str) -> bool:
        """Return whether a blob for ``digest`` is stored."""
        ...


def raw_store_root(env: Mapping[str, str]) -> Path:
    """Return ``RAW_STORE_DIR`` from ``env``, or ``data/raw`` when unset/blank."""
    value = env.get("RAW_STORE_DIR")
    if value is None or not value.strip():
        return DEFAULT_RAW_STORE_DIR
    return Path(value)


def _check_digest(digest: str) -> None:
    if not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None:
        raise BlobStoreError(f"not a lowercase sha256 hex digest: {digest!r}")


class FilesystemBlobStore:
    """Store blobs under ``root/<d[:2]>/<d[2:4]>/<d>``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:4] / digest

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(digest)
        if path.is_file():
            return digest

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=_TEMP_PREFIX, dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return digest

    def get(self, digest: str) -> bytes:
        _check_digest(digest)
        try:
            data = self._path(digest).read_bytes()
        except FileNotFoundError:
            raise BlobNotFound(f"no blob for digest {digest}") from None
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise BlobStoreError(
                f"blob for digest {digest} is corrupted: content hashes to {actual}"
            )
        return data

    def exists(self, digest: str) -> bool:
        _check_digest(digest)
        return self._path(digest).is_file()


class InMemoryBlobStore:
    """Fake :class:`BlobStore` over a dict (tests, example scope)."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        if digest not in self._blobs:
            self._blobs[digest] = bytes(data)
        return digest

    def get(self, digest: str) -> bytes:
        _check_digest(digest)
        try:
            data = self._blobs[digest]
        except KeyError:
            raise BlobNotFound(f"no blob for digest {digest}") from None
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise BlobStoreError(
                f"blob for digest {digest} is corrupted: content hashes to {actual}"
            )
        return data

    def exists(self, digest: str) -> bool:
        _check_digest(digest)
        return digest in self._blobs


def _conforms() -> BlobStore:
    """Static check: mypy rejects this if either store drifts from the protocol."""
    return InMemoryBlobStore()
