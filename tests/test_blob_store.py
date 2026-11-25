"""Tests for the content-addressed raw blob store (RS-01).

Both the filesystem and in-memory stores are exercised through the
:class:`adapters.blob_store.BlobStore` protocol so they cannot drift apart.
The filesystem tests use ``tmp_path`` and stay offline.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from adapters.blob_store import (
    BlobNotFound,
    BlobStore,
    BlobStoreError,
    FilesystemBlobStore,
    InMemoryBlobStore,
    raw_store_root,
)

DATA = b"<html><body>content-addressed raw bytes</body></html>"
DIGEST = hashlib.sha256(DATA).hexdigest()
OTHER = b"a second, different body"
BAD_DIGESTS = [
    "",
    "abc",
    DIGEST.upper(),
    DIGEST[:-1],
    DIGEST + "0",
    DIGEST[:2] + "/" + DIGEST[3:],
    "../" + DIGEST[3:],
    "g" * 64,
]


@pytest.fixture(params=["filesystem", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[BlobStore]:
    """One conforming store of each implementation."""
    if request.param == "filesystem":
        yield FilesystemBlobStore(tmp_path / "raw")
    else:
        yield InMemoryBlobStore()


def filesystem_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / digest[2:4] / digest


def test_both_implementations_satisfy_the_protocol(tmp_path: Path) -> None:
    assert isinstance(FilesystemBlobStore(tmp_path), BlobStore)
    assert isinstance(InMemoryBlobStore(), BlobStore)

    class NoBlobs:
        pass

    assert not isinstance(NoBlobs(), BlobStore)


def test_put_returns_the_lowercase_hex_digest(store: BlobStore) -> None:
    digest = store.put(DATA)

    assert digest == DIGEST
    assert digest == digest.lower()
    assert len(digest) == 64


def test_round_trip_returns_the_same_bytes(store: BlobStore) -> None:
    store.put(DATA)

    assert store.get(DIGEST) == DATA


def test_empty_body_round_trips(store: BlobStore) -> None:
    digest = store.put(b"")

    assert digest == hashlib.sha256(b"").hexdigest()
    assert store.get(digest) == b""


def test_put_is_idempotent(store: BlobStore) -> None:
    first = store.put(DATA)
    second = store.put(DATA)

    assert first == second
    assert store.get(first) == DATA


def test_exists_reflects_stored_blobs(store: BlobStore) -> None:
    missing = hashlib.sha256(b"not stored").hexdigest()

    assert store.exists(missing) is False
    store.put(DATA)
    assert store.exists(DIGEST) is True


def test_get_missing_digest_raises_blob_not_found(store: BlobStore) -> None:
    missing = hashlib.sha256(b"not stored").hexdigest()

    with pytest.raises(BlobNotFound):
        store.get(missing)


def test_blob_not_found_is_a_blob_store_error() -> None:
    assert issubclass(BlobNotFound, BlobStoreError)


@pytest.mark.parametrize("digest", BAD_DIGESTS)
def test_digest_that_is_not_64_lowercase_hex_is_rejected(store: BlobStore, digest: str) -> None:
    with pytest.raises(BlobStoreError):
        store.get(digest)
    with pytest.raises(BlobStoreError):
        store.exists(digest)


def test_filesystem_layout_is_two_two_full_digest(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    store = FilesystemBlobStore(root)

    digest = store.put(DATA)

    path = filesystem_path(root, digest)
    assert path.is_file()
    assert path.read_bytes() == DATA


def test_filesystem_put_does_not_rewrite_an_existing_blob(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    store = FilesystemBlobStore(root)
    digest = store.put(DATA)
    path = filesystem_path(root, digest)
    os.utime(path, (1_700_000_000, 1_700_000_000))
    before = path.stat().st_mtime_ns

    assert store.put(DATA) == digest
    assert path.stat().st_mtime_ns == before


def test_filesystem_put_leaves_no_temporary_files(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    store = FilesystemBlobStore(root)

    digest = store.put(DATA)
    other = store.put(OTHER)

    files = sorted(path for path in root.rglob("*") if path.is_file())
    assert files == sorted([filesystem_path(root, digest), filesystem_path(root, other)])


def test_corrupted_filesystem_blob_raises_blob_store_error(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    store = FilesystemBlobStore(root)
    digest = store.put(DATA)
    path = filesystem_path(root, digest)
    path.write_bytes(b"tampered bytes")

    with pytest.raises(BlobStoreError):
        store.get(digest)


def test_filesystem_get_missing_blob_is_not_corruption(tmp_path: Path) -> None:
    store = FilesystemBlobStore(tmp_path / "raw")
    missing = hashlib.sha256(b"not stored").hexdigest()

    with pytest.raises(BlobNotFound):
        store.get(missing)


def test_raw_store_root_defaults_to_data_raw() -> None:
    assert raw_store_root({}) == Path("data/raw")


def test_raw_store_root_uses_the_environment_override(tmp_path: Path) -> None:
    configured = tmp_path / "elsewhere" / "raw"

    assert raw_store_root({"RAW_STORE_DIR": str(configured)}) == configured


@pytest.mark.parametrize("value", ["", "   "])
def test_raw_store_root_treats_a_blank_override_as_default(value: str) -> None:
    assert raw_store_root({"RAW_STORE_DIR": value}) == Path("data/raw")
