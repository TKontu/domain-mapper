"""Embedder and reranker adapters over a local GPU service (AD-03).

ARCHITECTURE.md §6 puts embeddings and reranking on a local GPU service: the
Index stage embeds chunks into pgvector, and retrieval runs hybrid search →
top-k → reranker → top-n. Both services are assumed to speak JSON over HTTP in
the OpenAI-compatible shapes documented in ``docs/items/AD-03.md``:

* :class:`HttpEmbedder` posts to ``{base}/embeddings`` and reads
  ``{"data": [{"index": n, "embedding": [...]}, ...]}``.
* :class:`HttpReranker` posts to ``{base}/rerank`` and reads
  ``{"results": [{"index": n, "relevance_score": x}, ...]}``.

:class:`Embedder` and :class:`Reranker` are the protocols the pipeline talks to;
the HTTP classes are one implementation. HTTP goes through an injectable
transport callable whose default uses only :mod:`urllib.request`, so this adds
no dependency.

Failures surface as :class:`EmbeddingError` (status code plus a canned reason)
or its subclass :class:`EmbeddingShapeError` (the service returned vectors that
cannot be aligned to the inputs). Neither message contains request text or a
response body: input passages may hold private evidence.

:meth:`HttpEmbedder.embed` splits its input into batches of ``batch_size`` so
one request is in flight at a time and a caller can bound peak memory; the
caller still holds every output vector, so streaming callers must chunk their
own input. :meth:`HttpReranker.rerank` sends one request per call.

The module is **unwired by design**: the first production callers are the Index
stage row (embedder) and the Assess retrieval row (reranker), which will build
the adapters with :func:`build_embedder` and :func:`build_reranker`.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.client import responses as _HTTP_REASONS
from typing import Any, Final, Protocol, cast

from core.config import Settings

DEFAULT_TIMEOUT_SECONDS: Final = 60.0
DEFAULT_BATCH_SIZE: Final = 32


@dataclass(frozen=True)
class HttpResponse:
    """A minimal HTTP response: status and raw body, nothing else."""

    status: int
    body: bytes = b""


class Transport(Protocol):
    """Injected HTTP transport; the default is :func:`urllib_transport`."""

    def __call__(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> HttpResponse: ...


class EmbeddingError(RuntimeError):
    """A failed embedding or reranking call; never contains request text."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class EmbeddingShapeError(EmbeddingError):
    """The service returned vectors or hits that cannot be aligned to the input."""


@dataclass(frozen=True)
class RerankHit:
    """One reranked document: its position in the request and its score."""

    index: int
    score: float


class Embedder(Protocol):
    """The interface the Index stage talks to."""

    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    """The interface the Assess retrieval step talks to."""

    model: str

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankHit]: ...


def urllib_transport(
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> HttpResponse:
    """Default transport: one stdlib POST, no redirects beyond urllib's own."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(status=response.status, body=response.read())
    except urllib.error.HTTPError as exc:
        # The error body is deliberately not read: it can echo input text.
        return HttpResponse(status=exc.code)
    except OSError as exc:
        # Only the exception type: provider/transport text is not trusted here.
        raise EmbeddingError(f"transport failed: {type(exc).__name__}") from None


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_texts(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings, not a string")
    items = list(values)
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"{name} must contain only strings")
    return items


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a finite number of seconds > 0")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite number of seconds > 0")
    return float(timeout)


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


def _raise_for_status(response: HttpResponse) -> None:
    if 200 <= response.status < 300:
        return
    # A canned reason from the status line; never the response body.
    raise EmbeddingError(
        f"HTTP {response.status} {_HTTP_REASONS.get(response.status, 'error')}",
        status=response.status,
    )


def _decode_body(response: HttpResponse) -> dict[str, Any]:
    try:
        payload = json.loads(response.body)
    except ValueError:
        raise EmbeddingError("response body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise EmbeddingError("response body is not a JSON object")
    return cast("dict[str, Any]", payload)


def _float_vector(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise EmbeddingShapeError("response embedding is not a non-empty list")
    vector: list[float] = []
    for element in value:
        if (
            isinstance(element, bool)
            or not isinstance(element, (int, float))
            or not math.isfinite(element)
        ):
            raise EmbeddingShapeError("response embedding has a malformed value")
        vector.append(float(element))
    return vector


def _vectors_from(payload: Mapping[str, Any], *, expected: int) -> list[list[float]]:
    """Align a response's ``data`` items to input positions by their ``index``."""
    data = payload.get("data")
    if not isinstance(data, list):
        raise EmbeddingShapeError("response has no data list")
    if len(data) != expected:
        raise EmbeddingShapeError(f"response returned {len(data)} embeddings for {expected} inputs")
    ordered: list[list[float] | None] = [None] * expected
    width: int | None = None
    for item in data:
        if not isinstance(item, Mapping):
            raise EmbeddingShapeError("response data item is malformed")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise EmbeddingShapeError("response data item has no integer index")
        if not 0 <= index < expected:
            raise EmbeddingShapeError("response data item index is out of range")
        if ordered[index] is not None:
            raise EmbeddingShapeError("response data repeats an index")
        vector = _float_vector(item.get("embedding"))
        if width is None:
            width = len(vector)
        elif len(vector) != width:
            raise EmbeddingShapeError("response embeddings have inconsistent dimensions")
        ordered[index] = vector
    vectors: list[list[float]] = []
    for slot in ordered:
        if slot is None:
            raise EmbeddingShapeError("response is missing an embedding")
        vectors.append(slot)
    return vectors


def _hits_from(payload: Mapping[str, Any], *, document_count: int) -> list[RerankHit]:
    """Parse ``results`` into hits that point at real, distinct input documents."""
    results = payload.get("results")
    if not isinstance(results, list):
        raise EmbeddingShapeError("response has no results list")
    seen: set[int] = set()
    hits: list[RerankHit] = []
    for item in results:
        if not isinstance(item, Mapping):
            raise EmbeddingShapeError("response result is malformed")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise EmbeddingShapeError("response result has no integer index")
        if not 0 <= index < document_count:
            raise EmbeddingShapeError("response result index is out of range")
        if index in seen:
            raise EmbeddingShapeError("response results repeat an index")
        score = item.get("relevance_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise EmbeddingShapeError("response result has a malformed score")
        seen.add(index)
        hits.append(RerankHit(index=index, score=float(score)))
    return hits


class _BaseHttpAdapter:
    """Shared base URL, headers, timeout and POST plumbing."""

    model: str

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        transport: Transport | None,
        timeout: float,
    ) -> None:
        self.model = _require_text(model, "model")
        self._base_url = _require_text(base_url, "base_url").rstrip("/")
        self.transport: Transport = urllib_transport if transport is None else transport
        self._timeout = _validate_timeout(timeout)

    def _post(self, path: str, payload: Mapping[str, object]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        try:
            response = self.transport(
                f"{self._base_url}{path}", headers=headers, body=body, timeout=self._timeout
            )
        except EmbeddingError:
            raise
        except OSError as exc:
            # A custom transport may raise instead of returning a status.
            raise EmbeddingError(f"transport failed: {type(exc).__name__}") from None
        _raise_for_status(response)
        return _decode_body(response)


class HttpEmbedder(_BaseHttpAdapter):
    """OpenAI-compatible ``{base}/embeddings`` client that batches its input."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__(base_url, model, transport=transport, timeout=timeout)
        self._batch_size = _require_positive_int(batch_size, "batch_size")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = _require_texts(texts, "texts")
        vectors: list[list[float]] = []
        for start in range(0, len(items), self._batch_size):
            batch = items[start : start + self._batch_size]
            payload = self._post(
                "/embeddings",
                {"model": self.model, "input": batch},
            )
            vectors.extend(_vectors_from(payload, expected=len(batch)))
        return vectors


class HttpReranker(_BaseHttpAdapter):
    """``{base}/rerank`` client returning the top-n documents as hits."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(base_url, model, transport=transport, timeout=timeout)

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankHit]:
        query_text = _require_text(query, "query")
        items = _require_texts(documents, "documents")
        limit = _require_positive_int(top_n, "top_n")
        if not items:
            return []
        payload = self._post(
            "/rerank",
            {
                "model": self.model,
                "query": query_text,
                "documents": items,
                "top_n": limit,
            },
        )
        hits = _hits_from(payload, document_count=len(items))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


def build_embedder(
    settings: Settings,
    model: str,
    *,
    transport: Transport | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Embedder:
    """Build an :class:`Embedder` for ``settings.embedding_url`` and ``model``."""
    return HttpEmbedder(
        settings.embedding_url,
        model,
        transport=transport,
        timeout=timeout,
        batch_size=batch_size,
    )


def build_reranker(
    settings: Settings,
    model: str,
    *,
    transport: Transport | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Reranker:
    """Build a :class:`Reranker` for ``settings.reranker_url`` and ``model``.

    ``Settings`` carries the embedding and reranking services as separate
    endpoints (``DEFAULT_EMBEDDING_URL`` / ``DEFAULT_RERANKER_URL``), so the
    reranker reads ``reranker_url`` rather than posting ``/rerank`` to the
    embedder's URL.
    """
    return HttpReranker(settings.reranker_url, model, transport=transport, timeout=timeout)
