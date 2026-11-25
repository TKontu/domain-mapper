"""Tests for the ``Embedder`` and ``Reranker`` adapters (AD-03).

Every test runs against a fake transport, so the suite is offline: no embedding
or reranking service is contacted.
"""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import dataclass
from typing import Any

from adapters.embedding import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TIMEOUT_SECONDS,
    Embedder,
    EmbeddingError,
    EmbeddingShapeError,
    HttpEmbedder,
    HttpReranker,
    HttpResponse,
    RerankHit,
    Reranker,
    build_embedder,
    build_reranker,
)
from core.config import Settings

SECRET_TEXT = "Private document text that must never reach an error"


@dataclass
class Call:
    url: str
    headers: dict[str, str]
    body: bytes
    timeout: float

    def payload(self) -> dict[str, Any]:
        return json.loads(self.body)


class FakeTransport:
    """Scripted transport; one queued response (or exception) per call."""

    def __init__(self, *responses: HttpResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[Call] = []

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        self.calls.append(Call(url=url, headers=dict(headers), body=body, timeout=timeout))
        if url.endswith("/embeddings"):
            return self._next()
        return self._next()

    def _next(self) -> HttpResponse:
        if not self.responses:
            raise AssertionError("unexpected request")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def json_response(payload: Any, *, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, body=json.dumps(payload).encode("utf-8"))


def embedding_response(items: list[tuple[int, list[float]]], *, status: int = 200) -> HttpResponse:
    """An OpenAI-compatible embeddings body: data items carry ``index``."""
    return json_response(
        {
            "model": "fake-embed",
            "data": [{"index": index, "embedding": vector} for index, vector in items],
        },
        status=status,
    )


def rerank_response(items: list[tuple[int, float]], *, status: int = 200) -> HttpResponse:
    """An assumed rerank body: results carry ``index`` and ``relevance_score``."""
    return json_response(
        {"results": [{"index": index, "relevance_score": score} for index, score in items]},
        status=status,
    )


def embedder(transport: FakeTransport, **kwargs: Any) -> HttpEmbedder:
    options: dict[str, Any] = {"transport": transport}
    options.update(kwargs)
    return HttpEmbedder("http://gpu.test:8001/v1", "fake-embed", **options)


def reranker(transport: FakeTransport, **kwargs: Any) -> HttpReranker:
    options: dict[str, Any] = {"transport": transport}
    options.update(kwargs)
    return HttpReranker("http://gpu.test:8002/v1", "fake-rerank", **options)


class EmbedProtocolTests(unittest.TestCase):
    def test_concrete_implementations_satisfy_protocols(self) -> None:
        transport = FakeTransport()
        embed_impl: Embedder = embedder(transport)
        rerank_impl: Reranker = reranker(transport)
        self.assertEqual(embed_impl.model, "fake-embed")
        self.assertEqual(rerank_impl.model, "fake-rerank")

    def test_rerank_hit_is_frozen(self) -> None:
        hit = RerankHit(index=2, score=0.5)
        with self.assertRaises(Exception):
            hit.index = 3  # type: ignore[misc]


class EmbedRequestTests(unittest.TestCase):
    def test_embed_sends_url_body_and_model(self) -> None:
        transport = FakeTransport(embedding_response([(0, [0.1, 0.2]), (1, [0.3, 0.4])]))
        vectors = embedder(transport).embed(["alpha", "beta"])

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        (call,) = transport.calls
        self.assertEqual(call.url, "http://gpu.test:8001/v1/embeddings")
        self.assertEqual(call.headers["Content-Type"], "application/json")
        self.assertEqual(call.payload(), {"model": "fake-embed", "input": ["alpha", "beta"]})
        self.assertEqual(call.timeout, DEFAULT_TIMEOUT_SECONDS)

    def test_embed_strips_trailing_slash_from_base_url(self) -> None:
        transport = FakeTransport(embedding_response([(0, [1.0])]))
        HttpEmbedder("http://gpu.test:8001/v1/", "fake-embed", transport=transport).embed(["a"])
        self.assertEqual(transport.calls[0].url, "http://gpu.test:8001/v1/embeddings")

    def test_embed_accepts_a_tuple_of_texts(self) -> None:
        transport = FakeTransport(embedding_response([(0, [1.0])]))
        self.assertEqual(embedder(transport).embed(("only",)), [[1.0]])
        self.assertEqual(transport.calls[0].payload()["input"], ["only"])

    def test_embed_empty_input_returns_empty_without_a_request(self) -> None:
        transport = FakeTransport()
        self.assertEqual(embedder(transport).embed([]), [])
        self.assertEqual(transport.calls, [])

    def test_embed_rejects_a_bare_string(self) -> None:
        with self.assertRaises(TypeError):
            embedder(FakeTransport()).embed("not-a-sequence-of-texts")  # type: ignore[arg-type]

    def test_embed_rejects_a_non_string_item(self) -> None:
        with self.assertRaises(TypeError):
            embedder(FakeTransport()).embed(["ok", 3])  # type: ignore[list-item]


class EmbedBatchingTests(unittest.TestCase):
    def test_embed_splits_into_ceil_batches(self) -> None:
        texts = [f"text-{n}" for n in range(7)]
        transport = FakeTransport(
            embedding_response([(0, [0.0]), (1, [0.1]), (2, [0.2])]),
            embedding_response([(0, [0.3]), (1, [0.4]), (2, [0.5])]),
            embedding_response([(0, [0.6])]),
        )
        vectors = embedder(transport, batch_size=3).embed(texts)

        self.assertEqual(len(transport.calls), 3)
        received = [call.payload()["input"] for call in transport.calls]
        self.assertEqual(
            received,
            [
                ["text-0", "text-1", "text-2"],
                ["text-3", "text-4", "text-5"],
                ["text-6"],
            ],
        )
        self.assertEqual(vectors, [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])

    def test_embed_default_batch_size_is_used(self) -> None:
        texts = [f"t{n}" for n in range(DEFAULT_BATCH_SIZE + 1)]
        first = [(n, [float(n)]) for n in range(DEFAULT_BATCH_SIZE)]
        transport = FakeTransport(embedding_response(first), embedding_response([(0, [9.0])]))
        vectors = embedder(transport).embed(texts)

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(len(transport.calls[0].payload()["input"]), DEFAULT_BATCH_SIZE)
        self.assertEqual(len(transport.calls[1].payload()["input"]), 1)
        self.assertEqual(len(vectors), DEFAULT_BATCH_SIZE + 1)
        self.assertEqual(vectors[-1], [9.0])

    def test_embed_never_exceeds_batch_size_per_request(self) -> None:
        transport = FakeTransport(
            embedding_response([(0, [0.0]), (1, [0.1])]),
            embedding_response([(0, [0.2]), (1, [0.3])]),
        )
        embedder(transport, batch_size=2).embed(["a", "b", "c", "d"])
        self.assertEqual(len(transport.calls), 2)
        for call in transport.calls:
            self.assertLessEqual(len(call.payload()["input"]), 2)


class EmbedShapeTests(unittest.TestCase):
    def test_embed_reorders_vectors_by_index(self) -> None:
        transport = FakeTransport(embedding_response([(1, [1.0, 2.0]), (0, [3.0, 4.0])]))
        self.assertEqual(embedder(transport).embed(["first", "second"]), [[3.0, 4.0], [1.0, 2.0]])

    def test_embed_reorders_within_every_batch(self) -> None:
        transport = FakeTransport(
            embedding_response([(1, [10.0]), (0, [20.0])]),
            embedding_response([(0, [30.0])]),
        )
        self.assertEqual(
            embedder(transport, batch_size=2).embed(["a", "b", "c"]),
            [[20.0], [10.0], [30.0]],
        )

    def test_embed_raises_on_count_mismatch(self) -> None:
        transport = FakeTransport(embedding_response([(0, [1.0])]))
        with self.assertRaises(EmbeddingShapeError):
            embedder(transport).embed(["a", "b"])

    def test_embed_raises_on_out_of_range_index(self) -> None:
        transport = FakeTransport(embedding_response([(0, [1.0]), (2, [1.0])]))
        with self.assertRaises(EmbeddingShapeError):
            embedder(transport).embed(["a", "b"])

    def test_embed_raises_on_duplicate_index(self) -> None:
        transport = FakeTransport(embedding_response([(0, [1.0]), (0, [2.0])]))
        with self.assertRaises(EmbeddingShapeError):
            embedder(transport).embed(["a", "b"])

    def test_embed_raises_on_missing_integer_index(self) -> None:
        body = {"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]}
        transport = FakeTransport(json_response(body))
        with self.assertRaises(EmbeddingShapeError):
            embedder(transport).embed(["a", "b"])

    def test_embed_raises_on_true_index(self) -> None:
        transport = FakeTransport(embedding_response([(True, [1.0]), (0, [2.0])]))
        with self.assertRaises(EmbeddingShapeError):
            embedder(transport).embed(["a", "b"])  # type: ignore[list-item]

    def test_embed_raises_on_malformed_vector(self) -> None:
        cases: list[Any] = [
            {"index": 0, "embedding": []},
            {"index": 0, "embedding": "not-a-vector"},
            {"index": 0, "embedding": [1.0, "x"]},
            {"index": 0, "embedding": [True]},
            {"index": 0},
            "not-an-object",
        ]
        for item in cases:
            with self.subTest(item=item):
                transport = FakeTransport(json_response({"data": [item]}))
                with self.assertRaises(EmbeddingShapeError):
                    embedder(transport).embed(["a"])

    def test_embed_raises_on_non_finite_vector_value(self) -> None:
        for bad in (math.inf, math.nan):
            with self.subTest(bad=bad):
                transport = FakeTransport(
                    json_response({"data": [{"index": 0, "embedding": [bad]}]})
                )
                with self.assertRaises(EmbeddingShapeError):
                    embedder(transport).embed(["a"])

    def test_embed_raises_on_inconsistent_dimensions(self) -> None:
        transport = FakeTransport(embedding_response([(0, [1.0, 2.0]), (1, [1.0])]))
        with self.assertRaises(EmbeddingShapeError):
            embedder(transport).embed(["a", "b"])

    def test_embed_raises_when_data_is_missing(self) -> None:
        transport = FakeTransport(json_response({"model": "fake-embed"}))
        with self.assertRaises(EmbeddingShapeError):
            embedder(transport).embed(["a"])

    def test_embed_raises_on_non_object_body(self) -> None:
        transport = FakeTransport(json_response([1, 2, 3]))
        with self.assertRaises(EmbeddingError):
            embedder(transport).embed(["a"])

    def test_embed_raises_on_non_json_body(self) -> None:
        transport = FakeTransport(HttpResponse(status=200, body=b"<html>not json</html>"))
        with self.assertRaises(EmbeddingError):
            embedder(transport).embed(["a"])


class EmbedErrorTests(unittest.TestCase):
    def test_http_error_is_typed_and_does_not_echo_input(self) -> None:
        transport = FakeTransport(HttpResponse(status=500))
        with self.assertRaises(EmbeddingError) as caught:
            embedder(transport).embed([SECRET_TEXT])
        self.assertEqual(caught.exception.status, 500)
        self.assertNotIn(SECRET_TEXT, str(caught.exception))

    def test_http_error_body_is_not_echoed(self) -> None:
        body = json.dumps({"error": SECRET_TEXT}).encode("utf-8")
        transport = FakeTransport(HttpResponse(status=422, body=body))
        with self.assertRaises(EmbeddingError) as caught:
            embedder(transport).embed([SECRET_TEXT])
        self.assertEqual(caught.exception.status, 422)
        self.assertNotIn(SECRET_TEXT, str(caught.exception))

    def test_transport_failure_is_typed(self) -> None:
        transport = FakeTransport(ConnectionResetError("boom"))
        with self.assertRaises(EmbeddingError):
            embedder(transport).embed([SECRET_TEXT])

    def test_constructor_rejects_bad_batch_size(self) -> None:
        for bad in (0, -1, True, 2.5, "3", None):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    HttpEmbedder(
                        "http://gpu.test:8001/v1",
                        "fake-embed",
                        batch_size=bad,  # type: ignore[arg-type]
                    )

    def test_constructor_rejects_bad_timeout(self) -> None:
        for bad in (0, -1, math.inf, math.nan, True, "60"):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    HttpEmbedder(
                        "http://gpu.test:8001/v1",
                        "fake-embed",
                        timeout=bad,  # type: ignore[arg-type]
                    )

    def test_constructor_rejects_blank_base_url_or_model(self) -> None:
        for kwargs in ({"base_url": " "}, {"model": ""}):
            with self.subTest(kwargs=kwargs):
                options: dict[str, Any] = {
                    "base_url": "http://gpu.test:8001/v1",
                    "model": "fake-embed",
                }
                options.update(kwargs)
                with self.assertRaises(ValueError):
                    HttpEmbedder(**options)


class RerankTests(unittest.TestCase):
    def test_rerank_sends_url_body_and_model(self) -> None:
        transport = FakeTransport(rerank_response([(0, 0.9), (1, 0.1)]))
        hits = reranker(transport).rerank("query text", ["doc a", "doc b"], 2)

        self.assertEqual(hits, [RerankHit(index=0, score=0.9), RerankHit(index=1, score=0.1)])
        (call,) = transport.calls
        self.assertEqual(call.url, "http://gpu.test:8002/v1/rerank")
        self.assertEqual(call.headers["Content-Type"], "application/json")
        self.assertEqual(
            call.payload(),
            {
                "model": "fake-rerank",
                "query": "query text",
                "documents": ["doc a", "doc b"],
                "top_n": 2,
            },
        )
        self.assertEqual(call.timeout, DEFAULT_TIMEOUT_SECONDS)

    def test_rerank_returns_top_n_sorted_by_score_desc(self) -> None:
        transport = FakeTransport(rerank_response([(2, 0.2), (0, 0.95), (1, 0.5), (3, 0.7)]))
        hits = reranker(transport).rerank("q", ["d0", "d1", "d2", "d3"], 3)

        self.assertEqual(
            hits,
            [
                RerankHit(index=0, score=0.95),
                RerankHit(index=3, score=0.7),
                RerankHit(index=1, score=0.5),
            ],
        )

    def test_rerank_top_n_above_document_count_returns_everything(self) -> None:
        transport = FakeTransport(rerank_response([(1, 0.1), (0, 0.9)]))
        hits = reranker(transport).rerank("q", ["d0", "d1"], 10)
        self.assertEqual([hit.index for hit in hits], [0, 1])

    def test_rerank_empty_documents_returns_empty_without_a_request(self) -> None:
        transport = FakeTransport()
        self.assertEqual(reranker(transport).rerank("q", [], 3), [])
        self.assertEqual(transport.calls, [])

    def test_rerank_allows_empty_results(self) -> None:
        transport = FakeTransport(rerank_response([]))
        self.assertEqual(reranker(transport).rerank("q", ["d0"], 1), [])

    def test_rerank_strips_trailing_slash(self) -> None:
        transport = FakeTransport(rerank_response([(0, 1.0)]))
        HttpReranker("http://gpu.test:8002/v1/", "fake-rerank", transport=transport).rerank(
            "q", ["d0"], 1
        )
        self.assertEqual(transport.calls[0].url, "http://gpu.test:8002/v1/rerank")

    def test_rerank_rejects_bad_top_n(self) -> None:
        for bad in (0, -1, True, 2.5, "3", None):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    reranker(FakeTransport()).rerank(
                        "q",
                        ["d0"],
                        bad,  # type: ignore[arg-type]
                    )

    def test_rerank_rejects_bad_query_or_documents(self) -> None:
        adapter = reranker(FakeTransport())
        with self.assertRaises((TypeError, ValueError)):
            adapter.rerank(3, ["d0"], 1)  # type: ignore[arg-type]
        with self.assertRaises((TypeError, ValueError)):
            adapter.rerank("q", "a bare string", 1)  # type: ignore[arg-type]
        with self.assertRaises((TypeError, ValueError)):
            adapter.rerank("q", ["ok", 3], 1)  # type: ignore[list-item]

    def test_rerank_rejects_out_of_range_index(self) -> None:
        transport = FakeTransport(rerank_response([(1, 1.0)]))
        with self.assertRaises(EmbeddingShapeError):
            reranker(transport).rerank("q", ["only"], 1)

    def test_rerank_rejects_duplicate_index(self) -> None:
        transport = FakeTransport(rerank_response([(0, 1.0), (0, 0.5)]))
        with self.assertRaises(EmbeddingShapeError):
            reranker(transport).rerank("q", ["a", "b"], 2)

    def test_rerank_rejects_malformed_results(self) -> None:
        cases: list[Any] = [
            {"index": 0},
            {"index": 0, "relevance_score": "high"},
            {"index": 0, "relevance_score": True},
            {"index": "0", "relevance_score": 1.0},
            "not-an-object",
        ]
        for item in cases:
            with self.subTest(item=item):
                transport = FakeTransport(json_response({"results": [item]}))
                with self.assertRaises(EmbeddingShapeError):
                    reranker(transport).rerank("q", ["d0"], 1)

    def test_rerank_rejects_non_finite_score(self) -> None:
        for bad in (math.inf, math.nan):
            with self.subTest(bad=bad):
                transport = FakeTransport(
                    json_response({"results": [{"index": 0, "relevance_score": bad}]})
                )
                with self.assertRaises(EmbeddingShapeError):
                    reranker(transport).rerank("q", ["d0"], 1)

    def test_rerank_raises_when_results_are_missing(self) -> None:
        transport = FakeTransport(json_response({"model": "fake-rerank"}))
        with self.assertRaises(EmbeddingShapeError):
            reranker(transport).rerank("q", ["d0"], 1)

    def test_rerank_http_error_is_typed_and_does_not_echo_input(self) -> None:
        transport = FakeTransport(HttpResponse(status=503))
        with self.assertRaises(EmbeddingError) as caught:
            reranker(transport).rerank(SECRET_TEXT, [SECRET_TEXT], 1)
        self.assertEqual(caught.exception.status, 503)
        self.assertNotIn(SECRET_TEXT, str(caught.exception))

    def test_reranker_constructor_rejects_bad_timeout(self) -> None:
        for bad in (0, -1, math.inf, math.nan):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    HttpReranker(
                        "http://gpu.test:8002/v1",
                        "fake-rerank",
                        timeout=bad,  # type: ignore[arg-type]
                    )


class BuilderTests(unittest.TestCase):
    def test_build_embedder_uses_settings_and_model(self) -> None:
        transport = FakeTransport(embedding_response([(0, [1.0])]))
        settings = Settings(embedding_url="http://embed.test:9001/v1")
        built = build_embedder(settings, "embed-model", transport=transport, batch_size=4)

        self.assertEqual(built.model, "embed-model")
        built.embed(["a"])
        call = transport.calls[0]
        self.assertEqual(call.url, "http://embed.test:9001/v1/embeddings")
        self.assertEqual(call.payload()["model"], "embed-model")

    def test_build_embedder_passes_timeout(self) -> None:
        transport = FakeTransport(embedding_response([(0, [1.0])]))
        built = build_embedder(Settings(), "m", transport=transport, timeout=2.5)
        built.embed(["a"])
        self.assertEqual(transport.calls[0].timeout, 2.5)

    def test_build_reranker_uses_reranker_url_and_model(self) -> None:
        transport = FakeTransport(rerank_response([(0, 1.0)]))
        settings = Settings(reranker_url="http://rerank.test:9002/v1")
        built = build_reranker(settings, "rerank-model", transport=transport)

        self.assertEqual(built.model, "rerank-model")
        hits = built.rerank("q", ["d0"], 1)
        self.assertEqual(hits, [RerankHit(index=0, score=1.0)])
        self.assertEqual(transport.calls[0].url, "http://rerank.test:9002/v1/rerank")
        self.assertEqual(transport.calls[0].payload()["model"], "rerank-model")

    def test_builders_reject_blank_model(self) -> None:
        for build in (build_embedder, build_reranker):
            with self.subTest(build=build.__name__):
                with self.assertRaises(ValueError):
                    build(Settings(), " ")

    def test_builders_validate_batch_size_and_timeout(self) -> None:
        with self.assertRaises(ValueError):
            build_embedder(Settings(), "m", batch_size=0)
        with self.assertRaises(ValueError):
            build_reranker(Settings(), "m", timeout=math.inf)


if __name__ == "__main__":
    unittest.main()
