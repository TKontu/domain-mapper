"""Scope pack models and loader (ARCHITECTURE.md §3).

`load_scope_pack` is called by `landscape validate` (SP-02); `scope_pack_json_schemas` is called
by `landscape schema` (SP-03).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ScopePackError(ValueError):
    """A scope pack file is missing, unparsable or inconsistent."""

    def __init__(self, file: str, message: str) -> None:
        self.file = file
        super().__init__(f"{file}: {message}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- scope.yaml ---------------------------------------------------------------


class Segment(_Model):
    id: str
    label: str
    definition: str
    subtypes: list[str] = Field(default_factory=list)


class Layer(_Model):
    id: str
    countries: list[str] = Field(min_length=1)
    discovery_target: Literal["exhaustive", "near_exhaustive", "notable"]
    rubric_depth: Literal["full", "core"]


class CrawlPolicy(_Model):
    """Per-entity crawl policy (ARCHITECTURE.md §4 stage 4; CR-05).

    ``max_pages`` bounds all URLs crawled per entity; ``priority_paths`` are absolute paths on the
    entity's primary domain, crawled after its homepage in listed order.
    """

    max_pages: int = Field(default=1, ge=1, le=50)
    priority_paths: list[str] = Field(default_factory=list)
    # CR-06: fill the rest of ``max_pages`` with same-host links from fetched pages, breadth-first.
    discover_links: bool = False

    @field_validator("priority_paths")
    @classmethod
    def _valid_paths(cls, paths: list[str]) -> list[str]:
        seen: set[str] = set()
        for path in paths:
            if not path.startswith("/"):
                raise ValueError(f"priority path must start with '/': {path!r}")
            if path.startswith("//"):
                raise ValueError(f"priority path must not name a host: {path!r}")
            if "?" in path or "#" in path:
                raise ValueError(f"priority path must not have a query or fragment: {path!r}")
            if path == "/":
                raise ValueError("priority path '/' duplicates the homepage")
            if path in seen:
                raise ValueError(f"duplicate priority path: {path!r}")
            seen.add(path)
        return paths


class Scope(_Model):
    id: str
    version: str
    title: str
    purpose: str | None = None
    perspective: str
    segments: list[Segment] = Field(min_length=1)
    layers: list[Layer] = Field(min_length=1)
    languages: dict[str, list[str]] = Field(default_factory=dict)
    crawl: CrawlPolicy = Field(default_factory=CrawlPolicy)


# --- rubric.yaml --------------------------------------------------------------


class Status(_Model):
    id: str
    label: str
    value: float | None
    color: str


class Criterion(_Model):
    id: str
    question: str
    applies_to: list[str] = Field(min_length=1)
    statuses: dict[str, str]
    retrieval_hints: list[str] = Field(default_factory=list)
    verify_with: str | None = None
    evidence_min: int = Field(default=1, ge=0)
    core: bool = False


class Topic(_Model):
    id: str
    label: str
    segments: list[str] = Field(default_factory=list)  # empty = all segments
    criteria: list[Criterion] = Field(min_length=1)


class Rubric(_Model):
    status_scale: list[Status] = Field(min_length=1)
    topics: list[Topic] = Field(min_length=1)


# --- optional files -----------------------------------------------------------


class VocabularyEntry(_Model):
    id: str
    segments: list[str] = Field(default_factory=list)
    terms: dict[str, list[str]]  # language -> terms


class Vocabulary(_Model):
    entries: list[VocabularyEntry] = Field(default_factory=list)


class Source(_Model):
    id: str
    role: Literal["discovery", "register", "verification"]
    adapter: str
    countries: list[str] = Field(default_factory=list)
    access: str | None = None
    licence: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Sources(_Model):
    sources: list[Source] = Field(default_factory=list)


class Attribute(_Model):
    id: str
    label: str
    type: Literal["int", "float", "enum", "bool", "date"]
    unit: str | None = None
    values: list[str] = Field(default_factory=list)  # for enum
    applies_to: list[str] = Field(default_factory=list)


class Attributes(_Model):
    attributes: list[Attribute] = Field(default_factory=list)


class Profile(_Model):
    id: str
    label: str
    weights: dict[str, float]  # topic or criterion id -> weight


class Profiles(_Model):
    profiles: list[Profile] = Field(default_factory=list)


class View(_Model):
    id: str
    label: str
    kind: str
    options: dict[str, Any] = Field(default_factory=dict)


class Views(_Model):
    views: list[View] = Field(default_factory=list)


class ScopePack(_Model):
    path: Path
    scope: Scope
    rubric: Rubric
    vocabulary: Vocabulary = Vocabulary()
    sources: Sources = Sources()
    attributes: Attributes = Attributes()
    profiles: Profiles = Profiles()
    views: Views = Views()


# --- loading ------------------------------------------------------------------

_OPTIONAL: dict[str, type[_Model]] = {
    "vocabulary": Vocabulary,
    "sources": Sources,
    "attributes": Attributes,
    "profiles": Profiles,
    "views": Views,
}


_FILE_MODELS: dict[str, type[_Model]] = {
    "scope.yaml": Scope,
    "rubric.yaml": Rubric,
    **{f"{name}.yaml": model for name, model in _OPTIONAL.items()},
}

_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def scope_pack_json_schemas() -> dict[str, dict[str, Any]]:
    """Return a JSON Schema (draft 2020-12) for each scope-pack YAML file, keyed by file name."""
    return {
        file: {"$schema": _JSON_SCHEMA_DIALECT, **model.model_json_schema()}
        for file, model in _FILE_MODELS.items()
    }


def _read(path: Path, required: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ScopePackError(path.name, "required file is missing")
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScopePackError(path.name, f"invalid YAML: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ScopePackError(path.name, "top level must be a mapping")
    return data


def _parse[M: _Model](model: type[M], data: dict[str, Any], file: str) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ScopePackError(file, details) from exc


def _unique(ids: Iterable[str], file: str, what: str) -> None:
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        raise ScopePackError(file, f"duplicate {what} id(s): {', '.join(dupes)}")


def _known(refs: Iterable[str], known: set[str], file: str, where: str) -> None:
    unknown = sorted(set(refs) - known)
    if unknown:
        raise ScopePackError(file, f"{where} references unknown id(s): {', '.join(unknown)}")


def _check(pack: ScopePack) -> None:
    scope, rubric = pack.scope, pack.rubric
    _unique((s.id for s in scope.segments), "scope.yaml", "segment")
    _unique((la.id for la in scope.layers), "scope.yaml", "layer")
    _unique((s.id for s in rubric.status_scale), "rubric.yaml", "status")
    _unique((t.id for t in rubric.topics), "rubric.yaml", "topic")
    _unique((c.id for t in rubric.topics for c in t.criteria), "rubric.yaml", "criterion")
    segments = {s.id for s in scope.segments}
    statuses = {s.id for s in rubric.status_scale}
    for topic in rubric.topics:
        _known(topic.segments, segments, "rubric.yaml", f"topics.{topic.id}.segments")
        for c in topic.criteria:
            where = f"topics.{topic.id}.criteria.{c.id}"
            _known(c.applies_to, segments, "rubric.yaml", f"{where}.applies_to")
            _known(c.statuses, statuses, "rubric.yaml", f"{where}.statuses")
    _unique((e.id for e in pack.vocabulary.entries), "vocabulary.yaml", "entry")
    for e in pack.vocabulary.entries:
        _known(e.segments, segments, "vocabulary.yaml", f"entries.{e.id}.segments")
    _unique((s.id for s in pack.sources.sources), "sources.yaml", "source")
    _unique((a.id for a in pack.attributes.attributes), "attributes.yaml", "attribute")
    for a in pack.attributes.attributes:
        _known(a.applies_to, segments, "attributes.yaml", f"attributes.{a.id}.applies_to")
    _unique((p.id for p in pack.profiles.profiles), "profiles.yaml", "profile")
    _unique((v.id for v in pack.views.views), "views.yaml", "view")


def load_scope_pack(path: str | Path) -> ScopePack:
    """Load and validate the scope pack in directory `path`."""
    root = Path(path)
    scope_data = _read(root / "scope.yaml", required=True)
    rubric_data = _read(root / "rubric.yaml", required=True)
    assert scope_data is not None and rubric_data is not None
    parts: dict[str, Any] = {
        "scope": _parse(Scope, scope_data, "scope.yaml"),
        "rubric": _parse(Rubric, rubric_data, "rubric.yaml"),
    }
    for name, model in _OPTIONAL.items():
        file = f"{name}.yaml"
        data = _read(root / file, required=False)
        if data is not None:
            parts[name] = _parse(model, data, file)
    pack = ScopePack(path=root, **parts)
    _check(pack)
    return pack
