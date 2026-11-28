"""Tests for ``core.gold_set.load_gold_set`` (EV-02); neutral tmp_path fixtures only."""

from __future__ import annotations

import pytest

from core.gold_set import GoldLabel, GoldSetError, load_gold_set

STATUSES = {"strong", "standard", "weak", "red_flag", "silent", "na"}


def write(tmp_path, text: str):
    path = tmp_path / "labels.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_labels_and_skips_blank_lines(tmp_path):
    path = write(
        tmp_path,
        '{"primary_domain": "a.example", "criterion_id": "c1", "status": "strong"}\n'
        "\n   \n"
        '{"primary_domain": "b.example", "criterion_id": "c1", "status": "silent"}\n',
    )
    assert list(load_gold_set(path, STATUSES)) == [
        GoldLabel("a.example", "c1", "strong"),
        GoldLabel("b.example", "c1", "silent"),
    ]


@pytest.mark.parametrize(
    ("line", "lineno"),
    [
        ("not json secretvalue", 2),
        ('{"primary_domain": "x.example", "criterion_id": "c1"}', 2),
        ('{"primary_domain": "x.example", "criterion_id": "c1", "status": "bogus"}', 2),
        ('["a", "b"]', 2),
        ('{"primary_domain": 3, "criterion_id": "c1", "status": "strong"}', 2),
    ],
)
def test_bad_line_names_line_number_not_content(tmp_path, line, lineno):
    path = write(
        tmp_path,
        '{"primary_domain": "a.example", "criterion_id": "c1", "status": "strong"}\n' + line + "\n",
    )
    with pytest.raises(GoldSetError) as info:
        list(load_gold_set(path, STATUSES))
    assert f"line {lineno}" in str(info.value)
    assert "secretvalue" not in str(info.value)
    assert "bogus" not in str(info.value)


def test_duplicate_pair_is_error(tmp_path):
    row = '{"primary_domain": "a.example", "criterion_id": "c1", "status": "strong"}\n'
    path = write(tmp_path, row + row.replace("strong", "weak"))
    with pytest.raises(GoldSetError, match="line 2"):
        list(load_gold_set(path, STATUSES))
