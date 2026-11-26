"""Tests for the store of report numbers that the Word report reads through {{n:key}} placeholders."""

from __future__ import annotations

import json

import pytest

from nids import report_numbers


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "numbers.json"
    monkeypatch.setattr(report_numbers, "NUMBERS_JSON", path)
    return path


def test_record_merges_and_overwrites(store):
    report_numbers.record(a="1", b=2)
    report_numbers.record(b=3, c=0.5)
    assert report_numbers.load() == {"a": "1", "b": "3", "c": "0.5"}
    assert list(json.loads(store.read_text())) == ["a", "b", "c"]  # keys sorted for clean diffs


def test_record_rejects_unformatted_floats(store):
    with pytest.raises(ValueError, match="x"):
        report_numbers.record(x=0.123456789)  # format it first, as the report text will show it


def test_record_returns_what_it_saved_so_the_notebook_can_show_it(store):
    # The last cell of each notebook displays the recorded values, so a marker can match them to the report.
    assert report_numbers.record(a="1", b="2,345") == {"a": "1", "b": "2,345"}


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ([], ""),
        (["Decision Tree"], "Decision Tree"),
        (["Decision Tree", "Random Forest"], "Decision Tree and Random Forest"),
        (iter(["A", "B", "C"]), "A, B and C"),
    ],
)
def test_english_list_joins_names_as_the_report_writes_them(names, expected):
    assert report_numbers.english_list(names) == expected
