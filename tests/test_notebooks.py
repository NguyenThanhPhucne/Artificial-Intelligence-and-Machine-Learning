"""Checks on the executed notebooks that are committed with the project.

I re-execute every notebook before committing it; these tests catch a notebook that was saved with an
error in it, and confirm that the headings markers search for follow the report template exactly.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from nids import config

NOTEBOOKS = sorted((config.PROJECT_ROOT / "notebooks").glob("*.ipynb"))

TEMPLATE_HEADINGS = [
    "# 1 Report Introduction",
    "## 1.1 Dataset identification",
    "## 1.2 Supervised learning task identification",
    "# 2 Exploratory Data Analysis",
    "## 2.1 Question(s) identification",
    "## 2.2 Splitting the dataset",
    "## 2.3 Exploratory Data Analysis process and results",
    "## 2.4 EDA conclusions",
    "# 3 Experimental Design",
    "## 3.1 Identification of your chosen supervised learning algorithm(s)",
    "## 3.2 Identification of appropriate evaluation techniques",
    "## 3.3 Data cleaning and pre-processing transformations",
    "## 3.4 Limitations and Options",
    "# 4 Predictive Modelling / Model Development",
    "## 4.1 The predictive modelling process",
    '## 4.2 Evaluation results on "seen" data',
    "# 5 Evaluation and further modelling improvements",
    "# 6 Conclusion",
    "## 6.1 Summary of results",
    "## 6.2 Reflection on Individual Learning",
]


def _headings(path: Path) -> list[str]:
    notebook = nbformat.read(path, as_version=4)
    return [
        line.strip()
        for cell in notebook.cells if cell.cell_type == "markdown"
        for line in cell.source.splitlines() if line.startswith("#")
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_was_saved_without_errors(path):
    notebook = nbformat.read(path, as_version=4)
    errors = [
        f"cell {i}: {output.ename}: {output.evalue}"
        for i, cell in enumerate(notebook.cells) if cell.cell_type == "code"
        for output in cell.get("outputs", []) if output.output_type == "error"
    ]
    unexecuted = [i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code" and cell.execution_count is None]
    assert not errors, errors
    assert not unexecuted, f"cells never executed: {unexecuted}"


def test_every_template_heading_appears_across_notebooks():
    """Every template heading exists; each sub-section (##) lives in exactly one notebook.

    A top-level section may open in two notebooks (section 3 is split between notebooks 01 and 03,
    because data cleaning has to run before the split), but a sub-section must not be duplicated.
    """
    found = [h for path in NOTEBOOKS for h in _headings(path)]
    missing = [h for h in TEMPLATE_HEADINGS if h not in found]
    duplicated = [h for h in TEMPLATE_HEADINGS if h.startswith("## ") and found.count(h) > 1]
    assert not missing, f"missing headings: {missing}"
    assert not duplicated, f"headings used more than once: {duplicated}"
