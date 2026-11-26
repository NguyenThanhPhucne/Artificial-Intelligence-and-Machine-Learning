"""Checks on the executed notebooks that are committed with the project.

I re-execute every notebook before committing it; these tests catch a notebook that was saved with an
error in it, and confirm that the headings markers search for follow the report template exactly.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest

from nids import config

NOTEBOOKS = sorted((config.PROJECT_ROOT / "notebooks").glob("*.ipynb"))

# The notebooks keep the report's template headings, but only for sections that run code or show results; the
# text-only sections (introduction, questions, algorithm choice, limitations, conclusion, reflection) live in the
# Word report alone. A marker can still open the report beside the notebook and find each section's evidence.
TEMPLATE_HEADINGS = [
    "## 1.1 Dataset identification",
    "# 3 Experimental Design",
    "## 3.3 Data cleaning and pre-processing transformations",
    "# 2 Exploratory Data Analysis",
    "## 2.2 Splitting the dataset",
    "## 2.3 Exploratory Data Analysis process and results",
    "## 2.4 EDA conclusions",
    "## 3.2 Identification of appropriate evaluation techniques",
    "# 4 Predictive Modelling / Model Development",
    "## 4.1 The predictive modelling process",
    '## 4.2 Evaluation results on "seen" data',
    "# 5 Evaluation and further modelling improvements",
    "### Step 1: Model Selection",
    "### Step 2: Data Preparation",
    "### Step 3: Model Development",
    "### Step 4: Model Evaluation",
    "### Step 5: Model Improvement",
    "### Step 6: Validation",
    "### Step 7: Comparative Analysis of Multiple Algorithms",
]
REPORT_ONLY_HEADINGS = [
    "# 1 Report Introduction",
    "## 1.2 Supervised learning task identification",
    "## 2.1 Question(s) identification",
    "## 3.1 Identification of your chosen supervised learning algorithm(s)",
    "## 3.4 Limitations and Options",
    "# 6 Conclusion",
    "## 6.1 Summary of results",
    "## 6.2 Reflection on Individual Learning",
]
MAX_CODE_LINES = 40


def _headings(path: Path) -> list[str]:
    notebook = nbformat.read(path, as_version=4)
    return [
        line.strip()
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("#")
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_was_saved_without_errors(path):
    notebook = nbformat.read(path, as_version=4)
    errors = [
        f"cell {i}: {output.ename}: {output.evalue}"
        for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    unexecuted = [
        i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code" and cell.execution_count is None
    ]
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


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_colab_cell_tries_both_drive_locations(path):
    # Students upload the folder either inside CMP6202_NIDS or straight into My Drive (the README allows both),
    # so the setup cell must look in both places and pick the one that holds the nids package.
    first_code = next(c.source for c in nbformat.read(path, as_version=4).cells if c.cell_type == "code")
    assert "/content/drive/MyDrive/CMP6202_NIDS/Project" in first_code
    assert "/content/drive/MyDrive/Project" in first_code
    assert '"src", "nids"' in first_code


def test_report_only_sections_are_not_in_the_notebooks():
    found = [h for path in NOTEBOOKS for h in _headings(path)]
    assert not [h for h in REPORT_ONLY_HEADINGS if h in found]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cells_are_short_and_write_no_report_text(path):
    # Lab-style cells do one thing each; copy-ready report paragraphs belong in the Word report, not here.
    code = [c.source for c in nbformat.read(path, as_version=4).cells if c.cell_type == "code"]
    assert not [s.splitlines()[0] for s in code if re.search(r"save_text|TEXT_DIR|reports/text", s)]
    assert max(len(s.splitlines()) for s in code) <= MAX_CODE_LINES


def _outputs_text(path: Path) -> list[tuple[int, str]]:
    found = []
    for i, cell in enumerate(nbformat.read(path, as_version=4).cells):
        for output in cell.get("outputs", []):
            text = output.get("text", "") + "".join(output.get("data", {}).get("text/plain", ""))
            found.append((i, text if isinstance(text, str) else "".join(text)))
    return found


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_show_no_local_home_path(path):
    # Marking is anonymous: a printed path such as /Users/<name>/... would reveal the student.
    assert not [i for i, text in _outputs_text(path) if "/Users/" in text or "/home/" in text]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_unexplained_warnings_in_outputs(path):
    notebook = nbformat.read(path, as_version=4)
    warned = [
        i
        for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("name") == "stderr" and "Warning" in output.get("text", "")
    ]
    assert not warned


def test_template_headings_appear_in_template_order():
    found = [h for path in NOTEBOOKS for h in _headings(path)]
    first = [found.index(h) for h in TEMPLATE_HEADINGS if h in found]
    # "# 3 Experimental Design" opens in notebook 01 (cleaning) and again in 03; order is checked by first occurrence,
    # except that 3.2 (notebook 03) legitimately follows section 2 (notebook 02).
    expected = [h for h in TEMPLATE_HEADINGS]
    assert [h for _, h in sorted(zip(first, expected, strict=True))] == expected


def test_section_3_3_does_not_reuse_section_5_step_names():
    # Section 5's "Step 1"–"Step 7" are template headings; cleaning steps in 3.3 must not use the same words.
    nb01 = next(p for p in NOTEBOOKS if p.name.startswith("01"))
    assert not [h for h in _headings(nb01) if h.startswith("### Step")]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_last_cell_shows_the_recorded_numbers(path):
    # A marker checks a report number against the notebook, so the final record cell must display what it saved.
    last = [c for c in nbformat.read(path, as_version=4).cells if c.cell_type == "code"][-1]
    assert "report_numbers.record" in last.source and last.get("outputs")


def test_strengths_table_claims_match_the_measured_training_times():
    # The strengths/weaknesses table sits beside measured numbers; a claim like "fastest to train" must agree with them.
    import pandas as pd

    strengths = pd.read_csv(config.TABLES_DIR / "5_7_strengths_weaknesses.csv", index_col=0)
    fit = pd.read_csv(config.TABLES_DIR / "5_5_macro_selection.csv", index_col=0)["fit_seconds"]
    claimed = strengths.index[strengths["property that explains the result"].str.contains("fastest to train")]
    assert set(claimed) <= {fit.idxmin()}
