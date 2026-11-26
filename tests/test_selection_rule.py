"""Tests for the pre-registered selection rule and the tagged search cache (H3: fingerprint-free models)."""

from __future__ import annotations

import pandas as pd

from nids import config, evaluation, models


def test_highest_cv_macro_f1_wins_when_no_other_model_is_within_its_standard_deviation():
    mean = pd.Series({"A": 0.990, "B": 0.999, "C": 0.950})
    std = pd.Series({"A": 0.001, "B": 0.0001, "C": 0.01})
    speed = pd.Series({"A": 0.1, "B": 9.0, "C": 0.01})
    chosen, tied = evaluation.apply_selection_rule(mean, std, speed)
    assert chosen == "B" and tied == ["B"]


def test_a_tie_within_the_leaders_standard_deviation_goes_to_the_faster_model():
    mean = pd.Series({"A": 0.995, "B": 0.999, "C": 0.950})
    std = pd.Series({"A": 0.001, "B": 0.005, "C": 0.01})
    speed = pd.Series({"A": 0.5, "B": 9.0, "C": 0.01})
    chosen, tied = evaluation.apply_selection_rule(mean, std, speed)
    assert tied == ["A", "B"] and chosen == "A"  # C is fastest but not tied


def test_a_tag_gives_a_separate_cache_file_and_no_tag_keeps_the_old_name():
    old = models._cache_path(models.TREE, 20, 42)
    assert old == config.MODELS_DIR / "decision_tree_search_n20_seed42.joblib"
    tagged = models._cache_path(models.TREE, 20, 42, tag="nofp")
    assert tagged.name == "decision_tree_search_n20_seed42_nofp.joblib" and tagged != old
