"""Label a new CICFlowMeter CSV with my recommended model (the Decision Tree).

The Week 6 lab ends by predicting on new data; this module does the same for flow files. It applies
the row-wise cleaning from notebook 01 (non-finite and impossible negative values are dropped and
counted) and then the fitted pipeline, which already contains every learned pre-processing step.

Usage:  uv run python -m nids.predict path/to/flows.csv [--out predictions.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from . import config, data, evaluation, models

FINAL_MODEL_PATH = config.MODELS_DIR / "final_decision_tree.joblib"


def load_or_fit_final_model() -> BaseEstimator:
    """Return the recommended Decision Tree, fitted on the whole training set.

    Notebook 04 saves this model. If the file is missing (for example in a fresh clone, because
    models/ is not in the repository), I rebuild it the same way: the Decision Tree configuration
    with the best cross-validated macro F1 from the cached or re-run search (about 20 seconds),
    refitted on the 60,000 training flows. I only load a file that this function or notebook 04 wrote
    into the local models/ folder, because joblib files can execute code when loaded.
    """
    if FINAL_MODEL_PATH.exists():
        return joblib.load(FINAL_MODEL_PATH)
    split = data.load_split(data.load_sample())
    run = models.load_or_run_search(models.TREE, split.X_train, split.y_train)
    pipeline, _, _ = evaluation.selected_pipeline(run, "f1_macro")
    model, _ = evaluation.fit_timed(pipeline, split.X_train, split.y_train)
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, FINAL_MODEL_PATH)
    return model


def predict_csv(path: Path, model: BaseEstimator | None = None) -> tuple[pd.DataFrame, int]:
    """Return (row number in the file, predicted label) for every usable row, and the rows dropped.

    A Label column is optional: if present it is ignored, so the same function can score a labelled
    test file. A missing feature column is an error rather than a silent default, because inventing a
    value for a flow statistic would change the prediction.
    """
    model = load_or_fit_final_model() if model is None else model
    frame = pd.concat(data.read_raw_in_chunks(Path(path)))
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(f"The file is missing columns the model needs: {', '.join(missing)}")
    if config.LABEL_COL not in frame.columns:
        frame[config.LABEL_COL] = "unknown"  # clean_chunk counts removals per label
    kept = data.clean_chunk(frame).kept
    predicted = model.predict(kept[expected])
    return pd.DataFrame({"row": kept.index, "predicted_label": predicted}), len(frame) - len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label CICFlowMeter flows as Benign, FTP or SSH brute force.")
    parser.add_argument("csv", type=Path, help="CSV with the CSE-CIC-IDS2018 CICFlowMeter columns")
    parser.add_argument("--out", type=Path, default=None, help="where to write row,predicted_label")
    args = parser.parse_args()
    predictions, dropped = predict_csv(args.csv)
    print(f"Scored {len(predictions):,} flows; dropped {dropped:,} rows with non-finite or impossible values.")
    print(predictions["predicted_label"].value_counts().to_string())
    if args.out is not None:
        predictions.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
