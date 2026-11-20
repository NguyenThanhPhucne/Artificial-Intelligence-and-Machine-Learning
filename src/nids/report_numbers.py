"""A single store of the numbers my Word report quotes.

Each notebook records the values its copy-ready text uses, already formatted the way the report
shows them (for example "0.9953 ± 0.0089" or "1,179"). The report builder only substitutes these
strings into {{n:key}} placeholders, so a number in the report is always the number the code
produced, and re-running a notebook updates the report on the next build.
"""

from __future__ import annotations

import json

from . import config

NUMBERS_JSON = config.REPORTS_DIR / "numbers.json"


def load() -> dict[str, str]:
    """Return every recorded value (an empty dict before any notebook has run)."""
    if not NUMBERS_JSON.exists():
        return {}
    return json.loads(NUMBERS_JSON.read_text())


def record(**values: str | int | float) -> None:
    """Merge `values` into the store, overwriting existing keys, and keep the keys sorted.

    I refuse floats with more than four decimals: such a value has not been formatted for the reader
    yet, and silently printing 0.123456789 in a report would be a presentation error.
    """
    for key, value in values.items():
        if isinstance(value, float) and round(value, 4) != value:
            raise ValueError(f"Format {key} before recording it (got {value!r})")
    stored = load()
    stored.update({key: str(value) for key, value in values.items()})
    NUMBERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_JSON.write_text(json.dumps(dict(sorted(stored.items())), indent=1, ensure_ascii=False) + "\n")
