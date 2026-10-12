"""Paths and record I/O shared by the runners and the scorer.

Every run writes results/<router>_<variant>.jsonl, one record per alert:
    {"id", "alert", "gold", "router", "variant",
     "answers": {decision: {...}}, "latency_ms", "usage", ...extras}
Fixed filenames (no timestamps) so the article's numbers point at stable
files; re-running an arm overwrites it.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALERTS = ROOT / "data" / "alerts.jsonl"
RESULTS = ROOT / "results"


def load_alerts(limit: int | None = None) -> list[dict]:
    return [json.loads(line) for line in ALERTS.open()][:limit]


def results_path(router: str, variant: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / f"{router}_{variant}.jsonl"


def load_records(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open()]
