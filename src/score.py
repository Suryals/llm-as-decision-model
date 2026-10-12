"""Score result files: accuracy, calibration, error overlap, latency.

Usage:  uv run python -m src.score results/*.jsonl

calibration() was written by Claude at Chief's request (the hand-written
spine rule was waived for this project); its spec is tests/test_score.py.
"""

import math
import sys

from src.dataset import load_records

DECISIONS = ("category", "priority", "page_human")
BUCKETS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))


def correct(record: dict, decision: str) -> bool:
    answer, gold = record["answers"][decision], record["gold"][decision]
    if decision == "page_human":
        return (answer["noul"] >= 0.5) == gold
    return answer["choice"] == gold


def accuracy(records: list[dict]) -> dict[str, float]:
    return {d: sum(correct(r, d) for r in records) / len(records) for d in DECISIONS}


def calibration(records: list[dict], decision: str) -> dict:
    """When the router says confidence 0.9, is it right ~90% of the time?

    Buckets the top-choice confidence, then per bucket compares the mean
    confidence (what it claimed) with the hit rate (what it delivered).
    A calibrated router has the two nearly equal. `ece` (expected
    calibration error) is the gap averaged over all records: 0 is perfect.
    The top bucket includes 1.0.
    """
    pairs = [(r["answers"][decision]["confidence"], correct(r, decision)) for r in records]
    out: dict = {}
    ece = 0.0
    for lo, hi in BUCKETS:
        inside = [(c, ok) for c, ok in pairs if lo <= c < hi or (hi == 1.0 and c == 1.0)]
        key = f"{lo}-{hi}"
        if not inside:
            out[key] = {"n": 0, "mean_conf": None, "hit_rate": None}
            continue
        mean_conf = sum(c for c, _ in inside) / len(inside)
        hit_rate = sum(ok for _, ok in inside) / len(inside)
        out[key] = {"n": len(inside), "mean_conf": round(mean_conf, 4), "hit_rate": hit_rate}
        ece += len(inside) / len(pairs) * abs(mean_conf - hit_rate)
    out["ece"] = round(ece, 4)
    return out


def error_overlap(a: list[dict], b: list[dict], decision: str) -> dict[str, int]:
    """Do two routers miss the same alerts? Shared misses point at the
    labels or the question; unique misses point at the model."""
    miss_a = {r["id"] for r in a if not correct(r, decision)}
    miss_b = {r["id"] for r in b if not correct(r, decision)}
    return {"both": len(miss_a & miss_b), "only_a": len(miss_a - miss_b),
            "only_b": len(miss_b - miss_a)}


def missed_pages_at(records: list[dict], max_false: int) -> dict[str, int]:
    """Fewest missed pages a router can reach while raising at most
    `max_false` false pages, choosing its own P(page) threshold.

    Compares routers at a matched false-page budget, so a router that is
    simply more trigger-happy at 0.5 doesn't look better than it is."""
    scored = [(r["answers"]["page_human"]["noul"], r["gold"]["page_human"]) for r in records]
    n_pages = sum(g for _, g in scored)
    best = {"missed": n_pages, "false": 0}  # threshold above everything: page nothing
    for t in sorted({p for p, _ in scored}, reverse=True):
        false = sum(1 for p, g in scored if p >= t and not g)
        if false > max_false:
            break  # lowering the threshold further only adds false pages
        missed = sum(1 for p, g in scored if p < t and g)
        if missed < best["missed"]:
            best = {"missed": missed, "false": false}
    return best


def latency(records: list[dict]) -> dict[str, float]:
    """Nearest-rank percentiles of per-alert latency (all three decisions)."""
    ms = sorted(r["latency_ms"] for r in records)

    def pct(p: float) -> float:
        return ms[max(0, math.ceil(p * len(ms)) - 1)]

    return {"p50": pct(0.50), "p99": pct(0.99), "max": ms[-1]}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        records = load_records(path)
        acc = accuracy(records)
        lat = latency(records)
        print(f"{path}  n={len(records)}")
        print("  " + "  ".join(f"{d} {acc[d]:.0%}" for d in DECISIONS))
        for d in ("category", "priority"):
            print(f"  {d} ECE {calibration(records, d)['ece']:.3f}")
        print(f"  latency p50 {lat['p50']:.0f}ms  p99 {lat['p99']:.0f}ms")
