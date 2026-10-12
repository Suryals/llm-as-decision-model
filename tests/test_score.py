"""Hand-computed cases for the scorer. Each expected number is worked out in
the comment next to it, so the test doubles as the spec for calibration()."""

import pytest

from src.score import accuracy, calibration, error_overlap, latency


def rec(id_, cat, conf, gold_cat, prio="P2", gold_prio="P2", page=0.1, gold_page=False,
        ms=100.0):
    return {
        "id": id_,
        "gold": {"category": gold_cat, "priority": gold_prio, "page_human": gold_page},
        "answers": {
            "category": {"choice": cat, "confidence": conf},
            "priority": {"choice": prio, "confidence": 0.8},
            "page_human": {"noul": page},
        },
        "latency_ms": ms,
    }


# 5 category answers:
#   conf 0.95 right, 0.92 right, 0.91 wrong   -> bucket 0.9-1.0
#   conf 0.75 right                           -> bucket 0.7-0.9
#   conf 0.55 wrong                           -> bucket 0.5-0.7
RECORDS = [
    rec("a", "noise", 0.95, "noise"),
    rec("b", "capacity", 0.92, "capacity"),
    rec("c", "security", 0.91, "availability"),
    rec("d", "performance", 0.75, "performance"),
    rec("e", "noise", 0.55, "capacity"),
]


def test_calibration_buckets():
    cal = calibration(RECORDS, "category")
    # 0.9-1.0: mean conf (0.95+0.92+0.91)/3 = 0.9267, hit 2/3 = 0.6667
    assert cal["0.9-1.0"]["n"] == 3
    assert cal["0.9-1.0"]["mean_conf"] == pytest.approx(0.9267, abs=1e-4)
    assert cal["0.9-1.0"]["hit_rate"] == pytest.approx(2 / 3)
    # 0.7-0.9: one record, conf 0.75, right
    assert cal["0.7-0.9"] == {"n": 1, "mean_conf": 0.75, "hit_rate": 1.0}
    # 0.5-0.7: one record, conf 0.55, wrong
    assert cal["0.5-0.7"] == {"n": 1, "mean_conf": 0.55, "hit_rate": 0.0}


def test_calibration_low_bucket_and_ece():
    cal = calibration(RECORDS, "category")
    # nothing below 0.5 -> the bucket exists but is empty
    assert cal["0.0-0.5"]["n"] == 0
    # ECE = sum over buckets of n/N * |mean_conf - hit_rate|
    #     = 3/5*|0.9267-0.6667| + 1/5*|0.75-1| + 1/5*|0.55-0|
    #     = 0.156 + 0.05 + 0.11 = 0.316
    assert cal["ece"] == pytest.approx(0.316, abs=1e-3)


def test_calibration_conf_exactly_one_lands_in_top_bucket():
    cal = calibration([rec("x", "noise", 1.0, "noise")], "category")
    assert cal["0.9-1.0"]["n"] == 1


def test_accuracy_uses_half_threshold_for_page_human():
    rs = [rec("a", "noise", 0.9, "noise", page=0.5, gold_page=True),
          rec("b", "noise", 0.9, "capacity", page=0.49, gold_page=True)]
    acc = accuracy(rs)
    assert acc["category"] == 0.5
    assert acc["page_human"] == 0.5  # 0.5 counts as yes, 0.49 as no


def test_error_overlap_counts_shared_and_unique_misses():
    a = [rec("1", "noise", 0.9, "noise"), rec("2", "x", 0.9, "noise"), rec("3", "x", 0.9, "noise")]
    b = [rec("1", "x", 0.9, "noise"), rec("2", "x", 0.9, "noise"), rec("3", "noise", 0.9, "noise")]
    # a misses {2,3}; b misses {1,2}
    assert error_overlap(a, b, "category") == {"both": 1, "only_a": 1, "only_b": 1}


def test_latency_percentiles():
    rs = [rec(str(i), "noise", 0.9, "noise", ms=float(i)) for i in range(1, 101)]
    assert latency(rs) == {"p50": 50.0, "p99": 99.0, "max": 100.0}


def test_missed_pages_at_matched_false_pages():
    from src.score import missed_pages_at
    # gold pages: p=0.9, 0.6, 0.3 ; gold no-page: p=0.8, 0.2
    rs = [rec("a", "noise", 0.9, "noise", page=0.9, gold_page=True),
          rec("b", "noise", 0.9, "noise", page=0.6, gold_page=True),
          rec("c", "noise", 0.9, "noise", page=0.3, gold_page=True),
          rec("d", "noise", 0.9, "noise", page=0.8, gold_page=False),
          rec("e", "noise", 0.9, "noise", page=0.2, gold_page=False)]
    # allow 0 false pages: threshold must sit above 0.8 -> only "a" pages, 2 missed
    assert missed_pages_at(rs, max_false=0) == {"missed": 2, "false": 0}
    # allow 1 false page: threshold can drop to just above 0.2 -> pages a,b,c,d; 0 missed
    assert missed_pages_at(rs, max_false=1) == {"missed": 0, "false": 1}
