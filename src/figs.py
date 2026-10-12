"""Render article figures from results/. Light (LinkedIn) + dark (X) variants.

Usage:  uv run python -m src.figs

Palette is the dataviz-validated one from the first article (jev-router-lab),
same router -> colour mapping: Jev blue, Qwen3.8-27B orange, Qwen3.5-4B green.
"""

import matplotlib.pyplot as plt

from src.dataset import ROOT, RESULTS, load_records
from src.score import accuracy, calibration, error_overlap, latency, missed_pages_at

IMG = ROOT / "article" / "img"

MODES = {
    "light": {"surface": "#ffffff", "ink": "#1a1a19", "ink2": "#5f5e56",
              "grid": "#e6e5df", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "grid": "#3a3934", "series": ["#3987e5", "#d95926", "#199e70"]},
}

ROUTERS = [("jev", "Jev 1.13 (API)"), ("qwen3.8-27b", "Qwen3.8-27B (local)"),
           ("qwen3.5-4b", "Qwen3.5-4B (local)")]
MIN_N = 10  # calibration buckets with fewer alerts are not plotted

DECISIONS = [("category", "category"), ("priority", "priority"), ("page_human", "page a human?")]


def runs() -> dict:
    return {(r, v): load_records(RESULTS / f"{r}_{v}.jsonl")
            for r, _ in ROUTERS for v in ("plain", "rubric", "policy")}


def style(ax, c: dict) -> None:
    ax.set_facecolor(c["surface"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=c["ink2"], labelsize=8.5)
    ax.set_axisbelow(True)


def new_fig(c: dict, *args, **kw):
    fig, axes = plt.subplots(*args, dpi=200, **kw)
    fig.patch.set_facecolor(c["surface"])
    return fig, axes


def save(fig, name: str, mode: str, c: dict) -> None:
    fig.tight_layout()
    fig.savefig(IMG / f"{name}_{mode}.png", facecolor=c["surface"])
    plt.close(fig)


def accuracy_fig(R: dict, mode: str, c: dict) -> None:
    """Small multiples, one per decision: plain -> rubric per router."""
    fig, axes = new_fig(c, 1, 3, figsize=(10, 2.8), sharey=True)
    for ax, (d, title) in zip(axes, DECISIONS):
        style(ax, c)
        for i, (r, label) in enumerate(ROUTERS):
            plain = accuracy(R[(r, "plain")])[d]
            rubric = accuracy(R[(r, "rubric")])[d]
            y = len(ROUTERS) - 1 - i
            ax.plot([plain, rubric], [y, y], color=c["series"][i], linewidth=2, zorder=2)
            ax.scatter([plain], [y], s=40, facecolor=c["surface"], edgecolor=c["series"][i],
                       linewidth=2, zorder=3)
            ax.scatter([rubric], [y], s=40, color=c["series"][i], edgecolor=c["surface"],
                       linewidth=2, zorder=3)
            delta = round((rubric - plain) * 100)
            note = f"{rubric:.0%}" + (f"  ({delta:+d})" if delta else "")
            ax.text(max(plain, rubric) + 0.015, y, note, va="center", fontsize=8.5,
                    color=c["ink"])
        ax.set_title(title, color=c["ink"], fontsize=10.5, loc="left")
        ax.set_xlim(0.5, 1.05)
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(["50%", "75%", "100%"])
        ax.grid(axis="x", color=c["grid"], linewidth=0.7)
    axes[0].set_yticks(range(len(ROUTERS)))
    axes[0].set_yticklabels([label for _, label in reversed(ROUTERS)], color=c["ink"],
                            fontsize=9)
    fig.suptitle("Accuracy on 310 alerts. Hollow = plain definitions, solid = with the "
                 "priority rubric", color=c["ink"], fontsize=11, x=0.01, ha="left")
    save(fig, "accuracy", mode, c)


def calibration_fig(R: dict, mode: str, c: dict) -> None:
    """Reliability diagrams (rubric runs): claimed confidence vs actual hit rate."""
    fig, axes = new_fig(c, 1, 2, figsize=(9, 4.2), sharey=True)
    for ax, d in zip(axes, ("category", "priority")):
        style(ax, c)
        ax.plot([0.4, 1], [0.4, 1], color=c["ink2"], linewidth=1, linestyle="--", zorder=1)
        ax.text(0.42, 0.47, "perfectly calibrated", color=c["ink2"], fontsize=8, rotation=38)
        for i, (r, label) in enumerate(ROUTERS):
            cal = calibration(R[(r, "rubric")], d)
            # buckets under MIN_N alerts are noise (one miss = a 30-point swing)
            pts = [(b["mean_conf"], b["hit_rate"], b["n"]) for k, b in cal.items()
                   if k != "ece" and b["n"] >= MIN_N]
            xs, ys, ns = zip(*pts)
            ax.plot(xs, ys, color=c["series"][i], linewidth=2, zorder=2)
            ax.scatter(xs, ys, s=[max(16, n * 0.6) for n in ns], color=c["series"][i],
                       edgecolor=c["surface"], linewidth=2, zorder=3,
                       label=f"{label}  ECE {cal['ece']:.2f}")
        ax.set_title(f"{d} (rubric prompt)", color=c["ink"], fontsize=10.5, loc="left")
        ax.set_xlim(0.4, 1.02)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("confidence the router claimed (bucket mean)", color=c["ink2"],
                      fontsize=8.5)
        ax.grid(color=c["grid"], linewidth=0.7)
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=c["ink"],
                  markerscale=0.5)
    axes[0].set_ylabel("how often it was right", color=c["ink2"], fontsize=8.5)
    fig.suptitle("When it says 0.9, is it right 90% of the time? Dot size = alerts in "
                 f"bucket; buckets under {MIN_N} alerts not shown", color=c["ink"],
                 fontsize=11, x=0.01, ha="left")
    save(fig, "calibration", mode, c)


def overlap_fig(R: dict, mode: str, c: dict) -> None:
    """Priority misses (rubric): shared vs unique, Jev vs each Qwen."""
    fig, ax = new_fig(c, figsize=(9, 2.6))
    style(ax, c)
    jev = R[("jev", "rubric")]
    rows = []
    for i, (r, label) in enumerate(ROUTERS[1:], start=1):
        o = error_overlap(jev, R[(r, "rubric")], "priority")
        rows.append((f"Jev vs {label.split(' (')[0]}", o, c["series"][i]))
    for y, (label, o, qc) in enumerate(reversed(rows)):
        segs = [(o["only_a"], c["series"][0], "only Jev"), (o["both"], c["ink2"], "both"),
                (o["only_b"], qc, "only Qwen")]
        left = 0
        for n, col, name in segs:
            ax.barh(y, n, left=left, color=col, edgecolor=c["surface"], linewidth=2, height=0.55)
            if n >= 12:
                ax.text(left + n / 2, y, f"{name} {n}", ha="center", va="center",
                        fontsize=8.5, color=c["surface"])
            else:  # too thin to hold its label: put it after the bar
                ax.text(left + n + 1.5, y, f"{name} {n}", va="center", fontsize=8.5,
                        color=c["ink"])
            left += n
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _ in reversed(rows)], color=c["ink"], fontsize=9)
    ax.set_xlabel("priority misses out of 310 alerts", color=c["ink2"], fontsize=8.5)
    ax.grid(axis="x", color=c["grid"], linewidth=0.7)
    ax.set_xlim(0, 130)
    ax.set_title("Priority misses. Grey = alerts both got wrong: most of Jev's misses "
                 "are Qwen's too",
                 color=c["ink"], fontsize=11, loc="left")
    save(fig, "overlap", mode, c)


def latency_fig(R: dict, mode: str, c: dict) -> None:
    """p50 bar + p99 tick per router (rubric runs)."""
    fig, ax = new_fig(c, figsize=(9, 2.6))
    style(ax, c)
    for i, (r, label) in enumerate(ROUTERS):
        lat = latency(R[(r, "rubric")])
        y = len(ROUTERS) - 1 - i
        ax.barh(y, lat["p50"], color=c["series"][i], height=0.5, edgecolor=c["surface"],
                linewidth=2)
        ax.plot([lat["p99"]] * 2, [y - 0.25, y + 0.25], color=c["ink"], linewidth=2)
        ax.text(lat["p99"] + 40, y, f"p50 {lat['p50']:.0f}ms · p99 {lat['p99']:.0f}ms",
                va="center", fontsize=8.5, color=c["ink"])
    ax.set_yticks(range(len(ROUTERS)))
    ax.set_yticklabels([label for _, label in reversed(ROUTERS)], color=c["ink"], fontsize=9)
    ax.set_xlabel("ms per alert, all three decisions (bar = p50, tick = p99)",
                  color=c["ink2"], fontsize=8.5)
    ax.grid(axis="x", color=c["grid"], linewidth=0.7)
    ax.set_xlim(0, 3600)
    ax.set_title("Latency per alert (rubric prompt)", color=c["ink"], fontsize=11, loc="left")
    save(fig, "latency", mode, c)


def paging_fig(R: dict, mode: str, c: dict) -> None:
    """Missed pages vs false pages as each router's P(page) threshold moves.
    Lower-left is better; compares routers at matched false-page budgets."""
    fig, ax = new_fig(c, figsize=(9, 3.8))
    style(ax, c)
    tfidf = load_records(RESULTS / "tfidf-logreg_cv.jsonl")
    lines = [(label, R[(r, "rubric")], c["series"][i]) for i, (r, label) in enumerate(ROUTERS)]
    lines.append(("TF-IDF + logistic regression", tfidf, c["ink2"]))
    budgets = range(0, 101)
    for label, rs, col in lines:
        missed = [missed_pages_at(rs, b)["missed"] for b in budgets]
        ax.plot(list(budgets), missed, color=col, linewidth=2, label=label)
        default = sum(1 for r in rs if r["gold"]["page_human"]
                      and r["answers"]["page_human"]["noul"] < 0.5)
        false = sum(1 for r in rs if not r["gold"]["page_human"]
                    and r["answers"]["page_human"]["noul"] >= 0.5)
        if false <= 100:
            ax.scatter([false], [default], s=40, color=col, edgecolor=c["surface"],
                       linewidth=2, zorder=3)
    ax.set_xlabel("false pages allowed (of 168 alerts that shouldn't page)", color=c["ink2"],
                  fontsize=8.5)
    ax.set_ylabel("missed pages (of 142)", color=c["ink2"], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)
    ax.grid(color=c["grid"], linewidth=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=c["ink"])
    ax.set_title("Paging trade-off: lower is better. Dots = each router at its default "
                 "0.5 threshold", color=c["ink"], fontsize=11, loc="left")
    save(fig, "paging", mode, c)


RUNWAY = ("a016", "a096", "a127", "a141", "a151", "a187", "a288")  # runway in days/weeks


def rules_fig(R: dict, mode: str, c: dict) -> None:
    """Two kinds of rule, before -> after adding it: a rule about a whole
    category (security -> P1) vs one that needs a detail (runway -> P3)."""
    fig, axes = new_fig(c, 1, 2, figsize=(10, 2.8), sharey=True)
    panels = [
        ("'any security anomaly -> P1'\nsecurity alerts labeled P1, answered P1 (of 29)",
         lambda r: r["gold"]["category"] == "security" and r["gold"]["priority"] == "P1", "P1", 29),
        ("'days of runway -> P3'\nrunway-in-days alerts answered P3 (of 7)",
         lambda r: r["id"] in RUNWAY, "P3", 7),
    ]
    for ax, (title, keep, label, n) in zip(axes, panels):
        style(ax, c)
        for i, (r, name) in enumerate(ROUTERS):
            before, after = (sum(x["answers"]["priority"]["choice"] == label
                                 for x in R[(r, v)] if keep(x)) / n for v in ("rubric", "policy"))
            y = len(ROUTERS) - 1 - i
            ax.plot([before, after], [y, y], color=c["series"][i], linewidth=2, zorder=2)
            ax.scatter([before], [y], s=40, facecolor=c["surface"], edgecolor=c["series"][i],
                       linewidth=2, zorder=3)
            ax.scatter([after], [y], s=40, color=c["series"][i], edgecolor=c["surface"],
                       linewidth=2, zorder=3)
            ax.text(max(before, after) + 0.03, y, f"{round(before * n)} -> {round(after * n)}",
                    va="center", fontsize=8.5, color=c["ink"])
        ax.set_title(title, color=c["ink"], fontsize=9.5, loc="left")
        ax.set_xlim(-0.05, 1.2)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.grid(axis="x", color=c["grid"], linewidth=0.7)
    axes[0].set_yticks(range(len(ROUTERS)))
    axes[0].set_yticklabels([name for _, name in reversed(ROUTERS)], color=c["ink"], fontsize=9)
    save(fig, "rules", mode, c)


def method_fig(R: dict, mode: str, c: dict, alert_id: str = "a002") -> None:
    """One real alert -> the category probabilities Qwen3.5-4B put on each label."""
    r = next(x for x in R[("qwen3.5-4b", "rubric")] if x["id"] == alert_id)
    probs = sorted(r["answers"]["category"]["probabilities"].items(), key=lambda kv: kv[1])
    fig, ax = new_fig(c, figsize=(9, 2.8))
    style(ax, c)
    for y, (label, p) in enumerate(probs):
        ax.barh(y, p, color=c["series"][2], height=0.55, edgecolor=c["surface"], linewidth=2)
        ax.text(p + 0.01, y, f"{p:.2f}", va="center", fontsize=8.5, color=c["ink"])
    ax.set_yticks(range(len(probs)))
    ax.set_yticklabels([label for label, _ in probs], color=c["ink"], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([])
    ax.set_title(f"alert: \"{r['alert']}\"\nQwen3.5-4B, one forward pass, no text "
                 f"generated. Gold label: {r['gold']['category']}",
                 color=c["ink"], fontsize=9.5, loc="left")
    save(fig, "method", mode, c)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    R = runs()
    for mode, c in MODES.items():
        for fig in (rules_fig, accuracy_fig, calibration_fig, overlap_fig, paging_fig, latency_fig):
            fig(R, mode, c)
    print(f"wrote {sorted(p.name for p in IMG.glob('*.png'))}")
