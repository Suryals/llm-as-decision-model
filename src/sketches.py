"""Whiteboard-style explainer diagrams for the article (not data charts).

Usage:  uv run python -m src.sketches

Light = whiteboard (LinkedIn), dark = chalkboard (X). Numbers that appear in
a sketch are read from results/ or computed in the article, never typed in by
hand where a result file has them. Falls back to a plain font if no
handwriting font is installed (Chalkboard SE ships with macOS).
"""

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

from src.dataset import ROOT, RESULTS, load_records

IMG = ROOT / "article" / "img"
SCALE = 1.5  # fonts read small once the page shrinks a 12in canvas to column width
HAND = next((f for f in ("Chalkboard SE", "Humor Sans", "Comic Neue")
             if any(f == x.name for x in font_manager.fontManager.ttflist)), "DejaVu Sans")

MODES = {
    "light": {"bg": "#fdfdfb", "ink": "#24231f", "ink2": "#6b6a62", "note": "#fff4c2",
              "jev": "#2a78d6", "q27": "#eb6834", "q4": "#1baf7a", "bad": "#c73a3a"},
    "dark": {"bg": "#1f2420", "ink": "#f2f1ea", "ink2": "#b9b8ae", "note": "#3b3a2a",
             "jev": "#5ea0f0", "q27": "#f08a5d", "q4": "#3fcf98", "bad": "#f07a7a"},
}


def canvas(c: dict, w: float, h: float):
    fig = plt.figure(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(c["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    return fig, ax


def box(ax, c, x, y, w, h, text, edge=None, fill=None, size=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.8, edgecolor=edge or c["ink"],
                                facecolor=fill or c["bg"]))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size * SCALE,
            color=c["ink"], family=HAND, weight=weight, linespacing=1.25)


def arrow(ax, c, x1, y1, x2, y2, color=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color or c["ink"], lw=1.8,
                                mutation_scale=14, shrinkA=2, shrinkB=2))


def say(ax, c, x, y, text, size=10, color=None, ha="left", weight="normal", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size * SCALE, color=color or c["ink"],
            family=HAND, weight=weight, linespacing=1.3)


def bars(ax, c, x, y, probs, color, width=14, row=2.6):
    """Tiny hand-drawn probability bars, top choice first."""
    for i, (label, p) in enumerate(probs):
        yy = y - i * row
        say(ax, c, x, yy, label, size=8.5, ha="right")
        ax.plot([x + 0.8, x + 0.8 + max(p, 0.01) * width], [yy, yy], color=color, lw=6,
                solid_capstyle="round")
        say(ax, c, x + 1.6 + max(p, 0.01) * width, yy, f"{p:.2f}", size=8.5)


def save(fig, name, mode):
    fig.savefig(IMG / f"{name}_{mode}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def answer(router: str, alert_id: str = "a002") -> dict:
    return next(r for r in load_records(RESULTS / f"{router}_rubric.jsonl") if r["id"] == alert_id)


def concept(mode: str, c: dict) -> None:
    """Three ways to get a decision out of a model, on one real alert."""
    jev, q4 = answer("jev"), answer("qwen3.5-4b")
    top = lambda r: sorted(r["answers"]["category"]["probabilities"].items(),
                           key=lambda kv: -kv[1])[:3]
    fig, ax = canvas(c, 12, 7)
    say(ax, c, 3, 65, "Three ways to get a decision out of a model", size=17, weight="bold")
    say(ax, c, 3, 59, f"the alert: \"{jev['alert']}\"", size=9.5, color=c["ink2"])
    say(ax, c, 3, 55, "the question: what type of alert is this?", size=9.5, color=c["ink2"])

    rows = [
        (45, "1. Ask an LLM\n    to write", c["ink2"], "LLM\nwrites ~50 tokens", None),
        (25, "2. A System One\n    model (Jev)", c["jev"], "Jev\nwrites nothing", top(jev)),
        (8, "3. Read the\n    LLM's odds", c["q4"],
         "Qwen3.5-4B\n1 forward pass,\nwrites nothing", top(q4)),
    ]
    for y, label, col, model, probs in rows:
        say(ax, c, 3, y, label, size=11, color=col, weight="bold")
        box(ax, c, 27, y - 5, 13, 10, "alert +\nquestion", size=9.5)
        arrow(ax, c, 41, y, 45, y)
        box(ax, c, 46, y - 5, 20, 10, model, edge=col, size=9.5)
        arrow(ax, c, 67, y, 71, y)
        if probs is None:
            say(ax, c, 72, y + 3.5, "\"This looks like an availability\nissue, because the 5xx...\"",
                size=9, color=c["ink2"])
            say(ax, c, 72, y - 2.5, "-> parse the text -> availability", size=9)
            say(ax, c, 72, y - 6.5, "no confidence score, slow", size=9, color=c["bad"])
        else:
            say(ax, c, 72, y + 7, "probabilities over the labels", size=8.5, color=c["ink2"])
            bars(ax, c, 88, y + 3.6, probs, col, row=3.6)
    save(fig, "concept", mode)


def experiment(mode: str, c: dict) -> None:
    """What I did: one dataset, same questions, four routers, four prompts."""
    fig, ax = canvas(c, 12, 6)
    say(ax, c, 4, 56, "What I did", size=17, weight="bold")
    box(ax, c, 4, 25, 18, 16, "310 synthetic\nalerts\n\ncapacity, outages,\nsecurity, noise...",
        fill=c["note"], size=9.5)
    arrow(ax, c, 23, 33, 29, 33)
    box(ax, c, 30, 22, 22, 22, "the same 3 questions\n\n1. what type?\n2. P1, P2 or P3?\n"
        "3. page a human?", size=9.5)
    routers = [("Jev (API)", c["jev"]), ("Qwen3.8-27B (local)", c["q27"]),
               ("Qwen3.5-4B (local)", c["q4"]), ("TF-IDF classifier\n(no LLM, the floor)", c["ink2"])]
    for i, (name, col) in enumerate(routers):
        y = 44 - i * 10.5
        arrow(ax, c, 53, 33, 60, y + 3.5, color=col)
        box(ax, c, 61, y, 21, 7, name, edge=col, size=9.5)
    say(ax, c, 86, 52, "prompts tried", size=11, weight="bold")
    say(ax, c, 86, 48, "plain definitions\n+ priority rubric\n+ 2 rules from the errors\n"
        "+ one sentence moved (Qwen)", size=9.5, va="top")
    say(ax, c, 86, 27, "scored on", size=11, weight="bold")
    say(ax, c, 86, 23, "accuracy\nis the confidence honest?\nmisses at equal false pages\n"
        "did it follow the rule?", size=9.5, va="top")
    save(fig, "experiment", mode)


def labels(mode: str, c: dict) -> None:
    """Near-identical alerts, different gold labels."""
    pairs = [
        ("HDFS 91% full,\ngrowing 1.1TB a week", "P2",
         "Snapshot storage 92% of quota,\ngrowing 400GB a week", "P3"),
        ("Console login from a Tor exit\nnode, then project browsing", "P1",
         "Admin console login from a Tor\nexit node, passed MFA", "P2"),
    ]
    fig, ax = canvas(c, 12, 5.6)
    say(ax, c, 4, 51, "The labels disagree with themselves", size=17, weight="bold")
    say(ax, c, 4, 45.5, "pairs of alerts from the dataset, with the label each one carries",
        size=10, color=c["ink2"])
    for i, (a, la, b, lb) in enumerate(pairs):
        y = 27 - i * 19
        box(ax, c, 4, y, 34, 12, a, fill=c["note"], size=10)
        say(ax, c, 42, y + 6, la, size=16, weight="bold", color=c["bad"])
        say(ax, c, 52, y + 6, "vs", size=12, color=c["ink2"])
        box(ax, c, 60, y, 34, 12, b, fill=c["note"], size=10)
        say(ax, c, 98, y + 6, lb, size=16, weight="bold", color=c["bad"])
    say(ax, c, 4, 2.5, "so no router can score 100% on priority, and the real ceiling is unknown",
        size=10.5, color=c["ink2"])
    save(fig, "labels", mode)


def scorecard(mode: str, c: dict) -> None:
    """The takeaway: can a small LLM replace a dedicated decision model? Per decision."""
    rows = [
        ("What type of\nalert is it?", "yes", "Yes",
         "Same accuracy as Jev (94% vs 95%), and confident\n"
         "enough to auto-handle ~80% of alerts at 98%."),
        ("Should it page\na human?", "no", "No",
         "At the same false-page rate, Jev misses 15 real\n"
         "pages, the 4B 32. A 27B matches Jev, 8x slower."),
        ("P1, P2 or P3?", "?", "Nobody can",
         "Every model, Jev too, misses ~30% of P1s, and\n"
         "the labels disagree with themselves."),
        ("Will it follow a\nrule you add?", "no", "Not always",
         "\"Days of runway -> P3\": Jev 7 of 7, the 27B 6,\n"
         "the 4B 0. (All followed \"security -> P1\".)"),
    ]
    mark = {"yes": ("✓", c["q4"]), "no": ("✗", c["bad"]), "?": ("?", c["ink2"])}
    fig, ax = canvas(c, 12, 7.8)
    say(ax, c, 3, 73, "Can a small LLM replace a decision model?", size=17, weight="bold")
    say(ax, c, 3, 66.5, "Qwen3.5-4B on a laptop, read as a decision model, vs Jev. "
        "310 alerts, 3 decisions each.", size=9.5, color=c["ink2"])
    for x, h in ((3, "the decision"), (34, "4B replaces Jev?"), (60, "the evidence")):
        say(ax, c, x, 59, h, size=9, color=c["ink2"])
    ax.plot([3, 117], [56.5, 56.5], color=c["ink2"], lw=1, alpha=0.6)
    for i, (q, m, word, why) in enumerate(rows):
        y = 49.5 - i * 13.5
        say(ax, c, 3, y, q, size=11.5, weight="bold")
        sym, col = mark[m]
        ax.text(37, y, sym, ha="center", va="center", fontsize=30, color=col, weight="bold",
                family="DejaVu Sans")
        say(ax, c, 41.5, y, word, size=11, color=col, weight="bold")
        say(ax, c, 60, y, why, size=9.5)
        if i < len(rows) - 1:
            ax.plot([3, 117], [y - 6.8, y - 6.8], color=c["ink2"], lw=0.8, alpha=0.4)
    save(fig, "scorecard", mode)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    with plt.xkcd(scale=0.6, length=120, randomness=2):
        plt.rcParams["path.effects"] = []  # xkcd's white halo looks wrong on the chalkboard
        for mode, c in MODES.items():
            for sketch in (concept, experiment, labels, scorecard):
                sketch(mode, c)
    print(f"font: {HAND}; wrote concept/experiment/labels/scorecard x light/dark to {IMG}")
