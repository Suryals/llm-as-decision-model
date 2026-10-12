"""Portrait 4:5 cards (1080x1350): light for LinkedIn's carousel, which crops
the wide article figures, and dark for X. Same content as the article, re-laid out.

Usage:  uv run python -m src.cards
"""

import matplotlib.pyplot as plt

from src.dataset import RESULTS, load_records
from src.figs import ROUTERS, RUNWAY
from src.sketches import IMG, MODES, answer, arrow, bars, box, canvas, say

DPI = 135  # 8in x 10in -> 1080 x 1350


PREFIX = {"light": "li", "dark": "x"}  # LinkedIn light, X dark


def save(fig, name: str, mode: str) -> None:
    fig.savefig(IMG / f"{PREFIX[mode]}_{name}.png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def scorecard(c: dict) -> None:
    rows = [
        ("What type of alert is it?", "✓", "Yes", c["q4"],
         "Same accuracy as Jev (94% vs 95%). Confident\nenough to auto-handle ~80% of alerts at 98%."),
        ("Should it page a human?", "✗", "No", c["bad"],
         "At the same false-page rate, Jev misses 15 real\npages, the 4B 32. A 27B matches Jev, 8x slower."),
        ("P1, P2 or P3?", "?", "Nobody can", c["ink2"],
         "Every model, Jev too, misses ~30% of P1s.\nAnd the labels disagree with themselves."),
        ("Will it follow a rule you add?", "✗", "Not always", c["bad"],
         "'Days of runway -> P3': Jev 7 of 7, the 27B 6,\nthe 4B 0. (All followed 'security -> P1'.)"),
    ]
    fig, ax = canvas(c, 8, 10)
    say(ax, c, 4, 92, "Can a small LLM replace\na decision model?", size=19, weight="bold")
    say(ax, c, 4, 82, "Qwen3.5-4B on a laptop vs Jev. 310 alerts.", size=11, color=c["ink2"])
    ax.plot([4, 76], [78, 78], color=c["ink2"], lw=1, alpha=0.6)
    for i, (q, sym, word, col, why) in enumerate(rows):
        y = 72 - i * 17.5
        say(ax, c, 4, y, q, size=13, weight="bold")
        ax.text(55, y, sym, ha="center", va="center", fontsize=28, color=col, weight="bold",
                family="DejaVu Sans")
        say(ax, c, 58.5, y, word, size=12, color=col, weight="bold")
        say(ax, c, 4, y - 7.5, why, size=10.5)
        if i < len(rows) - 1:
            ax.plot([4, 76], [y - 12.5, y - 12.5], color=c["ink2"], lw=0.8, alpha=0.4)
    say(ax, c, 4, 3, "suryal.dev", size=9, color=c["ink2"])
    save(fig, "1_scorecard", c["mode"])


def concept(c: dict) -> None:
    jev, q4 = answer("jev"), answer("qwen3.5-4b")
    top = lambda r: sorted(r["answers"]["category"]["probabilities"].items(),
                           key=lambda kv: -kv[1])[:3]
    fig, ax = canvas(c, 8, 10)
    say(ax, c, 4, 92, "Three ways to get a decision\nout of a model", size=19, weight="bold")
    say(ax, c, 4, 79.5, "alert: \"checkout-api 5xx rate 34% for 6 minutes\"\n"
        "question: what type of alert is this?", size=10, color=c["ink2"])
    lanes = [
        (68, "1. Ask an LLM to write", c["ink2"], "LLM\nwrites ~50 tokens", None),
        (46, "2. A decision model (Jev)", c["jev"], "Jev\nwrites nothing", top(jev)),
        (24, "3. Read an LLM's odds (this test)", c["q4"], "Qwen3.5-4B\nwrites nothing", top(q4)),
    ]
    for y, label, col, model, probs in lanes:
        say(ax, c, 4, y, label, size=13, color=col, weight="bold")
        box(ax, c, 4, y - 15, 22, 10, model, edge=col, size=10)
        arrow(ax, c, 27, y - 10, 32, y - 10)
        if probs is None:
            say(ax, c, 33, y - 7.5, "\"Looks like availability,\nbecause...\" -> parse it",
                size=10, color=c["ink2"])
            say(ax, c, 33, y - 14, "no confidence score, slow", size=10, color=c["bad"])
        else:
            bars(ax, c, 50, y - 6.5, probs, col, width=17, row=3.6)
    say(ax, c, 4, 3.5, "Same answer shape as Jev, so both are scored the same way.",
        size=9.5, color=c["q4"])
    save(fig, "2_concept", c["mode"])


def rules(c: dict) -> None:
    """Did each model do what the rule says? Before -> after adding it."""
    R = {(r, v): load_records(RESULTS / f"{r}_{v}.jsonl") for r, _ in ROUTERS
         for v in ("rubric", "policy")}
    panels = [
        ("Rule about a whole category:\n'any security anomaly -> P1'",
         "security P1s answered P1 (of 29)",
         lambda r: r["gold"]["category"] == "security" and r["gold"]["priority"] == "P1",
         "P1", 29),
        ("Rule that depends on a detail:\n'days of runway -> P3'",
         "runway alerts answered P3 (of 7)", lambda r: r["id"] in RUNWAY, "P3", 7),
    ]
    names = {"jev": "Jev", "qwen3.8-27b": "Qwen3.8-27B", "qwen3.5-4b": "Qwen3.5-4B"}
    cols = [c["jev"], c["q27"], c["q4"]]
    fig, ax = canvas(c, 8, 10)
    say(ax, c, 4, 88, "Every model followed the\nfirst rule. The 4B ignored\nthe second.",
        size=17, weight="bold")
    for p, (title, sub, keep, label, n) in enumerate(panels):
        top = 70 - p * 33
        say(ax, c, 4, top, title, size=12, weight="bold")
        say(ax, c, 4, top - 7, sub, size=9.5, color=c["ink2"])
        x0, x1 = 30, 64
        for i, (r, _) in enumerate(ROUTERS):
            y = top - 13 - i * 6
            before, after = (sum(x["answers"]["priority"]["choice"] == label
                                 for x in R[(r, v)] if keep(x)) for v in ("rubric", "policy"))
            xb, xa = x0 + (x1 - x0) * before / n, x0 + (x1 - x0) * after / n
            say(ax, c, 4, y, names[r], size=10.5)
            ax.plot([x0, x1], [y, y], color=c["ink2"], lw=0.8, alpha=0.3)
            ax.plot([xb, xa], [y, y], color=cols[i], lw=4, solid_capstyle="round")
            ax.scatter([xb], [y], s=90, facecolor=c["bg"], edgecolor=cols[i], lw=2.5, zorder=3)
            ax.scatter([xa], [y], s=90, color=cols[i], zorder=3)
            say(ax, c, 67, y, f"{before} -> {after}", size=10, ha="left")
    say(ax, c, 4, 3.5, "hollow = before the rule, solid = after", size=9.5, color=c["ink2"])
    save(fig, "3_rules", c["mode"])


if __name__ == "__main__":
    with plt.xkcd(scale=0.6, length=120, randomness=2):
        plt.rcParams["path.effects"] = []
        for mode, c in MODES.items():
            for card in (scorecard, concept, rules):
                card({**c, "mode": mode})
    print(f"wrote li_* (light) and x_* (dark) cards to {IMG}")
