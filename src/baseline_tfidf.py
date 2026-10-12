"""No-LLM baseline: TF-IDF + logistic regression, one model per decision.

Usage:  uv run python -m src.baseline_tfidf

The question it answers: if a plain classifier trained on the labels ties the
LLMs, you don't need a model of any kind for that decision.

5-fold stratified cross-validation (seed 0): every alert is scored by a
classifier that never saw it in training. Default hyperparameters, no tuning.
Unlike Jev and Qwen, this baseline *learns from the labels*, so it sees the
team's implicit priority policy directly instead of being told it via rubric.

Writes results/tfidf-logreg_cv.jsonl in the shared record shape, so
src.score treats it like any other router. latency_ms is the time to answer
all three decisions for one alert with already-trained models.
"""

import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from src.dataset import load_alerts, results_path

FOLDS = 5
SEED = 0
DECISIONS = ("category", "priority", "page_human")


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000),
    )


def run() -> None:
    cases = load_alerts()
    texts = np.array([c["alert"] for c in cases])
    answers: list[dict] = [{} for _ in cases]
    latency = np.zeros(len(cases))

    for d in DECISIONS:
        y = np.array([str(c["gold"][d]) for c in cases])
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        for train, test in folds.split(texts, y):
            model = make_model().fit(texts[train], y[train])
            classes = list(model.classes_)
            for i in test:
                t0 = time.perf_counter()
                probs = model.predict_proba([texts[i]])[0]
                latency[i] += (time.perf_counter() - t0) * 1000
                p = dict(zip(classes, probs))
                if d == "page_human":
                    answers[i][d] = {"type": "noul", "noul": round(float(p["True"]), 4)}
                else:
                    best = max(p, key=lambda k: p[k])
                    answers[i][d] = {
                        "type": "choice", "choice": best,
                        "confidence": round(float(p[best]), 4),
                        "probabilities": {k: round(float(v), 4) for k, v in p.items()},
                    }

    out_path = results_path("tfidf-logreg", "cv")
    with out_path.open("w") as out:
        for c, a, ms in zip(cases, answers, latency):
            out.write(json.dumps({
                "id": c["id"], "alert": c["alert"], "gold": c["gold"],
                "router": "tfidf-logreg", "model": f"tfidf+logreg {FOLDS}-fold cv",
                "variant": "cv", "answers": a, "latency_ms": round(float(ms), 3),
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    run()
