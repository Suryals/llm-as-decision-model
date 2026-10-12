"""Jev (TypeSafe "System One" API) runner.

Usage:  uv run python -m src.jev --variant plain|rubric [--limit N]

Needs JEV_API_KEY in .env. One request per alert answers all three
decisions. Writes results/jev_<variant>.jsonl.

API notes (jev-1.13.0): POST /v1/systemone, Bearer auth; `choice` questions
take `criteria` (not `options`); `noul` answers are P(yes).
"""

import argparse
import json
import os
import time

import httpx
from dotenv import load_dotenv

from src.dataset import load_alerts, results_path
from src.schema import VARIANTS, questions

JEV_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


def run(variant: str, limit: int | None) -> None:
    load_dotenv()
    key = os.environ["JEV_API_KEY"]
    qs = questions(variant)
    out_path = results_path("jev", variant)

    # One persistent connection: a fresh TLS handshake to the US endpoint
    # costs ~500-800ms from here, more than the decision itself.
    with httpx.Client(timeout=30.0) as client, out_path.open("w") as out:
        for i, case in enumerate(load_alerts(limit), 1):
            started = time.perf_counter()
            resp = client.post(
                JEV_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": MODEL, "state": {"alert": case["alert"]}, "questions": qs},
            )
            resp.raise_for_status()
            body = resp.json()
            out.write(json.dumps({
                "id": case["id"], "alert": case["alert"], "gold": case["gold"],
                "router": "jev", "model": body["model"], "variant": variant,
                "answers": body["answers"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "usage": body["usage"],
            }) + "\n")
            if i % 50 == 0:
                print(f"{i} done")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.variant, args.limit)
