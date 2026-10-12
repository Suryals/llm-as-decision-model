"""Qwen repurposed as a decision model: no text generation.

Each decision is one forward pass. We read the probability the model puts on
each label's token and renormalise over the allowed labels, which gives the
same output shape as Jev (choice + confidence + probabilities; yes/no as a
probability).

Backend: llama-server (llama.cpp). LM Studio's API returns `logprobs: null`,
llama-server returns them. Start it on the GGUF you want to test:

    llama-server -m <model>.gguf --port 8089 -ngl 99 -c 16384 -np 1 --jinja

Usage:  uv run python -m src.qwen_decision --variant plain|rubric|policy [--tiebreak-last] [--limit N]

Writes results/<model>_<variant>.jsonl, model named from the served GGUF.
"""

import argparse
import json
import math
import time
from pathlib import Path

import httpx

from src.dataset import load_alerts, results_path
from src.schema import PRIORITY_TIEBREAK, VARIANTS, questions

SERVER = "http://localhost:8089"
N_PROBS = 50  # top-k logprobs to read; a label outside the top 50 counts as 0

SYSTEM_PROMPT = """You are an L1 triage decision model for a big-data platform ops team.
You read one alert and answer one question about it.
Reply with only the answer label — no explanation, no punctuation."""

# Qwen chat template, thinking disabled. System + alert come first so the KV
# cache prefix is shared across the three questions for one alert.
# check_server() verifies the served model renders exactly this.
TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\nAlert: {alert}\n\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n{prefix}"
)

# Token text -> label it votes for. P1/P2/P3 tokenize as "P" + digit, so the
# priority prompt forces the "P" and we read the digit.
YES_NO = {"yes": "yes", "Yes": "yes", "no": "no", "No": "no"}


def render(decision: dict, tiebreak_last: bool = False) -> tuple[str, str, dict[str, str]]:
    """Schema question -> (question text, forced prefix, token -> label).

    Default keeps Jev's order: instructions (incl. any tie-break), then the
    labels. tiebreak_last moves the rubric tie-break after the label list,
    a prompt-sensitivity check only (Jev's API has no equivalent)."""
    if decision["type"] == "noul":
        return decision["instructions"] + " Answer yes or no.", "", YES_NO
    bullets = "\n".join(f"- {k}: {v}" for k, v in decision["criteria"].items())
    instructions = decision["instructions"]
    if tiebreak_last and PRIORITY_TIEBREAK in instructions:
        instructions = instructions.replace(f" {PRIORITY_TIEBREAK}", "")
        bullets += f"\n{PRIORITY_TIEBREAK}"
    text = f"{instructions}\nAnswer with one of:\n{bullets}"
    labels = list(decision["criteria"])
    if all(k[0] == "P" and k[1:].isdigit() for k in labels):
        return text, "P", {k[1:]: k for k in labels}
    return text, "", {k: k for k in labels}


def check_server(client: httpx.Client, qs: dict) -> str:
    """Fail fast if the served model's template or tokenizer would silently
    break the scoring. Returns the model name."""
    props = client.get(f"{SERVER}/props").raise_for_status().json()
    rendered = client.post(f"{SERVER}/apply-template", json={
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }).raise_for_status().json()["prompt"]
    expected = TEMPLATE.format(system="S", alert="", question="", prefix="").replace(
        "Alert: \n\n", "U")
    if rendered != expected:
        raise SystemExit(f"chat template mismatch:\n{rendered!r}\nvs\n{expected!r}")
    for name, decision in qs.items():
        for token_text in render(decision)[2]:
            pieces = client.post(f"{SERVER}/tokenize", json={"content": token_text}) \
                .raise_for_status().json()["tokens"]
            if len(pieces) != 1:
                raise SystemExit(f"{name}: {token_text!r} is {len(pieces)} tokens")
    return Path(props["model_path"]).stem


def label_probs(client: httpx.Client, alert: str, question: str, prefix: str,
                tokens: dict[str, str]) -> tuple[dict[str, float], float, int]:
    """One forward pass, one token. Returns (label -> prob renormalised over
    allowed labels, raw mass on allowed labels, prompt tokens evaluated)."""
    resp = client.post(f"{SERVER}/completion", json={
        "prompt": TEMPLATE.format(system=SYSTEM_PROMPT, alert=alert,
                                  question=question, prefix=prefix),
        "n_predict": 1,
        "n_probs": N_PROBS,
        "temperature": 0,
        "cache_prompt": True,
    }, timeout=120.0)
    resp.raise_for_status()
    body = resp.json()
    mass: dict[str, float] = {}
    for t in body["completion_probabilities"][0]["top_logprobs"]:
        label = tokens.get(t["token"])
        if label is not None:
            mass[label] = mass.get(label, 0.0) + math.exp(t["logprob"])
    covered = sum(mass.values())
    labels = sorted(set(tokens.values()))
    if covered == 0:  # model ignored the format entirely
        probs = {k: 1 / len(labels) for k in labels}
    else:
        probs = {k: mass.get(k, 0.0) / covered for k in labels}
    return probs, covered, body.get("tokens_evaluated", 0)


def decide(client: httpx.Client, alert: str, rendered: dict) -> dict:
    started = time.perf_counter()
    answers, label_mass, prompt_tokens = {}, {}, 0
    for name, (question, prefix, tokens) in rendered.items():
        probs, covered, n_tok = label_probs(client, alert, question, prefix, tokens)
        label_mass[name] = round(covered, 4)
        prompt_tokens += n_tok
        if set(probs) == {"yes", "no"}:
            answers[name] = {"type": "noul", "noul": round(probs["yes"], 4)}
        else:
            best = max(probs, key=lambda k: probs[k])
            answers[name] = {
                "type": "choice", "choice": best,
                "confidence": round(probs[best], 4),
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
            }
    return {
        "answers": answers,
        "label_mass": label_mass,  # how much raw probability landed on valid labels
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
    }


def run(variant: str, limit: int | None, tiebreak_last: bool = False) -> None:
    qs = questions(variant)
    rendered = {name: render(d, tiebreak_last) for name, d in qs.items()}
    with httpx.Client() as client:
        model = check_server(client, qs)
        router = model.split("-Q")[0].lower()  # Qwen3.5-4B-Q8_0 -> qwen3.5-4b
        suffix = "-tiebreak-last" if tiebreak_last else ""
        out_path = results_path(router, variant + suffix)
        decide(client, "warm-up: test alert, ignore", rendered)
        with out_path.open("w") as out:
            for i, case in enumerate(load_alerts(limit), 1):
                result = decide(client, case["alert"], rendered)
                out.write(json.dumps({
                    "id": case["id"], "alert": case["alert"], "gold": case["gold"],
                    "router": router, "model": model, "variant": variant + suffix,
                    **result,
                }) + "\n")
                if i % 50 == 0:
                    print(f"{i} done, last {result['latency_ms']:.0f}ms")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tiebreak-last", action="store_true",
                        help="prompt-sensitivity check: put the rubric tie-break after the labels")
    args = parser.parse_args()
    if args.tiebreak_last and args.variant == "plain":
        parser.error("--tiebreak-last needs --variant rubric or policy")
    run(args.variant, args.limit, args.tiebreak_last)
