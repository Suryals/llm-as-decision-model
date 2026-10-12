# llm-as-decision-model

Can a general LLM, read as a decision model, replace a dedicated "System One"
decision model like TypeSafe's Jev?

A local Qwen is never asked to write text. Each decision is **one forward
pass**: we read the probability it puts on each label's token and renormalise
over the allowed labels. The output has the same shape as Jev's (choice +
confidence + probabilities, yes/no as a probability), so both are scored the
same way.

**The data is synthetic.** Every alert was generated for this experiment;
the services, hostnames, IPs and companies in it are made up and don't
describe any real system.

**Setup:** 310 synthetic AIOps alerts (`data/alerts.jsonl`, generated and
labelled for the [first experiment](https://github.com/Suryals/jev-router-lab)),
three decisions per alert: category (5 classes), priority (P1/P2/P3), and
page a human (yes/no).

**Arms:** {Jev, Qwen3.8-27B Q6_K, Qwen3.5-4B Q8_0} × {plain, rubric, policy},
plus `--tiebreak-last` for the Qwen rubric runs. `rubric` swaps the one-line
priority definitions for a written rubric plus a tie-break. `policy` adds two
rules found by error analysis on these same alerts, so its score is an upper
bound, not a fair arm. All questions live once in `src/schema.py`, so Jev and Qwen get
identical wording.

## Run it

```bash
uv sync

# Jev — needs JEV_API_KEY in .env
uv run python -m src.jev --variant plain
uv run python -m src.jev --variant rubric
uv run python -m src.jev --variant policy

# Qwen — llama.cpp's llama-server (LM Studio's API returns no logprobs)
brew install llama.cpp
llama-server -m Qwen3.5-4B-Q8_0.gguf --port 8089 -ngl 99 -c 16384 -np 1 --jinja
uv run python -m src.qwen_decision --variant plain
uv run python -m src.qwen_decision --variant rubric
uv run python -m src.qwen_decision --variant rubric --tiebreak-last
uv run python -m src.qwen_decision --variant policy
# repeat with Qwen3.8-27B-Q6_K.gguf

# no-LLM floor: TF-IDF + logistic regression, 5-fold CV
uv run python -m src.baseline_tfidf

uv run python -m src.score results/*.jsonl
uv run python -m src.figs
uv run python -m src.sketches
uv run pytest
```

GGUFs: [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
(`Qwen3.5-4B-Q8_0.gguf`), [lmstudio-community/Qwen3.8-27B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF)
(`Qwen3.8-27B-Q6_K.gguf`). Measured on an M5 Max (128GB).

`src/qwen_decision.py` checks at startup that the served model renders the
expected chat template and that every label is a single token. It refuses to
run rather than silently mis-score.

## Layout

| file | what |
|---|---|
| `src/schema.py` | the three decisions, plain definitions, priority rubric |
| `src/jev.py` | Jev runner |
| `src/qwen_decision.py` | Qwen-as-decision-model runner (llama-server) |
| `src/baseline_tfidf.py` | no-LLM baseline, cross-validated |
| `src/score.py` | accuracy, calibration (ECE), error overlap, matched-rate paging, latency |
| `src/figs.py` | article data charts, light + dark |
| `src/sketches.py` | whiteboard explainer diagrams, light + dark |
| `tests/test_score.py` | hand-computed cases for the scorer |
| `results/` | one file per arm; every number in the write-up comes from these |

## Caveats

- The alerts are synthetic. The priority labels aren't fully consistent
  (similar alerts carry different priorities), so no router can reach 100%
  on priority.
- One dataset, one domain. "Matches Jev" here means on these 310 alerts.
- Latency for Qwen is local Apple Silicon. The 27B slows under sustained load
  (GPU throttling), the 4B does not. Jev latency includes the network round trip.

## License

MIT, see `LICENSE`. The write-up and analysis code were written by Claude
(Anthropic) at the author's direction.
