# Design — llm-as-decision-model (agreed 2026-10-12)

**Question:** do you need a dedicated "System One" decision model (TypeSafe Jev),
or can a general LLM be repurposed as one?

**Method:** a local Qwen is never asked to write text. Each decision is one
forward pass; we read the probability it puts on each label's token and
renormalise over the allowed labels. That yields the same output shape as Jev
(choice + confidence + probabilities, yes/no as a probability).

**Runs (6):** {Jev, Qwen3.8-27B Q6_K, Qwen3.5-4B Q8_0} × {plain, rubric} on the
same 310 synthetic alerts. "rubric" swaps the one-line priority definitions for
a written rubric + tie-break. The rubric text lives once in `src/schema.py`, so
Jev and Qwen get identical wording.

**Scope (option A):** only these 6 runs. No size curve, no hand re-label.
`calibration()` written by Claude at Chief's request (waived the hand-written
spine rule for this project); Chief reviews.

**Article:** chart-first, lab-notebook voice, measured claim ("on this data,
with a written rubric"). Charts: accuracy × rubric gain, calibration, error
overlap, latency. Nothing is published, pushed, or posted until Chief reviews
the draft.

Predecessor: `~/projects/jev-router-lab` (testing Jev vs Qwen/Sonnet as text
generators, published 2026-09-24).
