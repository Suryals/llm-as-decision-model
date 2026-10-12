# Social copy — DRAFT, not posted

Images: LinkedIn uses the light versions, X the dark ones.

---

## LinkedIn

Image 1: `img/scorecard_light.png`, Image 2: `img/concept_light.png`, Image 3: `img/rules_light.png`

Do you need a dedicated decision model, or can a small LLM do the job?

Last month I tested Jev, a "System One" model that never writes text. You ask
it a typed question and it returns probabilities. People are now using
ordinary LLMs the same way: skip the text, read the probability the model puts
on each label. So I tried it.

What I thought:
- A small local LLM read this way could replace Jev.
- A written priority rubric would fix the hard decision.
- The leftover priority errors were a missing policy, not a weak model.

What I did:
- 310 synthetic alerts, 3 questions: what type, P1/P2/P3, page a human?
- Jev vs Qwen3.5-4B and Qwen3.8-27B on my laptop, plus a no-LLM classifier as
  a floor
- One forward pass per decision, no text generated, and Jev gets the exact
  same questions

What I found:
- What type of alert: the 4B matches Jev (94% vs 95%) and is confident enough
  to auto-handle ~80% of alerts.
- Page a human: Jev wins. At the same false-page rate it misses 15 pages, the
  4B misses 32. The 27B ties Jev but is 8x slower.
- Priority: nobody is reliable, Jev included. Every router misses ~30% of
  P1s. I wrote the missing rules down and the score didn't move. The models
  followed the rules; the labels didn't.
- Following a rule that depends on a detail in the alert ("days of runway →
  P3"): Jev 7 of 7, the 27B 6 of 7, the 4B 0 of 7. One rule, 7 alerts, but a
  clear gap.

Takeaways if you're putting an LLM in a decision path:
1. Read probabilities, not text. You get a confidence score you can act on.
2. Decide per decision. A small model was enough for one of three here.
3. Compare at the same false-alarm rate, not the default threshold.
4. Test rule-following on its own, separate from accuracy.
5. Check your labels first. Mine disagreed with themselves, so nobody's
   ceiling is 100%.

Write-up, charts and every result file: [link]

---

## X (thread)

Image: `img/scorecard_dark.png` on 1/, `img/concept_dark.png` on 2/, `img/rules_dark.png` on 4/

1/ Do you need a dedicated decision model like Jev, or can a small LLM do it?
I read Qwen3.5-4B's label probabilities directly (one forward pass, no text)
and ran it against Jev on 310 alerts.

2/ The idea: an LLM already scores every possible next token. Ask "what type
of alert is this?", read the probability on each label, renormalize. Same
answer shape as Jev, ~300ms on a laptop.

3/ Category: the 4B matches Jev (94 vs 95%). Paging: Jev wins, 15 missed
pages vs 32 at the same false-page rate. A 27B ties Jev, 8x slower.
Priority: nobody is reliable, ~30% of P1s missed by everyone.

4/ I wrote the missing priority rules down. The score didn't move. The models
followed the rules, but the labels didn't. And on a rule that depends on a
detail ("days of runway → P3"): Jev 7/7, 27B 6/7, 4B 0/7.

5/ Takeaways: read probabilities not text, decide per decision, compare at
matched false-alarm rates, test rule-following separately, and check your
labels before you compare models. [link]
