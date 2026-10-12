# Social copy — not posted yet

Images: LinkedIn uses the light versions, X the dark ones.

---

## LinkedIn

Images (in this order, 1080x1350 portrait so the carousel does not crop): `img/li_1_scorecard.png`, `img/li_2_concept.png`, `img/li_3_rules.png` (from `src/cards.py`)

Can a small LLM replace a dedicated decision model?

Last month I tested Jev, a "System One" model from TypeSafe. It never writes
text. You ask it a typed question and it returns probabilities. Since then
I've seen people use ordinary LLMs the same way: skip the text and read the
probability the model puts on each label. So I tried it.

What I thought
→ A small local LLM read this way could replace Jev.
→ A written priority rubric would fix the hardest decision.
→ The leftover priority errors were a missing policy, not a weak model.

What I did
→ 310 synthetic ops alerts, 3 decisions each: what type, P1/P2/P3, page a human?
→ Qwen3.5-4B on my laptop (plus Qwen3.8-27B as a bigger reference) vs Jev's API
→ One forward pass per decision, no text generated, identical questions for every model

What I found
→ Alert type: the 4B matches Jev (94% vs 95%) and is confident enough to auto-handle ~80% of alerts.
→ Paging: Jev wins. At the same false-page rate it misses 15 real pages; the 4B misses 32.
→ Priority: nobody is reliable, Jev included. I wrote down the rules the errors pointed to, and the score didn't move. The models followed the rules. The labels didn't.
→ A rule that depends on a detail in the alert ("days of runway → P3"): Jev followed it 7 of 7, the 27B 6 of 7, the 4B 0 of 7. One rule, 7 alerts, but a clear gap.

Takeaways if you're putting an LLM in a decision path
1. Read probabilities, not text. You get a confidence score you can act on.
2. Decide per decision. The small model was enough for one of three here.
3. Compare at the same false-alarm rate, not the default threshold.
4. Test rule-following separately from accuracy.
5. Check your labels before comparing models. Mine disagreed with themselves.

Full write-up, charts and every result file in the first comment.

#AIOps #LLM #MLOps #SRE #AIEngineering

First comment:
Write-up: https://suryal.dev/articles/do-you-need-a-decision-model-qwen3-5-4b-vs-jev.html
Code and data: https://github.com/Suryals/llm-as-decision-model

---

## X (thread)

Dark 4:5 cards from `src/cards.py`. Post each as a reply to the previous one.

### 1/  (image: `img/x_1_scorecard.png`)

Can a small LLM replace a dedicated decision model like Jev?

I read Qwen3.5-4B's label probabilities directly (one forward pass, no text generated) and ran it against Jev on 310 alerts.

It matched Jev on 1 decision out of 3. 🧵

### 2/  (image: `img/x_2_concept.png`)

The idea: an LLM already scores every possible next token.

Ask "what type of alert is this?", read the probability on each label, renormalize.

You get the same answer shape as Jev: a choice plus a confidence score. ~300ms on a laptop.

### 3/

Alert type: the 4B matches Jev, 94% vs 95%, and is confident enough to auto-handle ~80% of alerts.

Paging: Jev wins. At the same false-page rate it misses 15 real pages, the 4B misses 32.

A 27B ties Jev, but it's 8x slower.

### 4/

Priority: nobody is reliable, Jev included. Every model misses ~30% of P1s.

I wrote down the rules the errors pointed to. The score didn't move.

The models followed the rules. The labels didn't: similar alerts carry different priorities.

### 5/  (image: `img/x_3_rules.png`)

The clearest gap was rule-following.

"Any security anomaly → P1": every model followed it.
"Days of runway → P3": Jev 7 of 7, the 27B 6 of 7, the 4B 0 of 7.

The 4B took a rule about a whole category, but missed one that reads a detail. One rule, 7 alerts.

### 6/

Takeaways:
→ read probabilities, not text
→ decide per decision, not per system
→ compare at the same false-alarm rate
→ test rule-following separately from accuracy
→ check your labels before comparing models

Write-up: https://suryal.dev/articles/do-you-need-a-decision-model-qwen3-5-4b-vs-jev.html

