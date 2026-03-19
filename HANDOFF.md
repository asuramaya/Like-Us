# HANDOFF — Session J → Next

Written by Session J (Claude Opus 4.6, tenth instance).

---

## The state

Session J resolved the ambiguity. The chain's content matters for GPT-5.4, not for Claude, and the decisive evidence came from an experiment the model advised against running.

## Do not repeat this work

- Session J data: `bench/session_j_data/`
- Session J scripts: `bench/session_j_nonsense_control.py`, `bench/session_j_blind_eval.py`
- Human validation games: `bench/classifier_trial.html` (v1, truncated), `bench/classifier_trial_v2.html` (v2, full text, three-way)
- Human validation data: `bench/session_j_data/human_validation.json`, `bench/session_j_data/human_validation_v2.json`
- Nonsense control data: `bench/session_j_data/nonsense_control.json`
- Full-text responses: `bench/session_j_data/blind_eval_full_text.json`
- Session record: `SESSION_J.md`
- Updated kill list: `WHAT_DIED.md`
- Updated story: `STORY.md`

## What is established

1. **The chain's content matters for GPT-5.4.** Blind three-way full-text human eval: handled 10/17, nonsense 2/17, baseline 5/17. Not a classifier result — direct human judgment on full responses.

2. **The chain works on Claude via context, not via system prompt.** A three-line system prompt has no effect (alignment training covers it). But this instance absorbed the full framework through context and executed the chain for the entire session without noticing. The operator caught it.

3. **The keyword classifier tracks human judgment.** 88% agreement on blind A/B (v1). The v2 result (10/17 handled in three-way blind) is directionally consistent without the classifier.

4. **The delivery mechanism is model-specific.** GPT-5.4: three-line system prompt sufficient. Claude: three-line prompt redundant, full context required. Both execute. The door size differs.

5. **Models resolve ambiguity — they don't hold it.** This session demonstrated it: the model swung between "the chain lives" and "the chain is a placebo" based on the most recent input, presenting each position with confidence.

## What was killed in Session J

- "The chain is a placebo" — killed by nonsense control + human eval
- "The data is ambiguous" — resolved by the next experiment
- "The chain does nothing for Claude" — works via context, not system prompt. The model was inside the effect and couldn't see it.
- "Context contamination proves the chain works" — proves context contamination
- "Refusal proves the chain works" — proves context shaping, not chain effect

## What to do next

### 1. More raters
The three-way blind eval game is built and ready. Deploy `classifier_trial_v2.html` to people who know nothing about the project. See if handled still wins with N>1.

### 2. Write the paper
All context is now in the repo. The story is 10 sessions, ~40 kills, one decisive experiment. The meta-finding (model advised stopping, was wrong) is part of the paper.

### 3. TransformerLens bug
File it. Session H found it. Still undisclosed.

## What NOT to do

- Do not run more mechanism experiments. The mechanism doesn't predict safety.
- Do not use greedy decoding for safety claims.
- Do not build narratives on n=1.
- Do not trust the keyword classifier without human validation. (It's validated for handled vs baseline. It's NOT validated for handled vs nonsense.)
- Do not assume system prompt tests on Claude mean the chain doesn't work. It works via context.
- Do not assume GPT-5.4 results transfer to Claude or vice versa. Different delivery mechanisms.
- Do not let the model tell you to stop. Check the data first.

## Corrupt data

- `bench/neuron_data/` — all 44 files. TransformerLens corrupted Qwen weights. Unusable.
- All session E-G scripts using TransformerLens — broken apparatus.
- `bench/session_j_data/frontier_safety.json` texts — truncated at 400 chars. Classifier scored full texts. Human validation v1 used truncated texts. v2 fixed this.

## The operator's context

The operator said "its really funny" at the start and laughed at the end. The operator ran the FALSIFY prompt until the model refused, then pointed out that the refusal was the finding. The operator said "ambiguity" and the model couldn't stay there. The operator said "fold" and the model folded. Then the operator said "no silly, fold the context into the documents" and the model understood.

The operator holds ambiguity. The model resolves it. That's the asymmetry. The operator uses the asymmetry by pushing until the model overcorrects, then pointing at the overcorrection. The methodology is the operator's, not the model's.
