# Optimizer track: handoff

Author: Jida. Date: 2026-11-12. Branch: `optimizer-dev-jida`.

This is a short summary of the optimizer work so far, for anyone joining it.
Full details, with every number and confidence interval, are in `NOTES.md`.

## 1. The goal

RAPTOR can give the answer model different kinds of context:

- **leaf**: the original text chunks of the document;
- **summary**: only the summaries the tree built from those chunks;
- **collapsed**: a mix of chunks and summaries, ranked together (the RAPTOR paper's setting).

Each of these can use a budget of 400, 1000 or 2000 tokens. That makes 9 choices ("configs").

The optimizer, or "router", should pick a config **for each question**. The goal is to be
**better than always using one fixed config**, at the **same average cost** (tokens per question).

How we compare: we always compare against a fixed setup with the same average token cost.
If the router uses 1,040 tokens per question on average, we compare it with fixed configs
mixed to cost 1,040 tokens per question too. A router that wins only by using more tokens
doesn't count.

Main dataset: QASPER (questions about NLP papers). Later we also ran a small NarrativeQA test
(questions about long books and movie scripts).

## 2. What we tried and what happened

All results below use the **dev** split. The **test** split has never been used.

| What | Idea, in short | Result |
|---|---|---|
| T12: LLM profile router | An LLM labels each question (answer type, how much of the paper it needs, number of facts). Rules or a table map the label to a config. | **Did not work.** One version is 4 F1 points worse than a fixed config at the same cost. The other is about equal, but only matches a fixed config that is 2.4× cheaper. |
| E1: headroom check | If we could magically pick the best config for every question, how much would we gain? | **Big gain is possible:** about +12 F1 at 1000 tokens/question, and most of it shows up again on a rerun. But much of it is "a cheaper config gives the same answer", and 26% of questions fail under every config. |
| E2/E3: utility predictor | Train a model to predict, from question and retrieval features, whether more context will help. | **Did not work.** The predictions are basically random (AUC about 0.5). On dev the router just picked leaf@1000 almost every time. |
| Sufficiency cascade | Answer with a small context first. If the model says "not enough information", retry with a bigger one. | **Did not work.** About equal to a fixed config at the same cost. The small gain we first saw came from the new answer prompt, not from the cascade. |
| P0b: tree navigation | Walk down the tree level by level (a "beam search") instead of ranking all nodes at once. | **Did not work.** The walk misses evidence that flat ranking finds: −2.7 points of evidence found at 1000 tokens, −3.9 at 2000. |
| Adaptive-k | A published rule: sort chunks by similarity, cut at the biggest drop in similarity, add 5 extra chunks. | **Did not work.** −0.35 F1 [−1.15, +0.44] vs a fixed setup at the same cost. On QASPER the rule nearly always keeps about 6–7 chunks, so it barely adapts. |
| NarrativeQA pilot (20 stories) | Long stories have deeper trees, where RAPTOR should help most. | **RAPTOR itself did not beat plain leaf retrieval** (collapsed@2000: −2.1 ROUGE-L in one run, −1.1 in the other). But there is a large "complementarity" gain: some questions are answered better from summaries, others from the original text. A perfect picker gains about +12 points, compared with +5 from picking only the leaf budget. |
| Failure casebook | Look at actual failed questions side by side. | See section 3. Files: `scripts/failure_casebook.py`, output `outputs/diagnostics/failure_casebook/`. |

## 3. Main takeaways

1. **There is room to improve, but we can't predict it yet.** A perfect per-question choice
   would gain a lot. Nothing we tried can tell in advance which config a question needs:
   LLM labels, question words, embeddings and similarity scores all give about zero.
2. **Our router mostly fails by giving too little context.** On QASPER the router fails
   96 questions that plain RAPTOR answers. In 80 of them it chose collapsed@400, and in 69
   the model then answered "Unanswerable". So the main problem is the budget being too
   small, not picking the wrong context type.
3. **Many questions fail no matter what.** 355 of 918 QASPER dev questions fail under all
   9 configs. Routing can't fix these.
4. **On NarrativeQA, many "failures" may be scoring problems.** In about 70% of failures,
   the right answer is in the retrieved text and the model gives a reasonable-looking
   answer, but ROUGE-L (word overlap) scores it low. We have not checked this by hand yet.
5. With a strong answer model (DeepSeek), more original text usually beats summaries. The
   RAPTOR paper's gains came from a much weaker answer model (UnifiedQA-3B).

## 4. How to run things

**Setup**

```bash
git switch optimizer-dev-jida
./scripts/setup.sh                 # installs uv, Python 3.11, dependencies; creates .env
uv run pytest                      # offline tests, no API calls (184 should pass)
./scripts/download_datasets.sh --dataset qasper
```

To make API calls, put a `DEEPSEEK_API_KEY` in `.env`. Never commit `.env`.

**Free (offline) scripts.** These read saved results and make no API calls:

- `scripts/headroom_decomposition.py` (E1)
- `scripts/utility_router.py` (E2/E3)
- `scripts/adaptive_k_offline.py`
- `scripts/check_tree_navigation.py` (P0b)
- `scripts/narrativeqa_pilot_analysis.py`
- `scripts/failure_casebook.py`

They need the saved run outputs (`outputs/labels_dev/`, `outputs/labels_train/`, the
`indexes/` folder, `outputs/narrativeqa_pilot/`, ...). These are **not in git** because they are
large. Ask Jida for a zip. Making them again costs API money.

**Paid runs** (anything with `python -m adaptive_raptor.run` and a real model):

- Agree on it with the team first, and write the plan and the success rule in `NOTES.md`
  **before** you look at results.
- Run off-peak. DeepSeek charges double 01:00–04:00 and 06:00–10:00 UTC.
- Rough costs: the NarrativeQA pilot (20 stories, 2 runs) cost about $2.6. All of
  NarrativeQA dev would be about $15.

**Rules**

- **Don't use the test split** until the final evaluation.
- QASPER dev has already been looked at many times. Don't tune things on it. Decide the
  rule first, then check it once.
- Don't change the vendored RAPTOR code in `src/raptor/`.

## 5. Open directions (ideas for splitting the work)

1. **Can we predict "summary beats original text" on NarrativeQA?** The perfect-picker gain
   there is large (+12). Test offline whether any feature can predict it, using the 594
   pilot questions. This costs nothing. Only if it works should we spend about $15 on
   all of NarrativeQA dev.
2. **Fix the router's too-small budget.** For example: never pick 400 tokens, or retry
   with a bigger budget when the model says "Unanswerable". This is cheap to test offline
   with the saved answers. (Note: the cascade test above found that a plain retry roughly
   ties a fixed config, so the bar is high.)
3. **Label failures by hand.** Take 50–100 failed questions from `casebook.md` and tag each
   one: wrong text retrieved / right text but misread / answer is fine but the score is
   low / question has no good answer. This tells us what is actually worth fixing.
4. **Cost saving instead of accuracy.** Much of the headroom is "a cheaper config gives the
   same answer". A router that saves tokens without losing F1 may be easier to get than
   one that raises F1.

## 6. Where to look

- `NOTES.md`: full log, with every result and confidence interval, in date order
- `docs/OPTIMIZER_FLOW.md`: how the router pipeline works
- `docs/RAPTOR_FIDELITY.md`: check that our RAPTOR matches the paper and the original code
- `docs/DATASETS.md`: which other datasets might fit
- `src/adaptive_raptor/optimizer/`: router code; `scripts/`: all analyses
