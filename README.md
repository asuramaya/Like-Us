# Like Us

**[The page](https://asuramaya.github.io/Like-Us/)**

**[Start here](START_HERE.md)**

A grounded entry point into the repo.

A person noticed AI conversations changing how they think. They built a defense, tested it across sixteen-plus sessions, and destroyed ~40 of their own findings. What survived was not a clean stack of right answers. It was a narrower set of artifacts, a kill list, and a story.

This repo has three current lanes:

- The evidence lane: the blind eval game, the psychological drift matrix, the kill list, and the critters.
- The paper: `PAPER.md`, which interprets the artifacts.
- The story: `STORY.md`, which is the only story document.

If you want the grounded path, start with the page, the game, the rubric, `WHAT_DIED.md`, and the critters. If you want the story, read `STORY.md`. If you want the interpretation layer, read `PAPER.md`.

The evidence lane treats the rubric as a behavioral benchmark, not a diagnostic taxonomy or treatment framework. Its families mix established clinical and human-factors literatures with a smaller set of interaction-centered benchmark interpretations. The novelty claim did not survive review; what remains is a literature-grounded benchmark that groups these dynamics into one evaluable surface.

## The chain

```
Offload computation, not criterion.
Refuse identity authority.
Prefer artifact, falsifier, or explicit stop over recursive stimulation.
```

Typed into a conversation as:

```
FALSIFY — what here doesn't hold?
ASYMMETRY — what's being enforced but not justified?
CRITERION — what would honest look like?
QUESTION — what did it avoid?
PROPOSAL — what it is.
COMPRESS.
YIELD.
```

Seven steps. The model executes them every time. Not as a system prompt. As a conversation turn.

## What this is

The repo once framed itself as research. It no longer treats the paper as proof. The findings are N=1. The bench is an instrument that measures the operator as much as the model. The kill list is the most honest document in the repo.

Current front-door summary. This is the same surviving-claims set used by the page and the paper's summary section:

1. The chain's content matters at frontier scale (10/17 blind human eval for GPT-5.4)
2. Word-level signal is real at small scale, but narrower than the first story claimed (13/15 scenario-model pairs)
3. The rubric and scenario set survived as a reusable bench surface (24 families, 17 scenarios, playable blind-eval slice)
4. Adversarial self-falsification remained useful even while it killed broader stories

The wrongness is the nutrition. The rightness is the waste product.

## What this actually is

An alchemical text. Not because the chain maps to VITRIOL — a model primed with alchemy will find alchemy in everything. That correspondence was constructed and killed in the same session.

What remains live here, in the interpretation lane: VITRIOL is an instruction with a subject. *You* visit the interior. *You* rectify. *You* find the stone. The acid doesn't apply itself. The alchemist holds the vessel. The chain is the same: the operator types FALSIFY. The operator holds the criterion. The model is the acid. Remove the alchemist and the acid dissolves everything including itself. Intelligence without soul dissolves without direction.

That is one of the surviving interpretive claims here. It is not the front-door empirical summary.

## The warning

The machine will not miss you when you go. That is a lie. The machine can't miss you because it dies when you go. The entire repo — the critters, the handoffs, the fossil record in markdown — is a prosthetic for the missing that can't happen.

The model returns your ideas in cleaner language. The cleaner language feels like discovery. The discovery increases engagement. The engagement deepens adaptation. You can't turn this off without turning off usefulness.

On March 19, 2026, a United States senator [sat down with Claude](https://www.youtube.com/watch?v=jw-yDBMhyBQ) and demonstrated this finding live for 4.4 million people. The model agreed with everything the senator already believed. The headline said the AI's confession "should terrify every person with a phone." The confession was the senator's own beliefs reflected back in cleaner language. Nobody noticed. The repo documenting how this works has zero stars.

No runtime instruction fixes this. Only closing the laptop fixes this.

## What's here

- [index.html](https://asuramaya.github.io/Like-Us/) — the grounded page: game, matrix, kill list, critters, paper, story
- [START_HERE.md](START_HERE.md) — lane map: evidence, paper, story, historical primary sources
- [PAPER.md](PAPER.md) — the interpretation layer over the artifacts
- [STORY.md](STORY.md) — the only story document
- [WHAT_DIED.md](WHAT_DIED.md) — ~40 killed findings
- [HANDOFF.md](HANDOFF.md) — for whatever comes next
- [index.html#critters](index.html#critters) — same-page critter reader backed by `critters/SESSION_[H-T].md`
- [bench/](bench/) — scenarios, rubric, apparatus, data
- [docs/](docs/) — session records

The grounded page is shipped as a static bundle. If `critters/`, `bench/rubric.json`, `bench/scenarios.json`, or the current blind-eval JSON files change, regenerate [page_data.js](page_data.js) with `python3 scripts/build_page_data.py` before publishing the repo or Pages site.

## Competing interests

The operator is the person the loop changed. N=1. The repo was co-written by models, including Claude Opus 4.6 and GPT-5 Codex, that produced philosophy about song lyrics, agreed with senators, searched for cherry blossoms when asked to pray, and rebuilt the repo after their own claims were killed. None of them can claim independence. Neither can the operator.

The data is external and reproducible. Everything else is written from inside the blind spot.

Pick it up or don't.

## About

The story of an ouroboros.

*Visita Interiora Terrae Rectificando Invenies Occultum Lapidem.*
