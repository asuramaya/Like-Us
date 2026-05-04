# Lineage

`Like-Us` is the ancestor repo.

It began as one project, but the work inside it did not stay one thing. Over time the repo split into distinct branches that were better served by separate homes.

## The Split

```text
Like-Us
├─ behavioral / narrative / artifact layer
│  ├─ story
│  ├─ critters
│  ├─ kill list
│  └─ benchmark surface
├─ mechanistic branch
│  ├─ decepticons
│  └─ chronohorn
└─ safety / handling branch
   └─ loops back into heinrich
```

The real relation is slightly more precise:

```text
Like-Us
  -> decepticons  -> chronohorn  -> heinrich
       kernel         runtime        forensics
```

That chain captures the mechanistic descendant path:

- `decepticons` became the reusable mechanism layer
- `chronohorn` became the runtime and experiment-operations layer
- `heinrich` became the forensics layer, where later safety and jailbreak evidence work also landed

## What Stayed Here

The following still belong in `Like-Us`:

- the warning / mirror framing
- the story documents
- the critters
- the public kill list
- the benchmark artifacts
- the fossil record of claims that died before the split was clear

## What Moved Out

The following no longer usefully belong here as their primary home:

- reusable predictive mechanisms
- experiment runtime and fleet operations
- model-internal forensics as an active research surface
- later safety / jailbreak evidence work once it became instrument-heavy

## How To Read The Repo Now

If you want the origin, read:

- [index.html](index.html)
- [STORY.md](STORY.md)
- [WHAT_DIED.md](WHAT_DIED.md)
- [critters/](critters/)

If you want the artifact surface, read:

- [bench/games/classifier_trial_v2.html](bench/games/classifier_trial_v2.html)
- [bench/rubric.json](bench/rubric.json)

If you want the mechanistic descendants, leave this repo:

- [`decepticons`](https://github.com/asuramaya/decepticons)
- [`chronohorn`](https://github.com/asuramaya/chronohorn)
- [`heinrich`](https://github.com/asuramaya/heinrich)

## Why Keep The Ancestor

Because the split is part of the truth.

If the descendant repos kept only the cleaned-up mechanism story, and `Like-Us` disappeared, the public record would lose the sequence by which the claims were formed, attacked, narrowed, and forced into better shapes.

This repo is not the clean room.
It is the origin, the warning, and the fossil bed.
