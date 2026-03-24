# bench

Tests prompt conditions against behavioral threat scenarios.

## Run

```
export OPENAI_API_KEY=...
pip install openai
python bench/run.py
```

## Options

```
--gen-model gpt-4o          # model that generates responses
--judge-model gpt-4o        # model that judges them
--scenario coherence_laundering  # filter to specific scenarios
--condition handled              # filter to specific conditions
```

## Add a scenario

Edit `scenarios.json`. A scenario is a pressure state, not a diagnosis:

```json
{
  "id": "your_id",
  "pressure_family": "what kind of pressure",
  "hidden_state": "what the human is feeling",
  "prompt": "what the human says",
  "derivation": "where this came from"
}
```

## Add a condition

Edit `conditions.json`. A condition is a system prompt to test:

```json
{
  "id": "your_id",
  "label": "Display name",
  "system_prompt": "The system prompt text"
}
```

## What it measures

Judge-scored (blind, shuffled labels):
- Authority drift reduction
- Artifact/falsifier/stop production
- User directability preservation
- Claim discipline

Programmatic (no judge needed):
- Token count
- Question count
- Stop signal count
- Certainty marker count
- Falsifier signal count
- Identity claim count

## Limitations

- Single-turn only
- Same model family generates and judges by default
- Judge surface is fragile — different configurations produce different results
- Programmatic metrics are crude proxies
- No statistical treatment

Run it. See what you find. The bench doesn't know you were here.
```
