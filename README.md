# Catalog Conversation Router

`catalog_conversation_router` is a Home Assistant custom integration that provides a custom conversation agent with local-first routing, catalog-grounded fuzzy correction, optional LLM translation, and final LLM fallback.

## What It Does

For each utterance:

1. Tries the selected **local conversation agent** directly.
2. If that fails, tries **catalog-grounded fuzzy correction** and retries local.
3. If that fails, asks the selected **LLM agent** to translate into one canonical phrase and retries local.
4. If that fails, optionally falls back to direct **LLM handling**.

The integration keeps a live catalog of entity and conversation targets, with manual target support.

## Architecture Overview

Core modules:

- `conversation.py`: Home Assistant custom conversation agent entrypoint.
- `agent_router.py`: Pipeline orchestration and path tracking.
- `catalog.py` + `catalog_sources.py`: Live catalog build/rebuild from entities + discoverable sources + manual targets.
- `matcher.py` + `phonetics.py`: Deterministic fuzzy matching with ASR-tolerant scoring.
- `safety.py`: High-risk gating and opposite-verb protections.
- `llm_adapter.py`: Structured LLM translation prompting and JSON parsing.
- `local_agent_adapter.py`: HA conversation delegation wrappers.
- `services.py`: `rebuild_catalog`, `get_catalog_stats`, `test_utterance`.

## Installation

1. Copy this repository into `config/custom_components/catalog_conversation_router`.
2. Restart Home Assistant.
3. Go to **Settings -> Devices & Services -> Add Integration**.
4. Search for **Catalog Conversation Router**.
5. Configure:
   - `local_agent_id` (preferred executor)
   - `llm_agent_id` (translator/fallback)
   - thresholds and toggles.

## Configuration Options

Required:

- `local_agent_id`
- `llm_agent_id`

Optional:

- `language` (default `en`)
- `fuzzy_enabled` (default `true`)
- `fuzzy_threshold` (default `0.84`)
- `ambiguity_gap` (default `0.08`)
- `llm_translate_enabled` (default `true`)
- `llm_fallback_enabled` (default `true`)
- `debug_enabled` (default `false`)
- `catalog_auto_refresh_enabled` (default `true`)
- `high_risk_threshold` (default `0.96`)
- `max_llm_candidates` (default `20`)
- `manual_targets` (list of custom target objects)

## Manual Conversation Targets

Use the options flow `manual_targets_json` field with a JSON list:

```json
[
  {
    "display_name": "Movie Mode",
    "target_type": "scene_shortcut",
    "sample_phrases": ["movie time", "start movie mode"],
    "canonical_phrase": "activate movie mode",
    "aliases": ["movie moat"],
    "enabled": true
  }
]
```

Manual targets are the primary reliable mechanism for custom conversation targets in v1.
Runtime discovery from `intent_script` and automation sentence triggers is best-effort and depends
on whether the running Home Assistant version exposes the required metadata at runtime.

## Services

- `catalog_conversation_router.rebuild_catalog`
- `catalog_conversation_router.get_catalog_stats`
- `catalog_conversation_router.test_utterance`

## Debugging

Enable `debug_enabled` to include routing traces in logs, including:

- normalized utterance
- selected path (`exact_local`, `fuzzy_local`, `llm_translated_local`, `llm_fallback`, `failed`)
- top fuzzy candidates and scores
- chosen canonical phrase
- LLM translation summary
- catalog revision

## Known Limitations

- Discovery of custom-sentence and intent-script targets is best-effort and may vary by HA version/configuration.
- In v1, manual conversation targets are recommended for predictable custom-target behavior.
- Exact HA conversation response internals can vary by conversation agent implementation.
- Some high-risk commands are intentionally blocked from weak fuzzy correction.

## Example Corrections

- `turn on the kitchen line` -> `turn on kitchen light`
- `activate movie moat` -> `activate movie mode`
- `turn off the haul light` -> `turn off hall light`
