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

# Catalog Conversation Router

`catalog_conversation_router` is a Home Assistant custom integration that provides a **smart conversation router** with local-first execution, catalog‑grounded fuzzy correction, optional LLM translation, and final LLM fallback.

The goal of the integration is to make voice control significantly more tolerant of:

- ASR mistakes
- partial commands
- missing area references
- near‑matches for entity names

while still keeping execution **deterministic and local-first**.

---

# How Routing Works

Every utterance follows this pipeline:

1. **Exact Local Agent**  
   The configured Home Assistant conversation agent is called directly.

   If the local agent successfully executes the command, the router stops.

2. **Catalog‑Grounded Fuzzy Correction**  
   If the local agent fails, the router attempts deterministic fuzzy matching using the internal catalog of:

   - exposed entities
   - conversation triggers
   - manual targets

   If a strong match is found, the router rewrites the command into a **canonical phrase** and retries the local agent.

3. **LLM Translation (optional)**  
   If fuzzy matching fails, the configured LLM agent can attempt to translate the utterance into a canonical command.

4. **LLM Fallback (optional)**  
   If translation fails, the request can optionally be handed directly to the LLM.

Typical routing paths:

- `exact_local`
- `fuzzy_local`
- `llm_translated_local`
- `llm_fallback`
- `failed`

---

# Key Features

## Local‑First Execution

Commands always attempt execution through the configured Home Assistant conversation agent before any fuzzy or LLM logic runs.

This preserves:

- native Home Assistant intent handling
- device integrations
- local automations

---

## Catalog‑Grounded Fuzzy Matching

The integration maintains a **live catalog** of targets including:

- exposed entities
- automation conversation triggers
- intent scripts
- manual targets

The fuzzy matcher is **deterministic and explainable** and uses multiple scoring signals:

- token similarity
- phonetic similarity
- edit distance
- action compatibility
- structure similarity
- semantic target similarity

Example corrections:

```
turn on the kitchen line → turn on kitchen light
activate movie moat → activate movie mode
turn off the haul light → turn off hall light
```

---

## Area‑Aware Matching

If a command does **not include an area**, the router can use the **area of the satellite/device that issued the command**.

Example:

Utterance:

```
turn off the fan
```

If the request originated from a satellite in **Great Room**, the matcher will prefer:

```
fan.great_room_fan
```

instead of unrelated fans elsewhere.

Area awareness is applied as a **ranking boost**, not a hard filter.

---

## Domain Hint Matching

Generic commands like:

```
turn on the light
turn off the fan
turn on the TV
```

receive additional scoring bias toward entities whose **domain matches the noun**.

Example:

```
fan → fan.* entities
light → light.* entities
```

---

# Catalog System

The router maintains a **live catalog** containing:

- entity targets
- conversation targets
- manual targets

The catalog is refreshed automatically or via service.

---

## Dumping the Catalog to a File

A debugging service allows writing the full catalog to disk.

Service:

```
catalog_conversation_router.dump_catalog_to_file
```

Default output:

```
/config/catalog_router_catalog.json
```

Optional parameter:

```yaml
path: catalog_router_catalog.json
```

This file contains:

- entity entries
- areas
- aliases
- tokens
- phonetic tokens
- conversation targets

This is extremely useful when diagnosing fuzzy matching behavior.

---

# Architecture Overview

Core modules:

```
conversation.py
```
Home Assistant custom conversation agent entrypoint.

```
agent_router.py
```
Pipeline orchestration and routing logic.

```
catalog.py
catalog_sources.py
```
Builds the live catalog of entities and conversation targets.

```
matcher.py
phonetics.py
```
Deterministic fuzzy matching and ASR‑tolerant scoring.

```
safety.py
```
Guards against dangerous corrections (ex: opposite verbs).

```
local_agent_adapter.py
```
Delegates execution to the configured Home Assistant conversation agent.

```
llm_adapter.py
```
Handles LLM translation and structured parsing.

```
services.py
```
Service definitions.

---

# Installation

1. Copy the repository into:

```
/config/custom_components/catalog_conversation_router
```

2. Restart Home Assistant.

3. Go to:

```
Settings → Devices & Services → Add Integration
```

4. Search for:

```
Catalog Conversation Router
```

5. Configure:

- `local_agent_id`
- `llm_agent_id`
- thresholds

---

# Configuration Options

Required:

```
local_agent_id
llm_agent_id
```

Optional:

```
language (default "en")
fuzzy_enabled (default true)
fuzzy_threshold (default 0.84)
ambiguity_gap (default 0.08)
llm_translate_enabled (default true)
llm_fallback_enabled (default true)
debug_enabled (default false)
catalog_auto_refresh_enabled (default true)
high_risk_threshold (default 0.96)
max_llm_candidates (default 20)
manual_targets
```

---

# Services

### Rebuild Catalog

```
catalog_conversation_router.rebuild_catalog
```

Forces a full catalog rebuild.

---

### Get Catalog Stats

```
catalog_conversation_router.get_catalog_stats
```

Returns:

- entity count
- conversation target count
- revision id

---

### Test Utterance

```
catalog_conversation_router.test_utterance
```

Allows testing routing without executing commands.

Optional parameters include:

```
area
satellite_id
device_id
```

The trace will show how the router evaluated the command.

---

### Dump Catalog to File

```
catalog_conversation_router.dump_catalog_to_file
```

Writes the full catalog to a JSON file for inspection.

---

# Debugging

Enable:

```
debug_enabled: true
```

Trace output includes:

- normalized utterance
- selected routing path
- exact local result
- fuzzy candidate list
- fuzzy scores
- canonical phrase chosen
- LLM translation summary
- catalog revision
- origin area
- effective area hint

Example path output:

```
selected_path: fuzzy_local
```

---

# Known Limitations

- Discovery of custom conversation triggers depends on Home Assistant internals and may vary by version.
- Some commands intentionally bypass fuzzy matching for safety reasons.
- Exact conversation response formats can vary between conversation agents.

---

# Example Corrections

```
turn on the kitchen line → turn on kitchen light
activate movie moat → activate movie mode
turn off the haul light → turn off hall light
turn off the fan (Great Room satellite) → fan.great_room_fan
```