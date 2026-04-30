# Catalog Conversation Router

`catalog_conversation_router` is a Home Assistant custom integration that routes conversation requests through a local-first, catalog-grounded pipeline.

It is designed to make voice control more tolerant of:
- ASR mistakes
- partial targets like `turn on the light`
- area-aware resolution from satellites/devices
- SuperArea grouping across related areas
- custom conversation phrases discovered from automations, blueprints, intent scripts, and manual targets

## Current Routing Order

The router currently evaluates an utterance in this order:

1. `local_state_query`
   - Deterministic state/status queries are intercepted first and sent to the Home Assistant local agent.
   - This includes:
     - entity state queries such as `is the office light on`
     - area/domain status queries such as `what lights are on in the kitchen`
     - temperature queries such as `what is the temperature in the great room`
   - Common ASR issues are normalized here too, for example `tempurature` and `office line`.
2. `fuzzy_local`
   - Deterministic catalog matching runs first.
   - If a candidate is accepted, the router sends the canonical phrase to the configured local conversation agent.
3. `llm_translated_local`
   - If fuzzy does not produce an accepted match, the LLM may translate the utterance into a canonical natural-language command.
   - The translation must exactly match a valid catalog phrase or entity command. Internal names such as `HassTurnOn` are rejected.
4. `exact_local`
   - The original utterance is sent unchanged to the configured local conversation agent.
5. `llm_fallback`
   - If enabled, the original utterance is sent directly to the configured LLM agent.

If the router is already inside an active continued conversation, it uses a hybrid continuation model:
- active local Home Assistant conversation -> direct local handoff
- active LLM fallback conversation:
  - local state queries may be intercepted and answered by the Home Assistant local agent
  - clear local control actions may also be intercepted and sent to the Home Assistant local agent
  - all other turns go straight back to the same LLM conversation thread

Typical routing paths are:
- `local_state_query`
- `fuzzy_local`
- `llm_translated_local`
- `exact_local`
- `llm_fallback`
- `failed`

When fuzzy already selects a valid route, LLM translation is skipped.

## Continued Conversations

The router preserves multi-turn clarification flows from both Home Assistant conversation handling and LLM fallback handling.

If a downstream response returns `continue_conversation: true`, the router remembers which executor owns that `conversation_id`.

On the next utterance with the same `conversation_id`, the router normally sends the reply back to the same executor with the normal Home Assistant conversation context, including:
- `conversation_id`
- `device_id`
- `satellite_id`
- `extra_system_prompt`

This applies to both:
- local Home Assistant conversation / sentence-trigger clarifications
- final LLM fallback clarifications

For active LLM conversations, the router can still step in for deterministic turns:
- exact state/status/temperature questions are answered by the Home Assistant local agent
- explicit local control actions such as `turn on the office drum light` are also allowed to go through the local path
- the active LLM owner is preserved, so later open-ended turns still return to the same LLM conversation

Examples:
- `turn on the hot tub` -> `What temp?` -> `99`
- `set a timer` -> `For how long?` -> `ten minutes`
- `Ask me a question` -> `What is the temperature in the great room?` -> local HA state answer while keeping the LLM thread available
- `The office` -> `the drum light` -> `turn on the office drum light` -> local HA action execution while keeping the LLM thread available

Once the downstream agent returns `continue_conversation: false`, the router clears that active continuation and future utterances resume the normal routing pipeline.

## Matching Behavior

### Catalog-grounded fuzzy matching

The matcher scores against:
- exposed entities
- conversation targets discovered from automations, blueprints, and intent scripts
- manual targets

Scoring uses multiple signals, including:
- token similarity
- phonetic similarity
- edit similarity
- action compatibility
- structure similarity
- semantic target similarity

The same general fuzzy logic is also used to recover likely state-query targets. Examples:
- `is the office line on` -> `Office Light`
- `what is the tempurature in the great room` -> temperature query in `Great Room`

### Area and SuperArea resolution

If an utterance omits an area, the matcher can use the origin area from the satellite/device context.

For generic domain requests such as:
- `turn on the light`
- `turn off the fan`

the router tries location resolution in this order:
1. exact area match
2. SuperArea match

`SuperArea` is read from Home Assistant area labels. The supported label formats are:
- `SuperArea: Great Room`
- slug-style IDs such as `superarea_great_room`

If a generic domain command maps to exactly one compatible entity in the origin area or SuperArea, that entity is preferred deterministically.

The router also uses this same locality order when recovering ambiguous state-query targets:
1. exact area match
2. SuperArea match
3. broader house match only if still safe

### Conversation target ambiguity handling

The matcher includes a few special cases for conversation targets:
- exact phrase matches win outright
- near-duplicate targets from the same automation/blueprint family do not block each other on ambiguity gap
- slot-based targets are penalized when required slots are missing

## Installation

1. Copy this repository to `/config/custom_components/catalog_conversation_router`
2. Restart Home Assistant
3. Add the integration from `Settings -> Devices & Services`
4. Configure:
   - `local_agent_id`
   - `llm_agent_id`
   - thresholds and toggles as needed

## Configuration Options

Required:
- `local_agent_id`
- `llm_agent_id`

Optional:
- `language` (default `en`)
- `fuzzy_enabled` (default `true`)
- `fuzzy_threshold`
- `ambiguity_gap`
- `llm_translate_enabled` (default `true`)
- `llm_fallback_enabled` (default `true`)
- `debug_enabled` (default `false`)
- `catalog_auto_refresh_enabled` (default `true`)
- `high_risk_threshold`
- `max_llm_candidates`
- `manual_targets`

## Manual Targets

Use the options flow `manual_targets_json` field with a JSON list such as:

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

Manual targets remain the most predictable way to add custom conversation phrases.

## Services

Available services:
- `catalog_conversation_router.rebuild_catalog`
- `catalog_conversation_router.get_catalog_stats`
- `catalog_conversation_router.test_utterance`
- `catalog_conversation_router.test_utterance_to_file`
- `catalog_conversation_router.dump_catalog`
- `catalog_conversation_router.dump_catalog_to_file`

## Preferred Debug Workflow

Use `catalog_conversation_router.test_utterance_to_file` when debugging routing behavior.

Example:

```yaml
service: catalog_conversation_router.test_utterance_to_file
data:
  text: "turn on the fan"
  area: "Kitchen"
  path: "router_debug.json"
```

This writes a JSON file to:

```text
/config/router_debug.json
```

The file includes the full decision tree, including:
- original and normalized utterance
- origin area and SuperArea context
- detected state-query metadata when applicable
- top fuzzy candidates with scores and detail
- fuzzy safety decision
- chosen canonical phrase
- rendered Assist input
- local state-query branch outcome
- exact local branch outcome
- fuzzy local branch outcome
- LLM translation summary and raw text when collected
- active continuation notes when a reply is routed directly back to local or LLM
- final selected path and executor

Set `execute_branches: true` only when you explicitly want to execute downstream branches while collecting the trace.

Example:

```yaml
service: catalog_conversation_router.test_utterance_to_file
data:
  text: "turn on the fan"
  area: "Kitchen"
  path: "router_debug_full.json"
  execute_branches: true
```

`catalog_conversation_router.test_utterance` returns the same trace payload as a service response without writing a file.

## Catalog Inspection

To inspect the live catalog, use:
- `catalog_conversation_router.dump_catalog`
- `catalog_conversation_router.dump_catalog_to_file`

The file-based variant writes to `/config`, defaulting to:

```text
/config/catalog_router_catalog.json
```

Catalog dumps are useful for verifying:
- entity names
- areas
- `super_area`
- aliases
- capabilities
- conversation targets discovered from YAML or blueprints

## Logging

Normal routing decisions are now debug-level only.

Warnings are reserved for actual failures or unavailable runtime components, so routine utterance debugging should happen through `test_utterance_to_file` rather than log inspection.

## Known Limitations

- Discovery of custom sentence triggers and intent scripts still depends on Home Assistant internals and can vary by version.
- Some high-risk corrections are intentionally blocked.
- Response payloads can vary by underlying conversation agent.
- Pronoun-only action follow-ups such as `turn it on` are not fully resolved yet unless the current turn still names the target clearly.

## Example Outcomes

- `what lights are on in the kitchen` -> local HA area/light state answer
- `is the office line on` -> fuzzy state-query recovery -> local HA state answer
- `what is the tempurature in the great room` -> typo-tolerant temperature query -> local HA temperature answer
- `turn on the ligh` in `Master Bedroom` -> `turn on Master Bedroom Light`
- `turn on the fan` from `Kitchen` with `SuperArea: Great Room` -> `turn on Great Room Fan`
- `turn on the spa` -> exact spa automation phrase wins without falling through to LLM
- `turn on the hot tub` -> `What temp?` -> `99` stays in the same local HA conversation
- `set a timer` -> `For how long?` -> `ten minutes` stays in the same LLM fallback conversation
