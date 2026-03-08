"""Constants for the Catalog Conversation Router integration."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "catalog_conversation_router"
PLATFORMS: Final[list[str]] = []

CONF_LOCAL_AGENT_ID: Final = "local_agent_id"
CONF_LLM_AGENT_ID: Final = "llm_agent_id"
CONF_LANGUAGE: Final = "language"
CONF_FUZZY_ENABLED: Final = "fuzzy_enabled"
CONF_FUZZY_THRESHOLD: Final = "fuzzy_threshold"
CONF_AMBIGUITY_GAP: Final = "ambiguity_gap"
CONF_LLM_TRANSLATE_ENABLED: Final = "llm_translate_enabled"
CONF_LLM_FALLBACK_ENABLED: Final = "llm_fallback_enabled"
CONF_DEBUG_ENABLED: Final = "debug_enabled"
CONF_CATALOG_AUTO_REFRESH_ENABLED: Final = "catalog_auto_refresh_enabled"
CONF_HIGH_RISK_THRESHOLD: Final = "high_risk_threshold"
CONF_MAX_LLM_CANDIDATES: Final = "max_llm_candidates"
CONF_MANUAL_TARGETS: Final = "manual_targets"

DEFAULT_FUZZY_ENABLED: Final = True
DEFAULT_FUZZY_THRESHOLD: Final = 0.84
DEFAULT_AMBIGUITY_GAP: Final = 0.08
DEFAULT_LLM_TRANSLATE_ENABLED: Final = True
DEFAULT_LLM_FALLBACK_ENABLED: Final = True
DEFAULT_DEBUG_ENABLED: Final = False
DEFAULT_CATALOG_AUTO_REFRESH_ENABLED: Final = True
DEFAULT_HIGH_RISK_THRESHOLD: Final = 0.96
DEFAULT_MAX_LLM_CANDIDATES: Final = 20

SERVICE_REBUILD_CATALOG: Final = "rebuild_catalog"
SERVICE_GET_CATALOG_STATS: Final = "get_catalog_stats"
SERVICE_TEST_UTTERANCE: Final = "test_utterance"

ATTR_TEXT: Final = "text"

LOGGER_NAME: Final = "custom_components.catalog_conversation_router"

UPDATE_LISTENER: Final = "update_listener"
