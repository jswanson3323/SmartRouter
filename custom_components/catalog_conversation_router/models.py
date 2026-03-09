"""Typed models for Catalog Conversation Router."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ResolutionPath(StrEnum):
    """Pipeline resolution path."""

    EXACT_LOCAL = "exact_local"
    FUZZY_LOCAL = "fuzzy_local"
    LLM_TRANSLATED_LOCAL = "llm_translated_local"
    LLM_FALLBACK = "llm_fallback"
    FAILED = "failed"


class FailureCategory(StrEnum):
    """Local execution failure categories."""

    NO_INTENT_MATCHED = "no_intent_matched"
    AMBIGUOUS_MATCH = "ambiguous_match"
    MISSING_TARGET_OR_SLOT = "missing_target_or_slot"
    UNSUPPORTED_ACTION = "unsupported_action"
    EXECUTION_FAILURE = "execution_failure"
    UNSAFE_OR_BLOCKED_ACTION = "unsafe_or_blocked_action"
    UNKNOWN_FAILURE = "unknown_failure"


class RiskTier(StrEnum):
    """Command risk level."""

    LOW = "low_risk"
    HIGH = "high_risk"


class CandidateType(StrEnum):
    """Catalog candidate type."""

    ENTITY_COMMAND = "entity_command"
    CONVERSATION_TARGET = "conversation_target"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class EntityTarget:
    """Catalog entity target."""

    entity_id: str
    name: str
    normalized_name: str
    aliases: list[str]
    domain: str
    area: str | None
    floor: str | None
    device_name: str | None
    exposed: bool | None
    capabilities: list[str]
    tokens: list[str]
    phonetic_tokens: list[str]


@dataclass(slots=True)
class ConversationTarget:
    """Catalog conversation target."""

    target_id: str
    type: str
    display_name: str
    normalized_name: str
    sample_phrases: list[str]
    canonical_phrase: str
    source: str
    slots: list[str]
    tokens: list[str]
    phonetic_tokens: list[str]
    aliases: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass(slots=True)
class CatalogMetadata:
    """Catalog metadata."""

    revision: str
    last_refreshed: str
    language: str
    entity_count: int
    conversation_target_count: int
    refresh_failures: int = 0


@dataclass(slots=True)
class Catalog:
    """Live catalog structure."""

    metadata: CatalogMetadata
    entity_targets: list[EntityTarget] = field(default_factory=list)
    conversation_targets: list[ConversationTarget] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize catalog."""
        return asdict(self)


@dataclass(slots=True)
class CandidateScore:
    """Scored candidate for fuzzy matching."""

    candidate_id: str
    candidate_type: CandidateType
    canonical_phrase: str
    score: float
    action: str | None
    target_name: str
    detail: dict[str, Any]


@dataclass(slots=True)
class MatchResult:
    """Result from fuzzy matcher."""

    matched: bool
    best: CandidateScore | None
    top_candidates: list[CandidateScore]
    inferred_action: str | None
    normalized_utterance: str
    parsed_target_before_normalization: str
    parsed_target_after_normalization: str
    origin_area: str | None = None
    effective_area_hint: str | None = None


@dataclass(slots=True)
class LocalAgentOutcome:
    """Result envelope for delegated local/LLM agent calls."""

    success: bool
    response: Any | None
    response_text: str | None
    failure_category: FailureCategory | None
    raw: Any | None
    response_type: str | None = None
    error_code: str | None = None
    processed_locally: bool | None = None


@dataclass(slots=True)
class LLMTranslationResult:
    """Structured LLM translation payload."""

    mode: str
    canonical_text: str | None
    confidence: float
    target_type: CandidateType
    notes: str | None
    valid: bool
    raw_text: str | None = None


@dataclass(slots=True)
class ResolutionTrace:
    """Debug trace payload."""

    original_utterance: str
    normalized_utterance: str
    selected_path: ResolutionPath = ResolutionPath.FAILED
    exact_local_outcome: str | None = None
    exact_local_response_text: str | None = None
    exact_local_response_type: str | None = None
    exact_local_error_code: str | None = None
    exact_local_processed_locally: bool | None = None
    origin_area: str | None = None
    effective_area_hint: str | None = None
    failure_category: str | None = None
    top_fuzzy_candidates: list[dict[str, Any]] = field(default_factory=list)
    chosen_canonical_phrase: str | None = None
    assist_pipeline_input: str | None = None
    llm_translation_summary: dict[str, Any] | None = None
    final_executor: str | None = None
    catalog_revision: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def as_dict(self) -> dict[str, Any]:
        """Serialize trace."""
        return asdict(self)


@dataclass(slots=True)
class RouterResult:
    """Router output."""

    path: ResolutionPath
    outcome: LocalAgentOutcome
    trace: ResolutionTrace


@dataclass(slots=True)
class RouterConfig:
    """Config used by routing pipeline."""

    local_agent_id: str
    llm_agent_id: str
    language: str
    fuzzy_enabled: bool
    fuzzy_threshold: float
    ambiguity_gap: float
    llm_translate_enabled: bool
    llm_fallback_enabled: bool
    debug_enabled: bool
    catalog_auto_refresh_enabled: bool
    high_risk_threshold: float
    max_llm_candidates: int
    manual_targets: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SafetyDecision:
    """Safety gate result for fuzzy execution."""

    allowed: bool
    reason: str
    risk_tier: RiskTier
