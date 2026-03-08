"""Delegate conversation calls to specific Home Assistant agents."""

from __future__ import annotations

import logging
from typing import Any

from .models import FailureCategory, LocalAgentOutcome

_LOGGER = logging.getLogger(__name__)


class AgentAdapter:
    """Adapter around Home Assistant conversation processing."""

    def __init__(self, hass: Any) -> None:
        self._hass = hass

    async def async_process(
        self,
        *,
        agent_id: str,
        text: str,
        language: str,
        conversation_id: str | None,
        context: Any,
    ) -> LocalAgentOutcome:
        """Process text through a target agent."""
        try:
            from homeassistant.components import conversation

            response = await conversation.async_converse(
                hass=self._hass,
                text=text,
                conversation_id=conversation_id,
                context=context,
                language=language,
                agent_id=agent_id,
            )
        except Exception as err:
            _LOGGER.warning("Agent call failed for %s: %s", agent_id, err)
            return LocalAgentOutcome(
                success=False,
                response=None,
                response_text=str(err),
                failure_category=FailureCategory.UNKNOWN_FAILURE,
                raw=None,
            )

        response_text = self._extract_response_text(response)
        success = self._is_success(response)
        failure = None if success else self._classify_failure(response_text)

        return LocalAgentOutcome(
            success=success,
            response=response,
            response_text=response_text,
            failure_category=failure,
            raw=response,
        )

    def _extract_response_text(self, response: Any) -> str | None:
        speech = None
        try:
            speech = response.response.speech.get("plain", {}).get("speech")
        except Exception:
            speech = None
        if speech:
            return str(speech)
        return getattr(response, "response", None) and str(response.response)

    def _is_success(self, response: Any) -> bool:
        try:
            response_type = response.response.response_type
            if str(response_type).lower() in {
                "action_done",
                "query_answer",
            }:
                return True
        except Exception:
            pass

        response_text = (self._extract_response_text(response) or "").lower()
        if not response_text:
            return False
        failure_markers = (
            "didn't understand",
            "did not understand",
            "couldn't",
            "unable",
            "not sure",
            "no matching",
            "unknown",
        )
        return not any(marker in response_text for marker in failure_markers)

    def _classify_failure(self, text: str | None) -> FailureCategory:
        lowered = (text or "").lower()
        if any(v in lowered for v in ("didn't understand", "did not understand", "unknown")):
            return FailureCategory.NO_INTENT_MATCHED
        if "ambiguous" in lowered:
            return FailureCategory.AMBIGUOUS_MATCH
        if any(v in lowered for v in ("missing", "which one", "what device", "what area")):
            return FailureCategory.MISSING_TARGET_OR_SLOT
        if any(v in lowered for v in ("unsupported", "can't do", "cannot do")):
            return FailureCategory.UNSUPPORTED_ACTION
        if any(v in lowered for v in ("blocked", "unsafe", "not allowed")):
            return FailureCategory.UNSAFE_OR_BLOCKED_ACTION
        if any(v in lowered for v in ("failed", "error", "exception")):
            return FailureCategory.EXECUTION_FAILURE
        return FailureCategory.UNKNOWN_FAILURE
