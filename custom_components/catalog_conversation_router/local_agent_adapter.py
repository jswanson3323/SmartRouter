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
        resolved_agent_id = None if agent_id in {"homeassistant", "__default__", "default"} else agent_id

        try:
            from homeassistant.components import conversation

            _LOGGER.debug(
                "LocalAgentAdapter.async_process text=%s configured_agent_id=%s resolved_agent_id=%s language=%s conversation_id=%s",
                text,
                agent_id,
                resolved_agent_id,
                language,
                conversation_id,
            )
            response = await conversation.async_converse(
                hass=self._hass,
                text=text,
                conversation_id=conversation_id,
                context=context,
                language=language,
                agent_id=resolved_agent_id,
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
        response_type = self._extract_response_type(response)
        error_code = self._extract_error_code(response)
        processed_locally = self._extract_processed_locally(response)
        success = self._is_success(response)
        failure = None if success else self._classify_failure(response_text, error_code=error_code, response_type=response_type)

        _LOGGER.debug(
            "LocalAgentAdapter result configured_agent_id=%s resolved_agent_id=%s response_type=%s error_code=%s processed_locally=%s success=%s response_text=%s",
            agent_id,
            resolved_agent_id,
            response_type,
            error_code,
            processed_locally,
            success,
            response_text,
        )

        return LocalAgentOutcome(
            success=success,
            response=response,
            response_text=response_text,
            failure_category=failure,
            raw=response,
            response_type=response_type,
            error_code=error_code,
            processed_locally=processed_locally,
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

    def _extract_response_type(self, response: Any) -> str | None:
        try:
            return str(response.response.response_type)
        except Exception:
            return None

    def _extract_error_code(self, response: Any) -> str | None:
        try:
            data = response.response.data
            if isinstance(data, dict):
                code = data.get("code")
            else:
                code = getattr(data, "code", None)
            return str(code) if code is not None else None
        except Exception:
            return None

    def _extract_processed_locally(self, response: Any) -> bool | None:
        for attr_path in (
            ("processed_locally",),
            ("intent_output", "processed_locally"),
            ("intent", "processed_locally"),
        ):
            try:
                value = response
                for attr in attr_path:
                    value = getattr(value, attr)
                if isinstance(value, bool):
                    return value
            except Exception:
                continue
        return None

    def _is_success(self, response: Any) -> bool:
        response_type = (self._extract_response_type(response) or "").lower()
        if response_type == "error":
            return False
        if response_type in {"action_done", "query_answer"}:
            return True

        error_code = (self._extract_error_code(response) or "").lower()
        if error_code:
            return False

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
            "not aware of any device",
            "no device called",
            "could not find",
            "can't find",
            "cannot find",
            "sorry, i am not aware",
        )
        return not any(marker in response_text for marker in failure_markers)

    def _classify_failure(self, text: str | None, *, error_code: str | None = None, response_type: str | None = None) -> FailureCategory:
        lowered = (text or "").lower()
        code = (error_code or "").lower()
        rtype = (response_type or "").lower()

        if code in {"no_valid_targets", "target_not_found", "entity_not_found"} or any(v in lowered for v in ("not aware of any device", "no device called", "could not find", "can't find", "cannot find")):
            return FailureCategory.MISSING_TARGET_OR_SLOT
        if code in {"no_intent_match", "intent_not_recognized"} or any(v in lowered for v in ("didn't understand", "did not understand", "unknown")):
            return FailureCategory.NO_INTENT_MATCHED
        if code in {"ambiguous_targets", "multiple_targets"} or "ambiguous" in lowered:
            return FailureCategory.AMBIGUOUS_MATCH
        if code in {"unsupported_action", "not_supported"} or any(v in lowered for v in ("unsupported", "can't do", "cannot do")):
            return FailureCategory.UNSUPPORTED_ACTION
        if code in {"blocked", "unsafe", "not_allowed"} or any(v in lowered for v in ("blocked", "unsafe", "not allowed")):
            return FailureCategory.UNSAFE_OR_BLOCKED_ACTION
        if rtype == "error" or code in {"execution_error", "service_error"} or any(v in lowered for v in ("failed", "error", "exception")):
            return FailureCategory.EXECUTION_FAILURE
        return FailureCategory.UNKNOWN_FAILURE
