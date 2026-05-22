"""Persistent conversation trace logging."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
import re
from typing import Any

import yaml

from .const import DOMAIN
from .models import LocalAgentOutcome, RouterResult

_LOGGER = logging.getLogger(__name__)

TRACE_LOG_RETENTION = timedelta(days=2)
TRACE_LOG_DIRNAME = "conversation_logs"
MAX_FILENAME_SLUG_CHARS = 48
NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


class ConversationTraceStore:
    """Write per-turn router traces to YAML files."""

    def __init__(self, hass: Any, *, entry_id: str, enabled: bool) -> None:
        self._hass = hass
        self._entry_id = entry_id
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Return whether trace persistence is enabled."""
        return self._enabled

    async def async_write_turn(
        self,
        *,
        user_input: Any,
        result: RouterResult,
        outcome: LocalAgentOutcome,
        streamed: bool,
    ) -> None:
        """Persist a single conversation turn if enabled."""
        if not self._enabled:
            return

        payload = self._build_payload(
            user_input=user_input,
            result=result,
            outcome=outcome,
            streamed=streamed,
        )

        def _write() -> Path:
            output_dir = Path(self._hass.config.path(DOMAIN, TRACE_LOG_DIRNAME))
            output_dir.mkdir(parents=True, exist_ok=True)
            self._purge_old_logs(output_dir)
            output_path = output_dir / self._build_filename(payload)
            output_path.write_text(
                yaml.safe_dump(
                    payload,
                    sort_keys=False,
                    allow_unicode=False,
                ),
                encoding="utf-8",
            )
            return output_path

        try:
            output_path = await self._hass.async_add_executor_job(_write)
            _LOGGER.debug("Wrote conversation trace log to %s", output_path)
        except Exception:
            _LOGGER.exception(
                "Failed to persist conversation trace for entry %s",
                self._entry_id,
            )

    def _build_payload(
        self,
        *,
        user_input: Any,
        result: RouterResult,
        outcome: LocalAgentOutcome,
        streamed: bool,
    ) -> dict[str, Any]:
        timestamp = datetime.now(tz=UTC)
        trace = result.trace.as_dict()
        payload: dict[str, Any] = {
            "logged_at": timestamp.isoformat(),
            "entry_id": self._entry_id,
            "utterance": getattr(user_input, "text", None),
            "path": result.path.value,
            "streamed": streamed,
            "input": {
                "language": getattr(user_input, "language", None),
                "conversation_id": getattr(user_input, "conversation_id", None),
                "device_id": getattr(user_input, "device_id", None),
                "satellite_id": getattr(user_input, "satellite_id", None),
                "agent_id": getattr(user_input, "agent_id", None),
                "extra_system_prompt": getattr(user_input, "extra_system_prompt", None),
            },
            "outcome": {
                "success": outcome.success,
                "response_text": outcome.response_text,
                "response_type": outcome.response_type,
                "error_code": outcome.error_code,
                "processed_locally": outcome.processed_locally,
                "conversation_id": outcome.conversation_id,
                "continue_conversation": outcome.continue_conversation,
                "failure_category": outcome.failure_category.value
                if outcome.failure_category is not None
                else None,
            },
            "trace": trace,
        }
        return payload

    def _build_filename(self, payload: dict[str, Any]) -> str:
        logged_at = str(payload["logged_at"]).replace(":", "-")
        utterance = str(payload.get("utterance") or "empty")
        slug = NON_SLUG_RE.sub("-", utterance.lower()).strip("-")[:MAX_FILENAME_SLUG_CHARS]
        if not slug:
            slug = "utterance"
        return f"{logged_at}_{slug}.yaml"

    def _purge_old_logs(self, output_dir: Path) -> None:
        cutoff = datetime.now(tz=UTC) - TRACE_LOG_RETENTION
        for path in output_dir.glob("*.yaml"):
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError:
                continue
            if modified < cutoff:
                try:
                    path.unlink()
                except OSError:
                    _LOGGER.warning("Failed to purge old conversation trace log %s", path)
