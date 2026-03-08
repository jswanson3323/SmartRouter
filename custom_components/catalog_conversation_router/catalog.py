"""Catalog manager."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from uuid import uuid4

from .catalog_sources import (
    ConversationTargetSource,
    EntityCatalogSource,
    ManualConversationTargetSource,
)
from .models import Catalog, CatalogMetadata, RouterConfig

_LOGGER = logging.getLogger(__name__)


class CatalogManager:
    """Build and manage a live catalog snapshot."""

    def __init__(self, hass, config: RouterConfig) -> None:
        self._hass = hass
        self._config = config
        self._entity_source = EntityCatalogSource()
        self._conversation_source = ConversationTargetSource()
        self._manual_source = ManualConversationTargetSource()
        self._catalog = Catalog(
            metadata=CatalogMetadata(
                revision="bootstrap",
                last_refreshed=datetime.now(tz=UTC).isoformat(),
                language=config.language,
                entity_count=0,
                conversation_target_count=0,
                refresh_failures=0,
            )
        )
        self._lock = asyncio.Lock()

    async def async_rebuild(self) -> Catalog:
        """Perform a full catalog rebuild."""
        async with self._lock:
            return await self._async_rebuild_locked()

    async def _async_rebuild_locked(self) -> Catalog:
        refresh_failures = self._catalog.metadata.refresh_failures
        try:
            entities = await self._entity_source.async_collect(self._hass)
            conv_targets = await self._conversation_source.async_collect(self._hass)
            manual_targets = await self._manual_source.async_collect(self._config.manual_targets)
            conv_targets.extend(manual_targets)

            self._catalog = Catalog(
                metadata=CatalogMetadata(
                    revision=uuid4().hex,
                    last_refreshed=datetime.now(tz=UTC).isoformat(),
                    language=self._config.language,
                    entity_count=len(entities),
                    conversation_target_count=len(conv_targets),
                    refresh_failures=refresh_failures,
                ),
                entity_targets=entities,
                conversation_targets=conv_targets,
            )
            return self._catalog
        except Exception:
            refresh_failures += 1
            self._catalog.metadata.refresh_failures = refresh_failures
            _LOGGER.exception("Catalog rebuild failed")
            return self._catalog

    def get_catalog(self) -> Catalog:
        """Get current snapshot."""
        return self._catalog

    def stats(self) -> dict[str, str | int]:
        """Serialize current stats."""
        meta = self._catalog.metadata
        return {
            "revision": meta.revision,
            "last_refreshed": meta.last_refreshed,
            "language": meta.language,
            "entity_count": meta.entity_count,
            "conversation_target_count": meta.conversation_target_count,
            "refresh_failures": meta.refresh_failures,
        }
