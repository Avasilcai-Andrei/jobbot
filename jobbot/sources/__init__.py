"""Source plugin registry. Each module registers one Source subclass per ATS/board."""
from __future__ import annotations

import logging

from ..config import Config
from .base import Source

log = logging.getLogger(__name__)

REGISTRY: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    REGISTRY[cls.kind] = cls
    return cls


def build_sources(cfg: Config) -> list[Source]:
    sources: list[Source] = []
    for kind, scfg in cfg.sources.items():
        if not scfg.enabled:
            continue
        cls = REGISTRY.get(kind)
        if cls is None:
            raise ValueError(f"Unknown source kind {kind!r} in config; known: {sorted(REGISTRY)}")
        for entry in getattr(scfg, cls.config_key):
            if (source := cls.from_config(cfg, entry)) is not None:
                sources.append(source)
    return sources


# Import plugin modules so they register themselves.
from . import (  # noqa: E402,F401
    email_source, greenhouse, lever, personio, recruitee, smartrecruiters, teamtailor, workable, workday,
)
