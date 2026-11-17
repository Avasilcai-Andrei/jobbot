"""Pluggable stage-2 scorers, selected by `llm.backend` in config.yaml."""
from __future__ import annotations

import logging

from ..config import Config
from ..cv import load_cv_text
from ..db import DB
from .base import MatchResult, Scorer, ScoreFailed, ScoreUnavailable, ScoreUnreachable

log = logging.getLogger(__name__)

__all__ = ["MatchResult", "Scorer", "ScoreFailed", "ScoreUnavailable", "ScoreUnreachable", "make_scorer"]


def make_scorer(cfg: Config, db: DB, *, model: str | None = None,
                cv_text: str | None = None) -> tuple[Scorer | None, str | None]:
    """Return (scorer, None), or (None, why scoring is off)."""
    backend = cfg.llm.backend
    if backend == "off":
        return None, "scoring is off (llm.backend: off)"
    if cv_text is None:
        cv_text = load_cv_text(cfg.cv_path, cfg.data_dir)
    if not cv_text:
        return None, f"no CV text (expected {cfg.cv_path})"
    if backend == "ollama":
        from .ollama import OllamaScorer
        return OllamaScorer(cfg, cv_text, db, model=model), None
    if backend == "anthropic":
        from .anthropic_api import AnthropicScorer
        return AnthropicScorer(cfg, cv_text, db, model=model), None
    raise ValueError(f"unknown llm.backend {backend!r}")
