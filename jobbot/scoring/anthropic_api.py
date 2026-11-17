"""Scoring with the Claude API (structured outputs guarantee schema-valid JSON)."""
from __future__ import annotations

import anthropic

from ..config import Config
from ..db import DB
from .base import RESULT_SCHEMA, Scorer, ScoreFailed, ScoreUnavailable

CONTEXT_TOKENS = 1_000_000


class AnthropicScorer(Scorer):
    backend = "anthropic"

    def __init__(self, cfg: Config, cv_text: str, db: DB, *, model: str | None = None,
                 client: anthropic.AsyncAnthropic | None = None):
        a = cfg.llm.anthropic
        super().__init__(cfg, cv_text, db, model=model or a.model, context_tokens=CONTEXT_TOKENS,
                         max_concurrency=a.max_concurrency)
        self.effort = a.effort
        self.has_key = client is not None or bool(cfg.secrets.anthropic_api_key)
        self.client = client or (anthropic.AsyncAnthropic(max_retries=3, timeout=120) if self.has_key else None)

    async def check_available(self) -> str | None:
        return None if self.has_key else "ANTHROPIC_API_KEY is not set"

    async def aclose(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            await self.client.close()

    async def _complete(self, system: str, messages: list[dict[str, str]]) -> str:
        if self.client is None:
            raise ScoreUnavailable("ANTHROPIC_API_KEY is not set")
        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                # The CV prefix is identical for every job, so cache it across calls in a run.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": RESULT_SCHEMA}},
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.APIConnectionError) as e:
            raise ScoreUnavailable(f"Claude API unavailable: {type(e).__name__}") from e
        except anthropic.APIError as e:
            raise ScoreFailed(f"Claude API error: {e}") from e
        if resp.stop_reason == "refusal":
            raise ScoreFailed(f"model declined (request {resp._request_id})")
        if resp.stop_reason == "max_tokens":
            raise ScoreFailed(f"response truncated (request {resp._request_id})")
        return next((b.text for b in resp.content if b.type == "text"), "")
