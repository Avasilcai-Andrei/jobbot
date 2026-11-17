"""Local scoring through Ollama's REST API (https://github.com/ollama/ollama/blob/main/docs/api.md)."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config import Config
from .. import ollama_ctl
from ..db import DB
from .base import RESULT_SCHEMA, Scorer, ScoreFailed, ScoreUnavailable, ScoreUnreachable

log = logging.getLogger(__name__)


class OllamaScorer(Scorer):
    backend = "ollama"
    uses_gpu = True

    def __init__(self, cfg: Config, cv_text: str, db: DB, *, model: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        o = cfg.llm.ollama
        super().__init__(cfg, cv_text, db, model=model or o.model, context_tokens=o.num_ctx,
                         max_concurrency=o.max_concurrency)
        self.opts = o
        self.url = o.url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.url,
            timeout=httpx.Timeout(o.timeout_seconds, connect=5.0),
            transport=transport,
        )
        self.last_stats: dict[str, Any] = {}
        self.processor_checked = False

    async def aclose(self) -> None:
        await self.client.aclose()

    async def unload(self) -> None:
        try:
            await ollama_ctl.unload(self.client, self.model)
        except httpx.HTTPError as e:
            log.warning("Could not unload %s: %s", self.model, type(e).__name__)

    async def check_available(self) -> str | None:
        self.processor_checked = False  # once per run, after the first answer (the model is loaded by then)
        self.unreachable = False
        try:
            resp = await self.client.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            self.unreachable = isinstance(e, httpx.TransportError)
            return f"Ollama is not reachable at {self.url} ({type(e).__name__}); is it running? (`ollama serve`)"
        names = {m.get("name") for m in resp.json().get("models", [])} | {m.get("model") for m in resp.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            return f"Ollama model {self.model!r} is not pulled; run `ollama pull {self.model}`"
        return None

    def request_body(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "format": RESULT_SCHEMA,
            "stream": False,
            "keep_alive": self.opts.keep_alive,
            "options": {
                "temperature": 0,
                "seed": 0,
                "num_ctx": self.opts.num_ctx,
                "num_predict": self.opts.num_predict,
            },
        }
        if self.opts.think is not None:
            body["think"] = self.opts.think
        return body

    async def _complete(self, system: str, messages: list[dict[str, str]]) -> str:
        try:
            resp = await self.client.post("/api/chat", json=self.request_body(system, messages))
        except httpx.TimeoutException:
            raise ScoreFailed(f"Ollama timed out after {self.opts.timeout_seconds:.0f}s") from None
        except httpx.TransportError as e:
            raise ScoreUnreachable(f"Ollama is not reachable at {self.url} ({type(e).__name__})") from None
        if resp.status_code == 404:
            raise ScoreUnavailable(f"Ollama model {self.model!r} not found; run `ollama pull {self.model}`")
        if resp.status_code >= 400:
            raise ScoreFailed(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        self.last_stats = {k: data.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration", "done_reason")}
        prompt_tokens = data.get("prompt_eval_count") or 0
        if prompt_tokens >= self.opts.num_ctx - 16:
            log.warning("Prompt used %d of %d context tokens; Ollama may have cut it. "
                        "Lower llm.max_description_tokens or raise llm.ollama.num_ctx", prompt_tokens, self.opts.num_ctx)
        if not self.processor_checked:
            self.processor_checked = True
            await self.check_processor()
        return (data.get("message") or {}).get("content") or ""

    async def check_processor(self) -> str | None:
        """Warn when the loaded model isn't fully on the GPU; returns what `ollama ps` shows as PROCESSOR.

        If Ollama starts before the NVIDIA driver is ready, its GPU discovery fails and it silently runs on the CPU
        (seen when it starts right at logon), which makes every score many times slower."""
        try:
            resp = await self.client.get("/api/ps", timeout=5.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
        except (httpx.HTTPError, ValueError) as e:
            log.debug("Couldn't read /api/ps: %s", e)
            return None
        entry = next((m for m in models if self.model in (m.get("name"), m.get("model"))
                      or f"{self.model}:latest" in (m.get("name"), m.get("model"))), None)
        if not entry or not entry.get("size"):
            return None
        size, vram = entry["size"], entry.get("size_vram") or 0
        cpu = round(100 * (size - vram) / size)
        processor = "100% GPU" if vram >= size else "100% CPU" if vram <= 0 else f"{cpu}%/{100 - cpu}% CPU/GPU"
        if vram < size:
            log.warning("Ollama is running %s on %s (`ollama ps`), so scoring is slow. Ollama probably started before "
                        "the GPU driver was ready: quit the Ollama tray app and start it again", self.model, processor)
        return processor
