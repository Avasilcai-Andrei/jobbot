"""A polite async HTTP client: identifies itself, rate-limits per host, backs off on errors."""
from __future__ import annotations

import asyncio
import logging
import random
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 120.0
BODY_PREVIEW_CHARS = 300


def safe_url(url: str) -> str:
    """Strip secrets (webhook tokens live in the path) before a URL goes into a log line."""
    u = httpx.URL(url)
    if "/webhooks/" in u.path:
        return f"{u.scheme}://{u.host}/…/webhooks/***"
    return str(u.copy_with(query=None))


class BadResponse(ValueError):
    """A successful (2xx) response whose body isn't what we expected, e.g. a maintenance page instead of JSON.

    The message carries the status, content-type, any redirects and the start of the body, so the log shows
    whether it was a maintenance page, a block page or an empty answer."""

    def __init__(self, resp: httpx.Response, problem: str):
        self.status_code = resp.status_code
        self.content_type = resp.headers.get("content-type", "")
        self.preview = " ".join(resp.text[: BODY_PREVIEW_CHARS * 2].split())[:BODY_PREVIEW_CHARS]
        hops = "".join(f" (redirected from {safe_url(str(r.request.url))}, HTTP {r.status_code})"
                       for r in resp.history)
        super().__init__(
            f"{problem}: HTTP {resp.status_code} from {safe_url(str(resp.request.url))}{hops}, "
            f"content-type {self.content_type or 'none'!r}, {len(resp.content)} bytes, body: {self.preview!r}"
        )


def json_body(resp: httpx.Response) -> Any:
    """resp.json(), but a body that isn't JSON raises BadResponse with the details instead of a bare
    JSONDecodeError ("Expecting value: line 1 column 1 (char 0)" says nothing about what came back)."""
    try:
        return resp.json()
    except ValueError as e:  # JSONDecodeError and UnicodeDecodeError are both ValueErrors
        raise BadResponse(resp, f"not JSON ({e})") from e


def xml_body(resp: httpx.Response, parse: Callable[[str], Any]) -> Any:
    """parse(resp.text), but a body that isn't well-formed XML (a maintenance or block page, an empty answer)
    raises BadResponse with the status, content-type and the start of the body, like json_body does."""
    try:
        return parse(resp.text)
    except ET.ParseError as e:
        raise BadResponse(resp, f"not XML ({e})") from e


class PoliteClient:
    def __init__(
        self,
        user_agent: str,
        timeout: float = 20,
        per_host_interval: float = 2.0,
        max_retries: int = 3,
        max_concurrency: int = 8,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self._per_host_interval = per_host_interval
        self._max_retries = max_retries
        self._sem = asyncio.Semaphore(max_concurrency)
        self._host_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._host_last: dict[str, float] = {}
        self._sleep = sleep

    async def __aenter__(self) -> PoliteClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _wait_turn(self, host: str) -> None:
        async with self._host_locks[host]:
            last = self._host_last.get(host)
            if last is not None:
                wait = self._per_host_interval - (time.monotonic() - last)
                if wait > 0:
                    await self._sleep(wait)
            self._host_last[host] = time.monotonic()

    def _backoff(self, attempt: int, resp: httpx.Response | None) -> float:
        if resp is not None and (ra := resp.headers.get("Retry-After")):
            try:
                return min(float(ra), MAX_RETRY_AFTER)
            except ValueError:
                pass
        return min(2.0 * 2**attempt, 60.0) + random.uniform(0, 1)

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        host = httpx.URL(url).host
        for attempt in range(self._max_retries + 1):
            await self._wait_turn(host)
            resp: httpx.Response | None = None
            error: Exception | None = None
            async with self._sem:
                try:
                    resp = await self._client.request(method, url, **kwargs)
                except httpx.TransportError as e:
                    error = e
            if resp is not None and resp.status_code not in RETRY_STATUSES:
                resp.raise_for_status()
                return resp
            if attempt == self._max_retries:
                if resp is not None:
                    resp.raise_for_status()
                assert error is not None
                raise error
            delay = self._backoff(attempt, resp)
            why = f"HTTP {resp.status_code}" if resp is not None else repr(error)
            log.info("%s %s failed (%s); retry %d in %.1fs", method, safe_url(url), why, attempt + 1, delay)
            await self._sleep(delay)
        raise AssertionError("unreachable")

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)
