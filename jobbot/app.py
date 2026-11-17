"""Wire the pieces together from a Config."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from .config import Config
from .db import DB
from .http import PoliteClient
from .notify import ConsoleNotifier, DiscordNotifier, EmailNotifier, Notifier, NotifierChain
from .pipeline import Pipeline
from .scoring import make_scorer
from .sources import build_sources

log = logging.getLogger(__name__)


def channels(cfg: Config, http: PoliteClient) -> list[tuple[str, Notifier]]:
    """The real notification channels configured in config.yaml + .env, Discord first."""
    out: list[tuple[str, Notifier]] = []
    s = cfg.secrets
    if cfg.notify.discord and s.discord_webhook_url:
        out.append(("Discord", DiscordNotifier(s.discord_webhook_url, http)))
    email = cfg.notify.email
    if email.mode != "off" and s.smtp_user and s.smtp_app_password and s.alert_email_to:
        out.append(("Email", EmailNotifier(s.smtp_user, s.smtp_app_password, s.alert_email_to, email.smtp_host,
                                           email.smtp_port, email.timeout_seconds)))
    elif email.mode != "off" and (s.smtp_user or s.smtp_app_password):
        log.warning("Email is half configured: set both SMTP_USER and SMTP_APP_PASSWORD in .env")
    return out


def make_notifier(cfg: Config, http: PoliteClient, console: bool) -> Notifier:
    if console:
        return ConsoleNotifier()
    chain = channels(cfg, http)
    if not chain:
        log.warning("No DISCORD_WEBHOOK_URL or SMTP credentials set; printing alerts to the console instead")
        return ConsoleNotifier()
    if len(chain) == 1:
        return chain[0][1]
    return NotifierChain(chain, mode="both" if cfg.notify.email.mode == "both" else "fallback")


@asynccontextmanager
async def open_pipeline(cfg: Config, *, console: bool = False, use_llm: bool = True) -> AsyncIterator[Pipeline]:
    db = DB(cfg.db_path)
    scorer, off_reason = make_scorer(cfg, db) if use_llm else (None, "scoring skipped")
    if off_reason and use_llm:
        level = logging.INFO if cfg.llm.backend == "off" else logging.WARNING
        log.log(level, "Scoring disabled: %s. Rule matches will be sent as [unscored].", off_reason)
    elif scorer:
        log.info("Scoring with %s", scorer.name)
    try:
        async with PoliteClient(
            user_agent=cfg.http.user_agent,
            timeout=cfg.http.timeout_seconds,
            per_host_interval=cfg.http.per_host_min_interval_seconds,
            max_retries=cfg.http.max_retries,
            max_concurrency=cfg.http.max_concurrency,
        ) as http:
            yield Pipeline(
                cfg=cfg,
                db=db,
                http=http,
                sources=build_sources(cfg),
                notifier=make_notifier(cfg, http, console),
                scorer=scorer,
                scorer_off_reason=off_reason,
            )
    finally:
        if scorer is not None:
            await scorer.aclose()
        db.close()
