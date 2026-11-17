"""Lever public postings API: https://github.com/lever/postings-api"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..http import PoliteClient, json_body
from ..models import Job
from ..text import html_to_text
from . import register
from .base import Source

HOSTS = {"global": "https://api.lever.co", "eu": "https://api.eu.lever.co"}


@register
class LeverSource(Source):
    kind = "lever"

    def __init__(self, slug: str, company: str, region: str = "global"):
        super().__init__(name=slug, company=company)
        self.slug = slug
        self.base = HOSTS[region]

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> LeverSource:
        return cls(slug=entry["slug"], company=entry.get("name") or entry["slug"], region=entry.get("region", "global"))

    async def fetch(self, http: PoliteClient) -> list[Job]:
        # The list endpoint already includes full descriptions, so no detail requests are needed.
        resp = await http.get(f"{self.base}/v0/postings/{self.slug}", params={"mode": "json"})
        return parse_postings(json_body(resp), self.feed, self.company)


def _description(p: dict[str, Any]) -> str:
    parts = [p.get("descriptionPlain") or ""]
    for lst in p.get("lists") or []:
        parts.append(f"{lst.get('text', '')}\n{html_to_text(lst.get('content') or '')}")
    parts.append(p.get("additionalPlain") or "")
    return "\n\n".join(s.strip() for s in parts if s and s.strip())


def parse_posting(p: dict[str, Any], feed: str, company: str) -> Job:
    cats = p.get("categories") or {}
    locations = cats.get("allLocations") or ([cats["location"]] if cats.get("location") else [])
    location = "; ".join(locations)
    if p.get("country"):
        location = f"{location} ({p['country']})" if location else p["country"]
    workplace = p.get("workplaceType")
    extra = {k: str(v) for k, v in (("Team", cats.get("team")), ("Commitment", cats.get("commitment")),
                                     ("Workplace", workplace)) if v and v != "unspecified"}
    created = p.get("createdAt")
    return Job(
        source="lever",
        job_id=p["id"],
        feed=feed,
        title=p["text"].strip(),
        company=company,
        location=location,
        url=p["hostedUrl"],
        description=_description(p),
        posted_at=datetime.fromtimestamp(created / 1000, tz=timezone.utc) if created else None,
        remote=True if workplace == "remote" else None,
        extra=extra,
    )


def parse_postings(payload: list[dict[str, Any]], feed: str, company: str) -> list[Job]:
    return [parse_posting(p, feed, company) for p in payload]
