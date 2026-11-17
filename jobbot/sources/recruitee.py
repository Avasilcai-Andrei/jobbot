"""Recruitee (Tellent) Careers Site API: https://docs.recruitee.com/reference/offers

GET https://{company}.recruitee.com/api/offers/ is public (no auth) and returns every published offer with its
description and requirements, so no detail requests are needed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..http import PoliteClient, json_body
from ..models import Job
from ..text import html_to_text
from . import register
from .base import Source


@register
class RecruiteeSource(Source):
    kind = "recruitee"

    def __init__(self, slug: str, company: str):
        super().__init__(name=slug, company=company)
        self.slug = slug

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> RecruiteeSource:
        return cls(slug=entry["slug"], company=entry.get("name") or entry["slug"])

    async def fetch(self, http: PoliteClient) -> list[Job]:
        resp = await http.get(f"https://{self.slug}.recruitee.com/api/offers/")
        return parse_offers(json_body(resp), self.feed, self.company)


def _when(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.strptime(s.removesuffix(" UTC"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _salary(o: dict[str, Any]) -> str | None:
    s = o.get("salary") or {}
    if not (s.get("min") or s.get("max")):
        return None
    span = "–".join(str(v) for v in (s.get("min"), s.get("max")) if v)
    return f"{span} {s.get('currency') or ''}/{s.get('period') or ''}".strip(" /")


def parse_offer(o: dict[str, Any], feed: str, company: str) -> Job:
    places = [", ".join(p for p in (loc.get("city"), loc.get("country")) if p) for loc in o.get("locations") or []]
    places = [p for p in places if p] or [o.get("location") or ""]
    workplace = "Remote" if o.get("remote") else "Hybrid" if o.get("hybrid") else "On-site" if o.get("on_site") else None
    parts = [html_to_text(o.get("description") or "")]
    if o.get("requirements"):
        parts.append("Requirements\n" + html_to_text(o["requirements"]))
    extra = {k: str(v) for k, v in (("Department", o.get("department")),
                                    ("Employment type", o.get("employment_type_code")),
                                    ("Experience", o.get("experience_code")),
                                    ("Education", o.get("education_code")),
                                    ("Workplace", workplace),
                                    ("Salary", _salary(o))) if v}
    return Job(
        source="recruitee",
        job_id=str(o["id"]),
        feed=feed,
        title=o["title"].strip(),
        company=company,
        location="; ".join(places),
        url=o.get("careers_url") or "",
        description="\n\n".join(p for p in parts if p),
        posted_at=_when(o.get("published_at")),
        remote=True if o.get("remote") else None,
        extra=extra,
    )


def parse_offers(payload: dict[str, Any], feed: str, company: str) -> list[Job]:
    return [parse_offer(o, feed, company) for o in payload.get("offers") or []]
