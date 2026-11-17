"""Greenhouse public Job Board API: https://developers.greenhouse.io/job-board.html"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..http import PoliteClient, json_body
from ..models import Job
from ..text import fold, unescape_html_to_text
from . import register
from .base import Source

API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"


@register
class GreenhouseSource(Source):
    kind = "greenhouse"

    def __init__(self, slug: str, company: str):
        super().__init__(name=slug, company=company)
        self.slug = slug

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> GreenhouseSource:
        return cls(slug=entry["slug"], company=entry.get("name") or entry["slug"])

    async def fetch(self, http: PoliteClient) -> list[Job]:
        resp = await http.get(API.format(slug=self.slug))
        return parse_jobs(json_body(resp), self.feed, self.company)

    async def fetch_details(self, http: PoliteClient, job: Job) -> Job:
        resp = await http.get(f"{API.format(slug=self.slug)}/{job.job_id}")
        return apply_details(job, json_body(resp))


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _metadata(item: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in item.get("metadata") or []:
        value = m.get("value")
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        if value not in (None, ""):
            out[m["name"]] = str(value)
    return out


def parse_job(item: dict[str, Any], feed: str, company: str) -> Job:
    meta = _metadata(item)
    location = (item.get("location") or {}).get("name", "")
    remote = True if any("remote" in fold(v) for v in [location, *meta.values()]) else None
    job = Job(
        source="greenhouse",
        job_id=str(item["id"]),
        feed=feed,
        title=item["title"].strip(),
        company=company or item.get("company_name", ""),
        location=location,
        url=item["absolute_url"],
        posted_at=_parse_dt(item.get("first_published")),
        remote=remote,
        extra=meta,
    )
    if item.get("content"):
        apply_details(job, item)
    return job


def parse_jobs(payload: dict[str, Any], feed: str, company: str) -> list[Job]:
    return [parse_job(item, feed, company) for item in payload.get("jobs", [])]


def apply_details(job: Job, payload: dict[str, Any]) -> Job:
    job.description = unescape_html_to_text(payload.get("content") or "")
    return job
