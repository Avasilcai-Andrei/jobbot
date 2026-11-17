"""SmartRecruiters public Posting API: https://developers.smartrecruiters.com/docs/posting-api"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..http import PoliteClient, json_body
from ..models import Job
from ..text import html_to_text
from . import register
from .base import Source

API = "https://api.smartrecruiters.com/v1/companies/{company}/postings"
PAGE = 100
MAX_PAGES = 20


@register
class SmartRecruitersSource(Source):
    kind = "smartrecruiters"

    def __init__(self, company_id: str, company: str, country: str | None = "ro"):
        super().__init__(name=company_id, company=company)
        self.company_id = company_id
        self.country = country

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> SmartRecruitersSource:
        return cls(company_id=entry["id"], company=entry.get("name") or entry["id"], country=entry.get("country", "ro"))

    async def fetch(self, http: PoliteClient) -> list[Job]:
        jobs: list[Job] = []
        params: dict[str, Any] = {"limit": PAGE}
        if self.country:
            params["country"] = self.country
        for page in range(MAX_PAGES):
            resp = await http.get(API.format(company=self.company_id), params={**params, "offset": page * PAGE})
            data = json_body(resp)
            jobs += parse_postings(data, self.feed, self.company, self.company_id)
            if (page + 1) * PAGE >= data.get("totalFound", 0):
                break
        return jobs

    async def fetch_details(self, http: PoliteClient, job: Job) -> Job:
        resp = await http.get(f"{API.format(company=self.company_id)}/{job.job_id}")
        return apply_details(job, json_body(resp))


def parse_posting(p: dict[str, Any], feed: str, company: str, company_id: str) -> Job:
    loc = p.get("location") or {}
    extra: dict[str, str] = {}
    if (lvl := (p.get("experienceLevel") or {}).get("label")):
        extra["Experience level"] = lvl
    if (emp := (p.get("typeOfEmployment") or {}).get("label")):
        extra["Employment"] = emp
    if loc.get("hybrid"):
        extra["Workplace"] = "Hybrid" + (f" ({loc['hybridDescription']})" if loc.get("hybridDescription") else "")
    elif loc.get("remote"):
        extra["Workplace"] = "Remote"
    location = loc.get("fullLocation") or ", ".join(filter(None, [loc.get("city"), loc.get("country")]))
    return Job(
        source="smartrecruiters",
        job_id=str(p["id"]),
        feed=feed,
        title=p["name"].strip(),
        company=company,
        location=location,
        url=f"https://jobs.smartrecruiters.com/{company_id}/{p['id']}",
        posted_at=datetime.fromisoformat(p["releasedDate"]) if p.get("releasedDate") else None,
        remote=True if loc.get("remote") else None,
        extra=extra,
    )


def parse_postings(payload: dict[str, Any], feed: str, company: str, company_id: str) -> list[Job]:
    return [parse_posting(p, feed, company, company_id) for p in payload.get("content", [])]


def apply_details(job: Job, payload: dict[str, Any]) -> Job:
    sections = ((payload.get("jobAd") or {}).get("sections") or {})
    parts = []
    for key in ("jobDescription", "qualifications", "additionalInformation", "companyDescription"):
        sec = sections.get(key) or {}
        if sec.get("text"):
            parts.append(f"{sec.get('title') or key}\n{html_to_text(sec['text'])}")
    job.description = "\n\n".join(parts)
    if payload.get("postingUrl"):
        job.url = payload["postingUrl"]
    return job
