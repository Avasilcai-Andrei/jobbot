import asyncio
from datetime import datetime, timezone

import httpx

from jobbot.sources import REGISTRY, build_sources
from jobbot.sources.greenhouse import GreenhouseSource, apply_details, parse_jobs

from .conftest import load_fixture, make_config, mock_client

LIST = "greenhouse/thoughtworks_jobs.json"
DETAIL = "greenhouse/thoughtworks_job_8258974.json"


def test_parse_list_fixture():
    jobs = parse_jobs(load_fixture(LIST), "greenhouse:thoughtworks", "Thoughtworks")
    assert len(jobs) == 7
    j = next(j for j in jobs if j.job_id == "8258974")
    assert j.source == "greenhouse"
    assert j.feed == "greenhouse:thoughtworks"
    assert j.title == "Consultant Systems Support Engineer"
    assert j.company == "Thoughtworks"
    assert j.location == "Bucharest - Dorobanti, Romania; Cluj, Romania; Iasi, Romania"
    assert j.url == "https://www.thoughtworks.com/careers/jobs/8258974?gh_jid=8258974"
    assert j.posted_at == datetime(2026, 10, 8, 10, 23, 24, tzinfo=timezone.utc)
    assert j.description is None  # list endpoint has no content; fetched later for survivors
    assert j.extra["No. of days in office"] == "Hybrid – 3 days on-site/week"
    assert j.remote is None


def test_titles_are_stripped():
    jobs = parse_jobs(load_fixture(LIST), "greenhouse:thoughtworks", "Thoughtworks")
    assert all(j.title == j.title.strip() for j in jobs)


def test_detail_fixture_gives_plain_text():
    jobs = parse_jobs(load_fixture(LIST), "greenhouse:thoughtworks", "Thoughtworks")
    j = apply_details(next(j for j in jobs if j.job_id == "8258974"), load_fixture(DETAIL))
    assert j.description
    assert "<" not in j.description and "&lt;" not in j.description and "&amp;" not in j.description
    assert "Consultant Systems Support Engineer" in j.description


def test_fetch_uses_public_board_api_without_network():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("/jobs"):
            return httpx.Response(200, json=load_fixture(LIST))
        return httpx.Response(200, json=load_fixture(DETAIL))

    async def go():
        async with mock_client(handler) as http:
            src = GreenhouseSource(slug="thoughtworks", company="Thoughtworks")
            jobs = await src.fetch(http)
            job = await src.fetch_details(http, jobs[0])
            return jobs, job

    jobs, job = asyncio.run(go())
    assert len(jobs) == 7 and job.description
    assert seen == [
        "https://boards-api.greenhouse.io/v1/boards/thoughtworks/jobs",
        f"https://boards-api.greenhouse.io/v1/boards/thoughtworks/jobs/{jobs[0].job_id}",
    ]


def test_registry_builds_one_source_per_company(tmp_path):
    cfg = make_config(tmp_path, sources={"greenhouse": {"companies": [
        {"slug": "thoughtworks", "name": "Thoughtworks"}, {"slug": "elastic"}]}})
    srcs = build_sources(cfg)
    assert "greenhouse" in REGISTRY
    assert [s.feed for s in srcs] == ["greenhouse:thoughtworks", "greenhouse:elastic"]
    assert srcs[1].company == "elastic"
