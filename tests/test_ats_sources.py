import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest

from jobbot.filters import RuleFilter
from jobbot.sources import build_sources
from jobbot.sources.lever import LeverSource, parse_postings as lever_parse
from jobbot.sources.smartrecruiters import SmartRecruitersSource, apply_details as sr_details
from jobbot.sources.smartrecruiters import parse_postings as sr_parse
from jobbot.sources.workday import WorkdaySource, apply_details as wd_details, find_country_facet, find_location_values
from jobbot.sources.workday import parse_jobs as wd_parse

from .conftest import load_fixture, make_config, mock_client

# --- Lever ------------------------------------------------------------------


def test_lever_parse_fixture():
    jobs = lever_parse(load_fixture("lever/matillion_postings.json"), "lever:matillion", "Matillion")
    assert len(jobs) == 4
    j = jobs[0]
    assert j.source == "lever" and j.feed == "lever:matillion" and j.company == "Matillion"
    assert j.url.startswith("https://jobs.lever.co/matillion/")
    assert j.location == "Manchester (GB)"
    assert j.posted_at == datetime.fromtimestamp(1791299280299 / 1000, tz=timezone.utc)
    assert j.remote is True and j.extra["Workplace"] == "remote"
    assert j.description and "<" not in j.description  # list endpoint carries the full text


def test_lever_fetch_hosts():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=load_fixture("lever/matillion_postings.json"))

    async def go():
        async with mock_client(handler) as http:
            await LeverSource("matillion", "Matillion").fetch(http)
            await LeverSource("acme", "Acme", region="eu").fetch(http)

    asyncio.run(go())
    assert seen == ["https://api.lever.co/v0/postings/matillion?mode=json",
                    "https://api.eu.lever.co/v0/postings/acme?mode=json"]


# --- SmartRecruiters ------------------------------------------------------------

SR_LIST = "smartrecruiters/endava_postings.json"
SR_DETAIL = "smartrecruiters/endava_posting_744000158502339.json"


def test_smartrecruiters_parse_fixture():
    jobs = sr_parse(load_fixture(SR_LIST), "smartrecruiters:Endava", "Endava", "Endava")
    assert len(jobs) == 6
    j = jobs[0]
    assert j.job_id == "744000158502339" and j.title == "DevOps Lead with Telco experience"
    assert j.location == "Cluj-Napoca, CJ, Romania"
    assert j.posted_at is not None and j.posted_at.tzinfo is not None
    assert j.extra["Experience level"] == "Mid-Senior Level"
    assert j.extra["Workplace"].startswith("Hybrid")
    assert j.description is None and j.remote is None
    j = sr_details(j, load_fixture(SR_DETAIL))
    assert j.description and "<" not in j.description
    assert j.url == "https://jobs.smartrecruiters.com/Endava/744000158502339-devops-lead-with-telco-experience"


def test_smartrecruiters_pages_until_total():
    calls = []
    page = load_fixture(SR_LIST)

    def handler(request):
        calls.append(dict(request.url.params))
        return httpx.Response(200, json={**page, "totalFound": 150})

    async def go():
        async with mock_client(handler) as http:
            return await SmartRecruitersSource("Endava", "Endava").fetch(http)

    jobs = asyncio.run(go())
    assert [c["offset"] for c in calls] == ["0", "100"]
    assert all(c["country"] == "ro" and c["limit"] == "100" for c in calls)
    assert len(jobs) == 12


# --- Workday ------------------------------------------------------------------


def test_workday_country_facet_nested():
    facets = load_fixture("workday/wk_unfiltered.json")["facets"]
    assert find_country_facet(facets, "Romania") == ("locationCountry", "f2e609fe92974a55a05fc1cdc2852122")
    assert find_country_facet(facets, "Atlantis") is None


def test_workday_location_fallback():
    facets = load_fixture("workday/ing_unfiltered.json")["facets"]
    assert find_country_facet(facets, "Romania") is None
    ids = find_location_values(facets, ["romania", "cluj", "bucharest"])
    assert len(ids) == 2  # "Bucharest - Dacia One" and "Cluj (Teodor Mihali)", not the Dutch/Belgian offices


def test_workday_parse_and_details():
    src = WorkdaySource("wk", 3, "External", "Wolters Kluwer")
    jobs = wd_parse(load_fixture("workday/wk_romania_jobs.json"), src)
    assert len(jobs) == 12
    j = jobs[0]
    assert j.job_id == "/job/ROU---Cluj-Napoca/Content-Management-Analyst_R0060279"
    assert j.url == "https://wk.wd3.myworkdayjobs.com/External/job/ROU---Cluj-Napoca/Content-Management-Analyst_R0060279"
    assert j.location == "ROU - Cluj-Napoca" and j.posted_at is None
    multi = next(j for j in jobs if j.location == "2 Locations")
    assert multi.extra["_location_incomplete"] == "1"
    j = wd_details(j, load_fixture("workday/wk_job_detail.json"))
    assert j.description and "<" not in j.description
    assert j.extra["Workplace"] == "Hybrid (8 days/month)" and j.extra["Posted date"] == "2026-10-29"
    assert j.remote is None


def test_workday_multi_location_passes_prefilter_until_details(cfg):
    rules = RuleFilter(cfg)
    src = WorkdaySource("wk", 3, "External", "Wolters Kluwer")
    multi = next(j for j in wd_parse(load_fixture("workday/wk_romania_jobs.json"), src) if j.location == "2 Locations")
    multi.title = "Junior Python Developer"
    assert rules.prefilter(multi).ok  # real locations unknown yet
    multi.description = "Python."
    multi.extra.pop("_location_incomplete")
    multi.location = "ESP - Madrid Castellana; ROU - Cluj-Napoca"
    assert rules.check(multi).location_rule.name == "Cluj-Napoca"
    multi.location = "ESP - Madrid Castellana; NLD - Amsterdam"
    assert not rules.check(multi).ok


def test_workday_fetch_applies_facet_and_pages():
    bodies = []
    unfiltered = load_fixture("workday/wk_unfiltered.json")
    page = load_fixture("workday/wk_romania_jobs.json")

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if not body["appliedFacets"]:
            return httpx.Response(200, json=unfiltered)
        return httpx.Response(200, json={**page, "total": 30})

    async def go():
        async with mock_client(handler) as http:
            src = WorkdaySource("wk", 3, "External", "Wolters Kluwer")
            first = await src.fetch(http)
            await src.fetch(http)  # facet id is cached: no second discovery request
            return first

    jobs = asyncio.run(go())
    assert len(jobs) == 24
    assert bodies[0]["appliedFacets"] == {}
    assert bodies[1]["appliedFacets"] == {"locationCountry": ["f2e609fe92974a55a05fc1cdc2852122"]}
    assert [b["offset"] for b in bodies[1:3]] == [0, 20] and all(b["limit"] == 20 for b in bodies[1:])
    assert len(bodies) == 5


def test_registry_builds_all_kinds(tmp_path):
    cfg = make_config(tmp_path, sources={
        "lever": {"companies": [{"slug": "acme", "region": "eu"}]},
        "smartrecruiters": {"companies": [{"id": "Endava"}]},
        "workday": {"companies": [{"tenant": "wk", "wd": 3, "site": "External", "name": "WK"}]},
        "greenhouse": {"enabled": False, "companies": [{"slug": "x"}]},
    })
    assert [s.feed for s in build_sources(cfg)] == ["lever:acme", "smartrecruiters:Endava", "workday:wk/External"]


# --- Workable -------------------------------------------------------------------

def test_workable_merges_locations_of_one_job(cfg):
    from jobbot.sources.workable import parse_widget
    jobs = parse_widget(load_fixture("workable/covergo_widget.json"), "workable:covergo", "CoverGo")
    by_id = {j.job_id: j for j in jobs}
    assert len(jobs) == 3 and set(by_id) >= {"7EE43D9E45", "46C4CB9C17"}  # 9 widget entries, 3 jobs
    pm = by_id["7EE43D9E45"]
    assert pm.location.split("; ") == ["Bucharest, Bucharest, Romania", "Cluj-Napoca, Cluj County, Romania",
                                       "Iași, Iași County, Romania"]
    assert pm.remote is True and pm.extra["Workplace"] == "Remote" and pm.extra["Posted date"]
    assert pm.url == "https://apply.workable.com/j/7EE43D9E45" and pm.posted_at is None
    assert pm.description and "<" not in pm.description  # details=true carries the full text
    rules = RuleFilter(cfg)
    assert rules.match_location(pm).name == "Cluj-Napoca"   # Cluj wins over Bucharest
    ba = by_id["46C4CB9C17"]
    assert ba.location.startswith("Kraków") and ba.location.endswith("Bucharest, Bucharest, Romania")


def test_workable_romania_remote_without_city(cfg):
    from jobbot.sources.workable import parse_widget
    ro, de = parse_widget(load_fixture("workable/flosum_widget.json"), "workable:flosum", "Flosum")
    assert ro.location == "Romania" and ro.remote is True
    assert de.location == "Germany"
    rules = RuleFilter(cfg)
    assert rules.match_location(ro).name == "Romania (remote)"
    assert rules.match_location(de) is None


def test_workable_fetch_url_and_config():
    from jobbot.sources.workable import WorkableSource
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=load_fixture("workable/flosum_widget.json"))

    async def go():
        async with mock_client(handler) as http:
            return await WorkableSource("flosum", "Flosum").fetch(http)

    jobs = asyncio.run(go())
    assert seen == ["https://apply.workable.com/api/v1/widget/accounts/flosum?details=true"]
    assert jobs[0].feed == "workable:flosum" and jobs[0].source == "workable"


# --- Teamtailor -----------------------------------------------------------------

def _rss(name):
    from .conftest import FIXTURES
    return (FIXTURES / "teamtailor" / f"{name}_jobs.rss").read_text(encoding="utf-8")


def test_teamtailor_parse_fixture(cfg):
    from jobbot.sources.teamtailor import parse_rss
    jobs = parse_rss(_rss("zitec"), "teamtailor:zitec", "Zitec")
    assert len(jobs) == 4
    sales = next(j for j in jobs if j.title == "Sales Consultant - Mirro")
    assert sales.location == "Bucharest, Romania" and sales.extra["Department"] == "Mirro"
    assert sales.job_id == "3ec36303-a889-46ce-beb8-155f32301285"
    assert sales.url == "https://careers.makeit.software/jobs/6503662-sales-consultant-mirro"  # own domain
    assert sales.posted_at is not None and sales.posted_at.utcoffset() is not None   # pubDate is precise
    assert sales.description and "<" not in sales.description and "&lt;" not in sales.description
    assert [j.location for j in jobs].count("") == 2           # Zitec leaves some jobs without a location
    rules = RuleFilter(cfg)
    assert rules.match_location(sales).name == "Bucharest"
    # "Romania (remote)" in the location name is enough for the remote rule, although remoteStatus says none
    (pool,) = parse_rss(_rss("zenitech"), "teamtailor:zenitech", "Zenitech")
    assert "Romania (remote)" in pool.location and pool.remote is None
    assert rules.match_location(pool).name == "Romania (remote)"


def test_teamtailor_default_location():
    from jobbot.sources.teamtailor import parse_rss
    jobs = parse_rss(_rss("zitec"), "teamtailor:zitec", "Zitec", default_location="Bucharest, Romania")
    assert all(j.location for j in jobs) and jobs[0].location == "Bucharest, Romania"


def test_teamtailor_fetch_pages_and_site_forms():
    from jobbot.sources.teamtailor import TeamtailorSource
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text=_rss("zitec"))

    async def go():
        async with mock_client(handler) as http:
            await TeamtailorSource("zitec", "Zitec").fetch(http)
            await TeamtailorSource("https://careers.example.com/", "Example").fetch(http)

    asyncio.run(go())
    assert seen == ["https://zitec.teamtailor.com/jobs.rss?per_page=100&offset=0",
                    "https://careers.example.com/jobs.rss?per_page=100&offset=0"]  # 4 < 100: one page each
    assert TeamtailorSource("zitec", "Zitec").feed == "teamtailor:zitec"


# --- Recruitee ------------------------------------------------------------------

def test_recruitee_parse_fixture(cfg):
    from jobbot.sources.recruitee import parse_offers
    (job,) = parse_offers(load_fixture("recruitee/ascendia_offers.json"), "recruitee:ascendia", "Ascendia")
    assert job.job_id == "2137642" and job.title == "Marketing Specialist"
    assert job.location == "Bucharest, Romania" and job.remote is True and job.extra["Workplace"] == "Remote"
    assert job.posted_at == datetime(2025, 5, 14, 14, 37, 9, tzinfo=timezone.utc)
    assert job.extra["Salary"] == "5000–7000 RON/month" and job.extra["Experience"] == "experienced"
    assert job.url == "https://ascendia.recruitee.com/o/marketing-specialist"
    assert "Requirements\n" in job.description and "<" not in job.description
    assert RuleFilter(cfg).match_location(job).name == "Romania (remote)"  # remote beats the Bucharest rule


def test_recruitee_fetch_url():
    from jobbot.sources.recruitee import RecruiteeSource
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=load_fixture("recruitee/ascendia_offers.json"))

    async def go():
        async with mock_client(handler) as http:
            return await RecruiteeSource("ascendia", "Ascendia").fetch(http)

    assert len(asyncio.run(go())) == 1 and seen == ["https://ascendia.recruitee.com/api/offers/"]


# --- Personio -------------------------------------------------------------------

def _personio(name):
    from .conftest import FIXTURES
    return (FIXTURES / "personio" / f"{name}.xml").read_text(encoding="utf-8")


def test_personio_parse_fixture():
    from jobbot.sources.personio import parse_xml
    jobs = parse_xml(_personio("tazz_default"), "personio:tazz", "TAZZ", "https://tazz.jobs.personio.com")
    assert len(jobs) == 3
    j = jobs[1]
    assert j.job_id == "2846366" and j.location == "TAZZ; TAZZ Oerlikon"   # office + additionalOffices
    assert j.url == "https://tazz.jobs.personio.com/job/2846366?language=en"
    assert j.posted_at == datetime(2026, 10, 16, 13, 38, 24, tzinfo=timezone.utc)
    assert j.extra["Seniority"] == "entry-level" and j.extra["Years of experience"] == "less than 1 year"
    assert j.description.startswith("Deine Aufgaben\n") and "<" not in j.description


def test_personio_falls_back_to_native_language_descriptions():
    from jobbot.sources.personio import PersonioSource
    seen = []

    def handler(request):
        seen.append(str(request.url))
        english = request.url.params.get("language") == "en"
        return httpx.Response(200, text=_personio("tazz_en" if english else "tazz_default"))

    async def go():
        async with mock_client(handler) as http:
            return await PersonioSource("tazz", "TAZZ", domain="com").fetch(http)

    jobs = asyncio.run(go())
    # the English feed has no descriptions for this German-only board, so the default feed fills them in
    assert seen == ["https://tazz.jobs.personio.com/xml?language=en", "https://tazz.jobs.personio.com/xml"]
    assert all(j.description.startswith("Deine Aufgaben") for j in jobs)
    assert PersonioSource("acme", "Acme", domain="de").base == "https://acme.jobs.personio.de"


def test_new_sources_build_from_config(tmp_path):
    cfg = make_config(tmp_path, sources={
        "workable": {"companies": [{"account": "garmin-cluj", "name": "Garmin Cluj"}]},
        "teamtailor": {"companies": [{"site": "zitec", "name": "Zitec", "default_location": "Bucharest, Romania"}]},
        "recruitee": {"companies": [{"slug": "ascendia", "name": "Ascendia"}]},
        "personio": {"companies": [{"slug": "acme", "name": "Acme", "domain": "de"}]},
    })
    feeds = {s.feed: s for s in build_sources(cfg)}
    assert set(feeds) == {"workable:garmin-cluj", "teamtailor:zitec", "recruitee:ascendia", "personio:acme"}
    assert feeds["teamtailor:zitec"].default_location == "Bucharest, Romania"
    assert feeds["workable:garmin-cluj"].company == "Garmin Cluj"


def test_workday_skips_stub_postings():
    # ING returned this exact entry among its real postings; it used to fail the whole feed.
    src = WorkdaySource("wk", 3, "External", "Wolters Kluwer")
    payload = load_fixture("workday/wk_romania_jobs.json")
    payload["jobPostings"].insert(1, {"bulletFields": ["REQ-10118936"]})
    assert len(wd_parse(payload, src)) == 12


MAINTENANCE = "<!DOCTYPE html>\n<html><head><title>Down for maintenance</title></head><body><p>Back soon</body></html>"


def test_teamtailor_and_personio_explain_a_non_xml_answer():
    from jobbot.http import BadResponse
    from jobbot.sources.personio import PersonioSource
    from jobbot.sources.teamtailor import TeamtailorSource

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=MAINTENANCE)

    async def go(source):
        async with mock_client(handler) as http:
            await source.fetch(http)

    for source, host in ((TeamtailorSource("zitec", "Zitec"), "zitec.teamtailor.com/jobs.rss"),
                         (PersonioSource("tazz", "TAZZ"), "tazz.jobs.personio.com/xml")):
        with pytest.raises(BadResponse) as exc:
            asyncio.run(go(source))
        msg = str(exc.value)
        assert msg.startswith("not XML (") and f"HTTP 200 from https://{host}" in msg
        assert "content-type 'text/html'" in msg and "Down for maintenance" in msg and exc.value.status_code == 200


def test_personio_native_language_refetch_reports_a_bad_answer_too():
    from jobbot.http import BadResponse
    from jobbot.sources.personio import PersonioSource

    def handler(request):
        if request.url.params.get("language") == "en":
            return httpx.Response(200, text=_personio("tazz_en"))  # empty descriptions -> triggers the refetch
        return httpx.Response(200, content=b"")

    async def go():
        async with mock_client(handler) as http:
            await PersonioSource("tazz", "TAZZ").fetch(http)

    with pytest.raises(BadResponse, match=r"not XML .*tazz\.jobs\.personio\.com/xml, content-type 'none', 0 bytes"):
        asyncio.run(go())


def test_workday_tenant_without_datacentre_number():
    """NXP moved to nxp.myworkdayjobs.com; `wd: null` in config selects that host form."""
    src = WorkdaySource.from_entry({"tenant": "nxp", "wd": None, "site": "careers", "name": "NXP"})
    assert src.host == "https://nxp.myworkdayjobs.com"
    assert src.api == "https://nxp.myworkdayjobs.com/wday/cxs/nxp/careers"
    assert WorkdaySource.from_entry({"tenant": "wk", "wd": 3, "site": "External"}).host == "https://wk.wd3.myworkdayjobs.com"
