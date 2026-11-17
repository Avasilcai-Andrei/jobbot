import asyncio

import httpx
import pytest

from jobbot.report import percentile
from jobbot.text import crosspost_key, html_to_text, unescape_html_to_text

from .conftest import mock_client


def test_crosspost_key_normalises():
    a = crosspost_key("Acme Romania S.R.L.", "Junior Data Engineer (m/f/d) - Cluj")
    b = crosspost_key("ACME", "Junior  Data-Engineer, Remote")
    assert a == b == "acme|junior data engineer"
    # different stacks stay different
    assert crosspost_key("Acme", "Software Engineer (Python)") != crosspost_key("Acme", "Software Engineer (Java)")
    # diacritics
    assert crosspost_key("Firmă", "Inginer Date București") == crosspost_key("Firma", "Inginer date")


def test_html_to_text():
    assert html_to_text("<p>Hello&nbsp;<b>world</b></p><ul><li>one</li><li>two</li></ul>") == "Hello world\n\n- one\n\n- two"
    assert unescape_html_to_text("&lt;p&gt;A &amp;amp; B&lt;/p&gt;") == "A & B"


def test_percentile():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(vals, 50) == 5.5
    assert percentile(vals, 0) == 1 and percentile(vals, 100) == 10
    assert percentile([7], 95) == 7


def test_http_sends_user_agent_and_retries_with_retry_after():
    calls = []
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["User-Agent"])
        if len(calls) < 3:
            return httpx.Response(429 if len(calls) == 1 else 503, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"ok": True})

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async def go():
        client = mock_client(handler)
        client._sleep = fake_sleep
        async with client:
            return await client.get("https://example.com/x")

    resp = asyncio.run(go())
    assert resp.json() == {"ok": True}
    assert calls == ["jobbot-test/0"] * 3
    assert sleeps == [7.0, 7.0]


def test_http_gives_up_and_does_not_retry_404():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404 if request.url.path == "/missing" else 500)

    async def go(path):
        async with mock_client(handler, max_retries=2) as client:
            return await client.get(f"https://example.com{path}")

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(go("/missing"))
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(go("/broken"))
    assert len(calls) == 3  # first try + 2 retries


def test_http_rate_limits_per_host():
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    async def go():
        client = mock_client(lambda r: httpx.Response(200), per_host_interval=5.0)
        client._sleep = fake_sleep
        async with client:
            await client.get("https://a.example/1")
            await client.get("https://a.example/2")
            await client.get("https://b.example/1")

    asyncio.run(go())
    assert len(sleeps) == 1 and 4.0 < sleeps[0] <= 5.0  # only the second hit on a.example waited


def test_clean_cv_text_removes_contact_details():
    from jobbot.cv import _glued_words, clean_cv_text
    raw = "Jane Doe\nCluj-Napoca | +40 712 345 678 | jane.doe@example.com\n\n\n\nSkills:   Python,  SQL"
    out = clean_cv_text(raw)
    assert "712" not in out and "example.com" not in out
    assert "[phone]" in out and "[email]" in out and "Skills: Python, SQL" in out
    assert "2024 – 2027" in clean_cv_text("BSc 2024 – 2027 (expected)")  # year ranges are not phone numbers
    assert _glued_words("Finalyearcomputersciencestudentat university") == 1


def test_json_body_explains_a_non_json_answer():
    from jobbot.http import BadResponse, json_body

    page = "<!DOCTYPE html>\n<html><head><title>Down for maintenance</title></head>\n<body>" + "x" * 500
    def handler(request):
        if request.url.path == "/wday/cxs/wk/External/jobs":
            return httpx.Response(302, headers={"Location": "https://wk.wd3.myworkdayjobs.com/maintenance"})
        if request.url.path == "/maintenance":
            return httpx.Response(200, headers={"content-type": "text/html; charset=UTF-8"}, text=page)
        return httpx.Response(200, json={"ok": 1})

    async def go(path):
        async with mock_client(handler) as client:
            return json_body(await client.post(f"https://wk.wd3.myworkdayjobs.com{path}?secret=1", json={}))

    assert asyncio.run(go("/fine")) == {"ok": 1}
    with pytest.raises(BadResponse) as exc:
        asyncio.run(go("/wday/cxs/wk/External/jobs"))
    msg = str(exc.value)
    assert msg.startswith("not JSON (Expecting value: line 1 column 1 (char 0)): HTTP 200 from "
                          "https://wk.wd3.myworkdayjobs.com/maintenance (redirected from "
                          "https://wk.wd3.myworkdayjobs.com/wday/cxs/wk/External/jobs, HTTP 302)")
    assert "content-type 'text/html; charset=UTF-8'" in msg and f"{len(page)} bytes" in msg
    assert "<title>Down for maintenance</title>" in msg and "secret" not in msg
    assert len(exc.value.preview) == 300 and "\n" not in exc.value.preview
    assert exc.value.status_code == 200


def test_json_body_on_an_empty_body():
    from jobbot.http import BadResponse, json_body

    resp = httpx.Response(200, content=b"", request=httpx.Request("GET", "https://x.example/api"))
    with pytest.raises(BadResponse, match=r"HTTP 200 from https://x.example/api, content-type 'none', 0 bytes, body: ''"):
        json_body(resp)
