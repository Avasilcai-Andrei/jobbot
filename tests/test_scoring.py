import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from jobbot.db import DB
from jobbot.notify import Alert, DiscordNotifier, NotifyError, discord_payload
from jobbot.scoring import ScoreFailed, ScoreUnavailable, make_scorer
from jobbot.scoring.base import TRUNCATION_NOTE, approx_tokens, parse_result, truncate_to_tokens
from jobbot.scoring.anthropic_api import AnthropicScorer
from jobbot.scoring.ollama import OllamaScorer

from .conftest import make_config, make_job, mock_client

GOOD = {"score": 81, "reason": "Good fit for a junior data role.", "missing_requirements": ["Airflow"]}


class FakeOllama:
    """Stands in for the Ollama server: /api/tags and /api/chat."""

    def __init__(self, replies=None, models=("qwen3.5:9b", "ministral-3:8b"), size_vram=5_762_732_522):
        self.replies = list(replies or [json.dumps(GOOD)])
        self.models = models
        self.size_vram = size_vram  # of a 5_762_732_522-byte model in /api/ps
        self.chats = []
        self.ps_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m, "model": m} for m in self.models]})
        if request.url.path == "/api/ps":
            self.ps_calls += 1
            return httpx.Response(200, json={"models": [{"name": m, "model": m, "size": 5_762_732_522,
                                                         "size_vram": self.size_vram} for m in self.models]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            self.chats.append(body)
            if body["model"] not in self.models:
                return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
            content = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            return httpx.Response(200, json={"model": body["model"], "message": {"role": "assistant", "content": content},
                                             "done": True, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 60})
        return httpx.Response(404)


def ollama(cfg, fake, model=None, db=None):
    return OllamaScorer(cfg, "CV: Python, SQL, pandas, Airflow course", db or DB(":memory:"),
                        model=model, transport=httpx.MockTransport(fake))


def run(coro):
    return asyncio.run(coro)


# --- Ollama ---------------------------------------------------------------

def test_ollama_request_shape(cfg):
    fake = FakeOllama()
    s = ollama(cfg, fake)
    assert run(s.check_available()) is None
    r = run(s.score(make_job(), "Cluj-Napoca"))
    assert r.score == 81 and r.missing_requirements == ["Airflow"]
    body = fake.chats[0]
    assert body["model"] == "qwen3.5:9b" and body["stream"] is False
    assert body["format"]["required"] == ["score", "reason", "missing_requirements"]
    assert body["options"]["temperature"] == 0 and body["options"]["num_ctx"] == 8192
    assert body["think"] is False
    assert body["messages"][0]["role"] == "system" and "CV: Python, SQL" in body["messages"][0]["content"]
    assert "Junior Data Engineer" in body["messages"][1]["content"]


def test_think_param_omitted_when_null(tmp_path):
    cfg = make_config(tmp_path, llm={"ollama": {"think": None}})
    fake = FakeOllama()
    run(ollama(cfg, fake).score(make_job()))
    assert "think" not in fake.chats[0]


def test_cache_hit_and_model_switch_rescores(cfg):
    db = DB(":memory:")
    fake = FakeOllama()
    run(ollama(cfg, fake, db=db).score(make_job()))
    run(ollama(cfg, fake, db=db).score(make_job()))
    assert len(fake.chats) == 1  # cached
    run(ollama(cfg, fake, model="ministral-3:8b", db=db).score(make_job()))
    assert len(fake.chats) == 2 and fake.chats[1]["model"] == "ministral-3:8b"


def test_malformed_output_retried_once_with_correction(cfg):
    fake = FakeOllama(replies=['{"score": "high", "reason": "x"}', json.dumps(GOOD)])
    r = run(ollama(cfg, fake).score(make_job()))
    assert r.score == 81 and len(fake.chats) == 2
    retry_msgs = fake.chats[1]["messages"]
    assert retry_msgs[-2]["role"] == "assistant" and retry_msgs[-1]["role"] == "user"
    assert "not valid" in retry_msgs[-1]["content"]


@pytest.mark.parametrize("bad", ["not json at all", '{"score": 150, "reason": "x", "missing_requirements": []}',
                                 '{"score": 70, "reason": "", "missing_requirements": []}'])
def test_malformed_twice_fails(cfg, bad):
    fake = FakeOllama(replies=[bad])
    with pytest.raises(ScoreFailed):
        run(ollama(cfg, fake).score(make_job()))
    assert len(fake.chats) == 2


def test_unreachable_and_missing_model(cfg):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    s = ollama(cfg, refuse)
    assert "not reachable" in run(s.check_available())
    with pytest.raises(ScoreUnavailable):
        run(s.score(make_job()))
    s = ollama(cfg, FakeOllama(models=("llama3.1:8b",)))
    assert "ollama pull qwen3.5:9b" in run(s.check_available())
    with pytest.raises(ScoreUnavailable):
        run(s.score(make_job()))


@pytest.mark.parametrize("size_vram, processor", [(5_762_732_522, "100% GPU"), (0, "100% CPU"),
                                                  (2_881_366_261, "50%/50% CPU/GPU")])
def test_processor_check_warns_unless_fully_on_gpu(cfg, caplog, size_vram, processor):
    fake = FakeOllama(size_vram=size_vram)
    s = ollama(cfg, fake)
    assert run(s.check_processor()) == processor
    assert ("slow" in caplog.text) == (processor != "100% GPU")


def test_processor_checked_once_per_run_after_first_answer(cfg):
    fake = FakeOllama(size_vram=0)
    s = ollama(cfg, fake)
    jobs = iter(make_job(job_id=str(i), description=f"Python and SQL, posting {i}.") for i in range(10))

    async def one_run(n):
        await s.check_available()
        for _ in range(n):
            await s.score(next(jobs))  # distinct text, so no cache hits

    run(one_run(3))
    assert fake.ps_calls == 1
    run(one_run(2))
    assert fake.ps_calls == 2
    run(one_run(0))  # nothing scored: nothing loaded, nothing to check
    assert fake.ps_calls == 2


def test_processor_check_ignores_unknown_answers(cfg):
    def old_server(request):
        return httpx.Response(404) if request.url.path == "/api/ps" else FakeOllama()(request)

    assert run(ollama(cfg, old_server).check_processor()) is None
    assert run(ollama(cfg, FakeOllama(models=("llama3.1:8b",))).check_processor()) is None


# --- truncation -----------------------------------------------------------

def test_truncate_prefers_boundaries():
    text = ("First paragraph sentence. " * 40 + "\n\n") * 10
    out, cut = truncate_to_tokens(text, 300)
    assert cut and out.endswith(TRUNCATION_NOTE)
    assert approx_tokens(out) <= 301
    assert out[: -len(TRUNCATION_NOTE)].endswith(".")
    assert truncate_to_tokens("short", 300) == ("short", False)


def test_description_budget_respects_context(tmp_path):
    long_desc = "Python and SQL work. " * 5000  # ~30k tokens
    small = make_config(tmp_path, llm={"ollama": {"num_ctx": 4096}, "max_description_tokens": 3000})
    fake = FakeOllama()
    s = ollama(small, fake)
    run(s.score(make_job(description=long_desc)))
    sent = fake.chats[0]["messages"]
    total = approx_tokens(sent[0]["content"]) + approx_tokens(sent[1]["content"])
    assert TRUNCATION_NOTE.strip() in sent[1]["content"]
    assert total <= 4096 - small.llm.reserve_output_tokens
    # with a big context the configured cap applies instead
    big = make_config(tmp_path, llm={"ollama": {"num_ctx": 32768}, "max_description_tokens": 3000})
    fake2 = FakeOllama()
    run(ollama(big, fake2).score(make_job(description=long_desc)))
    assert 2800 < approx_tokens(fake2.chats[0]["messages"][1]["content"]) < 3300


def test_parse_result_tolerates_code_fence():
    assert parse_result('```json\n{"score": 70, "reason": "ok", "missing_requirements": []}\n```').score == 70


# --- factory & Anthropic ----------------------------------------------------

def test_make_scorer(tmp_path):
    db = DB(":memory:")
    s, why = make_scorer(make_config(tmp_path, llm={"backend": "off"}), db)
    assert s is None and "off" in why
    s, why = make_scorer(make_config(tmp_path), db)  # no cv.pdf in tmp_path
    assert s is None and "no CV" in why
    s, why = make_scorer(make_config(tmp_path), db, cv_text="cv", model="ministral-3:8b")
    assert why is None and s.name == "ollama:ministral-3:8b"
    s, why = make_scorer(make_config(tmp_path, llm={"backend": "anthropic"}), db, cv_text="cv")
    assert s.name == "anthropic:claude-haiku-5-5" and run(s.check_available()) == "ANTHROPIC_API_KEY is not set"


class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason, _request_id="req_test",
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=json.dumps(GOOD))])


def test_anthropic_request_shape(cfg):
    client = SimpleNamespace(messages=FakeMessages())
    s = AnthropicScorer(cfg, "CV text", DB(":memory:"), client=client)
    assert run(s.score(make_job())).score == 81
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5" and call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    s = AnthropicScorer(cfg, "CV text", DB(":memory:"), client=SimpleNamespace(messages=FakeMessages("refusal")))
    with pytest.raises(ScoreFailed):
        run(s.score(make_job()))


# --- Discord ----------------------------------------------------------------

def _alert(**kw):
    job = make_job()
    defaults = dict(job=job, score=72, reason="Solid match.", missing=["Spark"], location_rule="Bucharest",
                    low_priority=True, first_seen_at=job.posted_at + timedelta(minutes=12))
    defaults.update(kw)
    return Alert(**defaults)


def test_discord_payload():
    e = discord_payload(_alert())["embeds"][0]
    assert e["title"].startswith("⚠️ Junior Data Engineer")
    fields = {f["name"]: f["value"] for f in e["fields"]}
    assert fields["Company"] == "Acme" and fields["Score"] == "72/100"
    assert "lower priority" in fields["Location"] and "Spark" in fields["Missing"]
    assert "detected after 12 min" in fields["Posted"]
    sid = _alert().job.short_id
    assert len(sid) == 6 and f"mark {sid} good" in fields["Mark"] and e["footer"]["text"].endswith(f"id {sid}")
    e = discord_payload(_alert(score=None, unscored="Ollama is not reachable", low_priority=False))["embeds"][0]
    assert e["title"].startswith("[unscored] Junior Data Engineer")
    assert {f["name"]: f["value"] for f in e["fields"]}["Score"] == "unscored"


def test_discord_notifier_hides_webhook_url_on_error():
    url = "https://discord.com/api/webhooks/123/SECRET-TOKEN"
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(204) if len(bodies) == 1 else httpx.Response(404, text='{"message": "Unknown Webhook"}')

    async def go():
        async with mock_client(handler) as http:
            n = DiscordNotifier(url, http)
            await n.send_alert(_alert())
            await n.send_text("hello")

    with pytest.raises(NotifyError) as exc:
        run(go())
    assert bodies[0]["embeds"][0]["fields"][2]["value"] == "72/100"
    assert "SECRET-TOKEN" not in str(exc.value) and "404" in str(exc.value)


def test_job_text_carries_todays_date_for_graduation_reasoning():
    from datetime import date
    from jobbot.scoring.base import format_job
    text = format_job(make_job(), "Cluj-Napoca", "desc", today=date(2026, 10, 30))
    assert text.splitlines()[1] == "Today's date: October 2026"


def test_employer_named_in_cv_is_hidden_from_the_model(cfg):
    from jobbot.scoring.base import EMPLOYER_PLACEHOLDER, hide_cv_employer
    cv = "working student, contoso srl (2023 - now). python, sql"
    job = make_job(title="Working Student Contoso Planning", company="Contoso",
                   description="Join Contoso Cluj Plant. Work LikeAContoso.", extra={"Dept": "Contoso Jucu"})
    hidden = hide_cv_employer(job, cv)
    assert hidden.company == EMPLOYER_PLACEHOLDER and "contoso" not in hidden.title.lower()
    assert "contoso" not in hidden.description.lower() and hidden.extra["Dept"] == "[Company] Jucu"
    assert job.company == "Contoso"                                         # the original is untouched
    assert hide_cv_employer(make_job(company="Endava"), cv).company == "Endava"
    # short names are replaced as whole words only
    ing = hide_cv_employer(make_job(company="ING", description="Hiring at ING: billing engineering"), "intern at ing")
    assert ing.description == "Hiring at [Company]: billing engineering"

    fake = FakeOllama()
    s = OllamaScorer(cfg, "CV: Contoso working student; Python, SQL", DB(":memory:"), transport=httpx.MockTransport(fake))
    run(s.score(job))
    sent = fake.chats[0]["messages"][1]["content"]
    assert "Contoso" not in sent and "Company: [Company]" in sent


def test_second_opinion_breaks_the_prompt_prefix_and_is_cached(cfg):
    from jobbot.scoring.base import SECOND_REVIEW_PREFIX
    reply = lambda s: json.dumps({**GOOD, "score": s})  # noqa: E731
    fake = FakeOllama(replies=[reply(60), reply(71), reply(99)])
    db = DB(":memory:")
    s = ollama(cfg, fake, db=db)
    r = run(s.score_stable(make_job(), None, threshold=65, margin=15))
    assert (r.score, r.scores) == (66, [60, 71])
    first, second = (c["messages"][0]["content"] for c in fake.chats)
    assert not first.startswith(SECOND_REVIEW_PREFIX) and second == SECOND_REVIEW_PREFIX + first
    # both opinions come from the cache next time: same answer, no requests
    assert run(ollama(cfg, fake, db=db).score_stable(make_job(), None, 65, 15)) == r and len(fake.chats) == 2
    # far from the threshold, or with the margin off: one request
    far = ollama(cfg, FakeOllama(replies=[reply(90)]))
    assert run(far.score_stable(make_job(), None, 65, 15)).scores is None
    off = FakeOllama(replies=[reply(64)])
    assert run(ollama(cfg, off).score_stable(make_job(), None, 65, None)).score == 64 and len(off.chats) == 1
