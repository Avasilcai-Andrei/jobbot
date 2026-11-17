import asyncio
import json

import httpx

from jobbot.calibrate import (LabelledJob, ModelRun, Outcome, auc, job_to_snapshot, load_labelled, report,
                              rule_verdicts, score_with_model, snapshot_to_job, suggest_threshold)

from .conftest import make_config, make_job
from .test_scoring import FakeOllama


def test_auc():
    assert auc([80, 90], [10, 20]) == 1.0
    assert auc([10], [80]) == 0.0
    assert auc([50], [50]) == 0.5
    assert auc([], [1]) is None


def test_suggest_threshold_prefers_recall_within_best_range():
    t, lo, hi = suggest_threshold(yes=[72, 82, 85], no=[0, 10, 40, 45])
    assert (lo, hi) == (46, 72) and lo <= t <= hi and t < 60
    # overlapping scores: best trade-off still found
    t, lo, hi = suggest_threshold(yes=[30, 50, 82], no=[10, 20, 60])
    assert lo <= 30  # catching every yes beats dropping one
    assert suggest_threshold([], [1]) is None


def write_set(tmp_path, labels_yaml, jobs):
    cal = tmp_path / "calibration"
    (cal / "jobs").mkdir(parents=True)
    (cal / "labels.yaml").write_text(labels_yaml, encoding="utf-8")
    for jid, job in jobs.items():
        (cal / "jobs" / f"{jid}.json").write_text(json.dumps(job_to_snapshot(job)), encoding="utf-8")
    return cal


def test_load_labelled_handles_yaml_booleans_and_bad_entries(tmp_path):
    cal = write_set(tmp_path, """
jobs:
  - {id: a, label: yes}
  - {id: b, label: no}
  - {id: c, label: "maybe", note: hmm}
  - {id: d, label: perhaps}
  - {id: missing, label: yes}
""", {k: make_job(k) for k in "abcd"})
    items = load_labelled(cal)
    assert [(i.id, i.label) for i in items] == [("a", "yes"), ("b", "no"), ("c", "maybe")]
    assert items[2].note == "hmm"


def test_snapshot_roundtrip():
    job = make_job("7", extra={"Workplace": "Hybrid", "_internal": "x"})
    back = snapshot_to_job(job_to_snapshot(job, original_location="Paris"))
    assert back.title == job.title and back.posted_at == job.posted_at
    assert back.extra == {"Workplace": "Hybrid"}


def test_score_with_model_and_report(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    items = [LabelledJob("a", "yes", make_job("a")), LabelledJob("b", "no", make_job("b", title="Java Dev"))]
    replies = [json.dumps({"score": 88, "reason": "fits", "missing_requirements": []}),
               json.dumps({"score": 12, "reason": "Java", "missing_requirements": ["Java"]})]
    fake = FakeOllama(replies=replies)

    from jobbot.scoring.ollama import OllamaScorer
    real_init = OllamaScorer.__init__
    monkeypatch.setattr(OllamaScorer, "__init__", lambda self, *a, **kw: real_init(
        self, *a, **{**kw, "transport": httpx.MockTransport(fake)}))
    run = asyncio.run(score_with_model(cfg, "CV", items, "qwen3.5:9b"))
    assert run.name == "ollama:qwen3.5:9b" and run.unavailable is None
    assert run.outcomes["a"].score == 88 and run.outcomes["b"].score == 12
    text = report(items, [run, ModelRun("ollama:other", unavailable="not pulled")], current_threshold=65)
    assert "separation (AUC yes vs no): 1.00" in text
    assert "suggested threshold" in text and "not run - not pulled" in text


def test_report_counts_errors(tmp_path):
    items = [LabelledJob("a", "yes", make_job("a")), LabelledJob("b", "no", make_job("b"))]
    run = ModelRun("ollama:m", outcomes={"a": Outcome(70, 1.0), "b": Outcome(None, 2.0, error="malformed output twice")})
    text = report(items, [run], 65)
    assert "failed/malformed: 1 of 2" in text and "ERR" in text


def test_report_applies_rules_before_threshold(tmp_path):
    cfg = make_config(tmp_path)
    items = [LabelledJob("a", "yes", make_job("a")),
             LabelledJob("b", "no", make_job("b", description="- Fluent in English and German")),
             LabelledJob("c", "no", make_job("c"))]
    dropped = rule_verdicts(cfg, items)
    assert dropped["a"] is None and dropped["b"].startswith("requires German")
    run = ModelRun("ollama:m", outcomes={"a": Outcome(80, 1.0), "b": Outcome(90, 1.0), "c": Outcome(20, 1.0)})
    text = report(items, [run], 65, dropped)
    assert "current threshold 65: 1/1 yes, 1/2 no alerted" in text
    assert "rules + threshold 65: 1/1 yes, 0/0 maybe, 0/2 no alerted" in text
    assert "drop: requires German" in text
