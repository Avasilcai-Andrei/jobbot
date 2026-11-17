"""calibrate-import-marks: good = yes, bad = no, from the stored jobs (no network)."""
from __future__ import annotations

from datetime import datetime, timezone

import yaml

from jobbot.calibrate import import_marks, load_labelled
from jobbot.db import DB

from .conftest import make_config, make_job

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)


def test_import_marks(tmp_path):
    cfg = make_config(tmp_path)
    db = DB(":memory:")
    for i, title in (("1", "Data Engineer"), ("2", "Junior Sales Assistant"), ("3", "Unmarked one"), ("4", "Not alerted")):
        job = make_job(i, title=title, description=f"description {i}")
        db.insert_job(job, status="notified", first_seen_at=NOW, crosspost_key=f"k{i}")
        if i != "4":
            db.update_job(job.key, notified_at=NOW)
    db.set_mark(("fake", "1"), "good", "real match")
    db.set_mark(("fake", "2"), "bad", "sales")
    first = import_marks(cfg, db)
    assert sorted((label, outcome) for _, label, _, outcome in first) == [("no", "added"), ("yes", "added")]
    cal = tmp_path / "calibration"
    items = {x.job.title: x for x in load_labelled(cal)}
    assert items["Data Engineer"].label == "yes" and items["Junior Sales Assistant"].label == "no"
    assert items["Data Engineer"].job.description == "description 1"
    assert "own label" in items["Data Engineer"].note
    # a second import changes nothing, and an existing label is never overwritten
    db.set_mark(("fake", "1"), "bad", "changed my mind")
    second = import_marks(cfg, db)
    assert all(o.startswith("already in the set") for *_, o in second)
    assert any("kept" in o for *_, o in second)
    assert len(yaml.safe_load((cal / "labels.yaml").read_text(encoding="utf-8"))["jobs"]) == 2
