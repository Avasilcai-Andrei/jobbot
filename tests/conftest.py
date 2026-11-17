from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from jobbot.config import Config, Secrets
from jobbot.http import PoliteClient
from jobbot.models import Job

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(rel: str) -> Any:
    return json.loads((FIXTURES / rel).read_text(encoding="utf-8"))


def make_config(tmp_path: Path, **overrides: Any) -> Config:
    """A config independent of the local config.yaml, so editing that never breaks tests."""
    raw: dict[str, Any] = {
        "profile": {
            "roles": [
                {"name": "Data Engineer", "keywords": ["data engineer", "etl"]},
                {"name": "Data Scientist", "keywords": ["data scientist", "machine learning"]},
                {"name": "Python / Backend", "keywords": ["python", "backend", "software engineer"]},
            ],
            "exclude_title_words": ["senior", "sr", "lead", "principal", "manager", "staff"],
            "exclude_if_years_at_least": 3,
            "exclude_text_regex": [r"\bunpaid\b.{0,40}\bintern", r"\bneplatit"],
        },
        "locations": {
            "remote_keywords": ["remote", "distributed", "la distanta"],
            "rules": [
                {"name": "Cluj-Napoca", "match": ["cluj"], "priority": "high"},
                {"name": "Romania (remote)", "match": ["romania"], "remote_only": True, "priority": "high"},
                {"name": "Bucharest", "match": ["bucharest", "bucuresti"], "priority": "low"},
            ],
        },
        "storage": {"db_path": str(tmp_path / "test.sqlite3")},
        "schedule": {"interval_minutes": 10},
        "base_dir": tmp_path,
    }
    raw.update(overrides)
    cfg = Config.model_validate(raw)
    cfg.secrets = Secrets()
    return cfg


def make_job(job_id: str = "1", **kw: Any) -> Job:
    defaults: dict[str, Any] = dict(
        source="fake", job_id=job_id, feed="fake:acme", title="Junior Data Engineer", company="Acme",
        location="Cluj-Napoca, Romania", url=f"https://example.com/jobs/{job_id}",
        description="We use Python and SQL. 1+ years of experience is a plus.",
        posted_at=datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc),
    )
    defaults.update(kw)
    return Job(**defaults)


async def _no_sleep(_: float) -> None:
    return None


def mock_client(handler: Any, **kw: Any) -> PoliteClient:
    kw.setdefault("per_host_interval", 0)
    return PoliteClient(user_agent="jobbot-test/0", transport=httpx.MockTransport(handler), sleep=_no_sleep, **kw)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return make_config(tmp_path)
