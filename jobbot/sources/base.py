from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from ..http import PoliteClient
from ..models import Job

if TYPE_CHECKING:
    from ..config import Config
    from ..db import DB


class Source(ABC):
    """One feed: a plugin kind (e.g. greenhouse) applied to one company/board.

    Subclasses keep the parsing in pure functions so tests can run them against
    recorded fixtures without any network access.
    """

    kind: ClassVar[str]
    config_key: ClassVar[str] = "companies"   # which list in `sources.<kind>` holds this kind's entries

    def __init__(self, name: str, company: str):
        self.name = name
        self.company = company

    @property
    def feed(self) -> str:
        return f"{self.kind}:{self.name}"

    @classmethod
    @abstractmethod
    def from_entry(cls, entry: dict[str, Any]) -> Source:
        """Build from one item of `sources.<kind>.companies` in config.yaml."""

    @classmethod
    def from_config(cls, cfg: Config, entry: dict[str, Any]) -> Source | None:
        """Like from_entry, for sources that need more of the config (secrets). None = skip this entry."""
        return cls.from_entry(entry)

    def bind(self, db: DB) -> None:
        """Called once by the Pipeline; sources that keep state in SQLite override this."""

    def after_ingest(self, now: Any) -> None:
        """Called by the Pipeline after the jobs from a successful fetch() were stored (never by dry-run)."""

    @abstractmethod
    async def fetch(self, http: PoliteClient) -> list[Job]:
        """Return the current listing. Descriptions may be left as None."""

    async def fetch_details(self, http: PoliteClient, job: Job) -> Job:
        """Fill in job.description. Only called for jobs that pass the title/location filter."""
        return job

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.feed}>"
