"""Load config.yaml (+ .env) into validated settings objects."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class Role(BaseModel):
    name: str
    keywords: list[str]


class OnsiteAbroad(BaseModel):
    # Drop jobs asking for at least this many on-site days a week in one of `places`. null = off.
    min_days: int | None = 1
    places: list[str] = [
        "london", "uk", "united kingdom", "england", "dublin", "ireland", "paris", "france", "berlin", "munich",
        "munchen", "hamburg", "frankfurt", "germany", "vienna", "wien", "austria", "zurich", "switzerland",
        "amsterdam", "netherlands", "brussels", "belgium", "madrid", "barcelona", "spain", "lisbon", "portugal",
        "milan", "italy", "warsaw", "krakow", "poland", "prague", "czech", "budapest", "hungary", "stockholm",
        "sweden", "copenhagen", "denmark", "oslo", "helsinki", "sofia", "bulgaria", "belgrade", "athens", "istanbul",
        "tel aviv", "israel", "new york", "san francisco", "mountain view", "seattle", "united states", "usa",
        "canada", "toronto", "bangalore", "india", "singapore",
    ]


class HardRequirements(BaseModel):
    """Requirements you can't meet; matched against the folded (lowercase, no diacritics) description."""
    # Languages you don't speak, each with the words that name it (English and Romanian forms).
    # A job is dropped when one appears next to a word from `language_required_words` in the same clause,
    # unless that clause marks it optional ("is a plus", "constituie un avantaj", ...).
    languages: dict[str, list[str]] = {
        "German": ["german", "deutsch", "germana", "germane"],
        "French": ["french", "francais", "franceza", "franceze"],
        "Dutch": ["dutch", "flemish", "nederlands", "olandeza", "olandeze"],
        "Hungarian": ["hungarian", "magyar", "maghiara", "maghiare"],
        "Italian": ["italian", "italiana", "italiene"],
        "Spanish": ["spanish", "espanol", "spaniola", "spaniole"],
        "Portuguese": ["portuguese", "portugheza", "portugheze"],
        "Polish": ["polish", "polona", "poloneza", "poloneze"],
        "Czech": ["czech", "ceha", "cehe"],
        "Swedish": ["swedish", "suedeza", "suedeze"],
        "Danish": ["danish", "daneza", "daneze"],
        "Norwegian": ["norwegian", "norvegiana", "norvegiene"],
        "Finnish": ["finnish", "finlandeza", "finlandeze"],
        "Greek": ["greek", "greaca"],
        "Turkish": ["turkish", "turca"],
        "Russian": ["russian", "rusa"],
        "Japanese": ["japanese", "japoneza", "japoneze"],
        "Chinese": ["chinese", "mandarin", "chineza", "chineze"],
        "Korean": ["korean", "coreeana"],
        "Arabic": ["arabic", "araba"],
        "Hebrew": ["hebrew", "ebraica"],
        "Bulgarian": ["bulgarian", "bulgara"],
        "Ukrainian": ["ukrainian", "ucraineana"],
        "Serbian": ["serbian", "sarba"],
        "Croatian": ["croatian", "croata"],
    }
    language_required_words: list[str] = [
        "fluent", "fluency", "fluently", "proficient", "proficiency", "advanced", "native", "mother tongue",
        "business level", "working knowledge", "c1", "c2", "b2", "written and spoken",
        "spoken and written", "verbal and written", "written and verbal", "oral and written", "speaker", "speaking",
        "language skills", "knowledge of", "command of", "required", "mandatory", "must",
        "fluenta", "avansat", "nivel", "cunoasterea", "cunostinte", "vorbit", "vorbita", "scris", "scrisa",
        "obligatoriu", "obligatorie", "necesar", "necesara", "limba", "limbii",
    ]
    # A Master's/PhD requirement drops the job unless a Bachelor's is also accepted somewhere in the text.
    graduate_only_regex: list[str] = [
        r"\b(?:master(?:'|’)?s?|msc|m\.\s?sc\.?|mres|phd|ph\.\s?d\.?|doctoral|doctorate)\s*"
        r"(?:or\s+(?:phd|ph\.\s?d\.?|doctoral|doctorate)\s*)?"
        r"(?:degree|student|students|program|programme|candidate|candidates|thesis|level)\b",
        r"\b(?:pursuing|enrolled in|studying for|completing)\s+(?:a|an|your)?\s*(?:master|msc|phd|ph\.\s?d|doctor)",
        r"\b(?:masterand\w*|masteranz\w*|doctorand\w*|studii de master|program\w* de master|masterat\w*)\b",
    ]
    undergraduate_regex: list[str] = [
        r"\b(?:bachelor\w*|bsc|b\.\s?sc|undergrad\w*|licent\w*|studii superioare)\b",
        r"\bbs\s*(?:degree|in\b|or\b|/)|\bb\.s\.",
    ]
    onsite_abroad: OnsiteAbroad = OnsiteAbroad()


class EmployerTitleExclusion(BaseModel):
    employer: str               # whole words in the company name, e.g. "Contoso" matches "Contoso Industries GmbH"
    title_contains: list[str]   # whole words/phrases in the title


class Profile(BaseModel):
    cv_path: Path = Path("cv.pdf")
    roles: list[Role]
    # Titles with one of these also pass the role check (as "Entry level"), whatever the field. Unlike role
    # names, they are not shown to the LLM, which decides whether the field fits.
    entry_level_keywords: list[str] = []
    exclude_title_words: list[str] = []
    exclude_title_at_employer: list[EmployerTitleExclusion] = []
    exclude_if_years_at_least: int | None = 3
    exclude_text_regex: list[str] = []
    hard_requirements: HardRequirements = HardRequirements()


class LocationRule(BaseModel):
    name: str
    match: list[str]
    priority: Literal["high", "low"] = "high"
    remote_only: bool = False
    enabled: bool = True
    min_score: int | None = None


class Locations(BaseModel):
    remote_keywords: list[str] = ["remote"]
    rules: list[LocationRule]


class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"
    num_ctx: int = 8192           # context window to allocate; CV + job must fit in it
    num_predict: int = 1024       # max output tokens
    think: bool | None = False    # for thinking models; null = don't send the parameter
    keep_alive: str = "2m"        # how long the model stays loaded after the last score
    timeout_seconds: float = 300
    max_concurrency: int = 1
    # When Ollama doesn't answer, start it once (the tray app, like the autostart wrapper) and wait this long.
    auto_start: bool = True
    auto_start_wait_seconds: float = 90


class AnthropicConfig(BaseModel):
    model: str = "claude-haiku-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_concurrency: int = 4


class LLMConfig(BaseModel):
    backend: Literal["ollama", "anthropic", "off"] = "ollama"
    max_description_tokens: int = 3000
    reserve_output_tokens: int = 1024
    # A score within this distance of the notify threshold is scored a second time with the model's prompt
    # cache cold, and the mean of the two is used. null = off.
    rescore_margin: int | None = 15
    ollama: OllamaConfig = OllamaConfig()
    anthropic: AnthropicConfig = AnthropicConfig()


class EmailConfig(BaseModel):
    # off | fallback (only when the Discord send fails) | both (every alert and notice goes to both)
    mode: Literal["off", "fallback", "both"] = "fallback"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465          # implicit TLS; credentials come from .env
    timeout_seconds: float = 30


class HeartbeatConfig(BaseModel):
    enabled: bool = True
    time: str = "09:00"           # local time (HH:MM); sent at the first run after it, once a day

    @field_validator("time")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        datetime.strptime(v, "%H:%M")
        return v


class DigestConfig(BaseModel):
    """Jobs that came from job-alert emails never alert on their own; they go out once a day in one digest."""
    enabled: bool = True
    time: str = "18:00"           # local time (HH:MM); sent at the first run at or after it, once a day
    title_hint: bool = False      # off by default (noisy): sort by a model hint (good/maybe/weak, never a score) when the model is free
    # Email jobs only (ATS jobs are unchanged): enter the digest only if the title has a word from
    # profile.entry_level_keywords (junior, intern, graduate, ...). The job-alert emails carry no level line.
    require_entry_level_title: bool = True
    max_items: int = 60           # longer digests are cut here; the rest stays queued for the next one

    @field_validator("time")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        datetime.strptime(v, "%H:%M")
        return v


class NotifyConfig(BaseModel):
    min_score: int = 65
    discord: bool = True
    email: EmailConfig = EmailConfig()
    heartbeat: HeartbeatConfig = HeartbeatConfig()
    digest: DigestConfig = DigestConfig()


class ControlConfig(BaseModel):
    """Pause / defer controls (`jobbot pause|defer|resume|unload|status`)."""
    pause_auto_resume_hours: float = 12      # `jobbot pause` resumes by itself after this long; 0 = never
    defer_max_hold_minutes: float | None = 120   # jobs waiting for scoring go out [unscored] after this; null = wait
    auto_defer: bool = True                  # defer scoring while the GPU has too little free VRAM
    min_free_vram_gb: float = 8              # ... when the model isn't loaded yet
    gpu_busy_util: float = 20                # GPU utilization (%) above this in a sample = someone is using the GPU
    gpu_quiet_minutes: float = 10            # ... and scoring waits until no sample was above it for this long
    gpu_sample_seconds: float = 30           # the bot samples nvidia-smi this often and keeps 15 minutes
    busy_process_patterns: list[str] = []   # name or command line, case-insensitive
    busy_lock_path: str = "data/gpu-busy.lock"   # other projects create/delete it while they use the GPU
    busy_max_hours: float = 12               # a busy flag or lock older than this is ignored (stale)
    nvidia_smi: str = "nvidia-smi"
    poll_seconds: float = 5                  # how often the running bot looks at the control state


class ScheduleConfig(BaseModel):
    interval_minutes: float = 10


class HTTPConfig(BaseModel):
    user_agent: str = "jobbot/0.1 (personal job-alert bot)"
    timeout_seconds: float = 20
    per_host_min_interval_seconds: float = 2.0
    max_retries: int = 3
    max_concurrency: int = 8


class HealthConfig(BaseModel):
    failure_threshold: int = 3


class DedupConfig(BaseModel):
    crosspost_window_days: int = 30


class StorageConfig(BaseModel):
    db_path: Path = Path("data/jobbot.sqlite3")


class SourceConfig(BaseModel):
    enabled: bool = True
    companies: list[dict[str, Any]] = []
    mailboxes: list[dict[str, Any]] = []   # only used by the `email` kind


class Secrets(BaseModel):
    discord_webhook_url: str | None = None
    anthropic_api_key: str | None = None
    smtp_user: str | None = None
    smtp_app_password: str | None = None
    alert_email_to: str | None = None   # defaults to smtp_user
    jobmail_address: str | None = None        # the dedicated inbox the email source reads (IMAP, read-only)
    jobmail_app_password: str | None = None


class Config(BaseModel):
    profile: Profile
    locations: Locations
    llm: LLMConfig = LLMConfig()
    notify: NotifyConfig = NotifyConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    control: ControlConfig = ControlConfig()
    http: HTTPConfig = HTTPConfig()
    health: HealthConfig = HealthConfig()
    dedup: DedupConfig = DedupConfig()
    storage: StorageConfig = StorageConfig()
    sources: dict[str, SourceConfig] = {}
    secrets: Secrets = Field(default_factory=Secrets, exclude=True)
    base_dir: Path = Path(".")

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def db_path(self) -> Path:
        return self.resolve(self.storage.db_path)

    @property
    def cv_path(self) -> Path:
        return self.resolve(self.profile.cv_path)

    @property
    def data_dir(self) -> Path:
        return self.db_path.parent


# Values from .env.example; a copied-but-unedited .env must not count as configured.
PLACEHOLDERS = {"https://discord.com/api/webhooks/<id>/<token>", "sk-ant-...", "you@gmail.com",
                "abcd efgh ijkl mnop", "your.jobmail@gmail.com"}


def _env(name: str) -> str | None:
    value = (os.environ.get(name) or "").strip()
    return value if value and value not in PLACEHOLDERS else None


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    # .env next to the config wins over nothing, but never over real env vars.
    load_dotenv(path.parent / ".env", override=False)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config.model_validate({**raw, "base_dir": path.parent})
    cfg.secrets = Secrets(
        discord_webhook_url=_env("DISCORD_WEBHOOK_URL"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        smtp_user=_env("SMTP_USER"),
        # Gmail shows app passwords in groups of four; spaces are not part of the password.
        smtp_app_password=(_env("SMTP_APP_PASSWORD") or "").replace(" ", "") or None,
        alert_email_to=_env("ALERT_EMAIL_TO") or _env("SMTP_USER"),
        jobmail_address=_env("JOBMAIL_ADDRESS"),
        jobmail_app_password=(_env("JOBMAIL_APP_PASSWORD") or "").replace(" ", "") or None,
    )
    return cfg
