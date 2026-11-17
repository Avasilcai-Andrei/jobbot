# jobbot

A personal job-alert bot. It polls company career feeds every 10 minutes, filters new postings against your profile
with rules, scores the survivors against your CV with a local LLM, and sends a Discord (or email) alert, usually
within 30 minutes of a posting going up. Jobs that arrive through job-alert emails (LinkedIn, Indeed) are collected
into one daily digest instead.

```
fetch all feeds (async, polite) ─▶ dedup (source+job_id) ─▶ stage 1 rules ─▶ cross-post dedup ─▶ stage 2 scorer ─▶ alert
                                                                                                 (Ollama | Anthropic | off)
job-alert emails (IMAP, read-only) ─▶ parsers ─▶ rules ─▶ ATS-copy dedup ─▶ daily digest
```

Features:
* Sources: Greenhouse, Lever, SmartRecruiters, Workday, Workable, Teamtailor, Recruitee and Personio public job APIs,
  plus job-alert emails read from a dedicated mailbox.
* Rules for titles, locations, years of experience, unpaid internships and hard requirements (languages,
  Master's/PhD-only, on-site days abroad), in English and Romanian.
* A pluggable scorer (local Ollama by default, the Anthropic API as an option, or off) with JSON-schema output,
  caching and a second opinion near the threshold.
* A calibration tool that measures how well a model separates good jobs from bad ones on labelled postings.
* Discord alerts with an email fallback, a daily heartbeat, feed-health warnings, and good/bad marks on alerts.
* Windows autostart via Task Scheduler, GPU-busy detection, and a terminal control panel.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env               # fill in DISCORD_WEBHOOK_URL (ANTHROPIC_API_KEY only if llm.backend: anthropic)
cp config.example.yaml config.yaml # then edit it
# put your CV at ./cv.pdf (a .txt CV also works)
```

Edit `config.yaml` to set your roles, locations, exclusions, threshold and companies. Also put your contact address
in `http.user_agent`, so the sites you poll know who is calling. `config.yaml` holds your personal profile, so keep
it out of version control (add it to `.git/info/exclude` or `.gitignore`); `config.example.yaml` is the template.
Another path can be given with `--config` or `JOBBOT_CONFIG`.

Secrets live only in `.env` (never in `config.yaml`); values copied unchanged from `.env.example` are ignored.
`.env`, `cv.pdf`, `data/` and `calibration/results.json` are gitignored.

The example targets Romania: the Workday and SmartRecruiters sources filter on Romania, and the rule patterns are
in English and Romanian. Locations and patterns are plain config, so other areas mostly need edits there.

### Local scoring with Ollama (default)

1. Install Ollama from https://ollama.com/download and make sure it is running (the tray app, or `ollama serve`).
2. Pull a model. Both of these fit fully in 12 GB of VRAM with the default 8k context:
   ```bash
   ollama pull qwen3.5:9b       # ~6.6 GB, default in config.yaml
   ollama pull ministral-3:8b   # ~6.0 GB
   ```
3. The model is set in `config.yaml` under `llm.ollama.model`. `python -m jobbot calibrate` compares models on
   your own labelled jobs (see [Calibration](#calibration)).

### Scoring backends

`llm.backend` is one of:
* `ollama` (default): a local model, see above.
* `anthropic`: the Anthropic API (`llm.anthropic.model`, default `claude-haiku-5-5`). Needs `ANTHROPIC_API_KEY`
  in `.env`. Uses structured outputs and prompt caching.
* `off`: no scoring; every rule match is sent as `[unscored]`.

### Discord

Create a webhook (Server Settings → Integrations → Webhooks → New Webhook → Copy URL), put it in `.env` as
`DISCORD_WEBHOOK_URL`, then run `python -m jobbot test-notify`.

### Email alerts (Gmail)

1. Turn on 2-Step Verification for the Google account, then create an app password at
   https://myaccount.google.com/apppasswords.
2. Put it in `.env`: `SMTP_USER=you@gmail.com`, `SMTP_APP_PASSWORD=<the 16 letters; spaces are fine>`, and
   optionally `ALERT_EMAIL_TO` (defaults to `SMTP_USER`).
3. `notify.email.mode` in `config.yaml`: `fallback` (the default: email only when the Discord send fails), `both`, or
   `off`. With only one channel configured, that one is used.

`python -m jobbot run` refuses to start without at least one channel, so no alert is silently lost.

### Job-alert emails (optional)

Job boards that must not be scraped can still email you their alerts. jobbot reads those emails from a dedicated
inbox:

1. Create a separate Gmail account for job alerts, enable IMAP, and create an app password for it.
2. Put `JOBMAIL_ADDRESS` and `JOBMAIL_APP_PASSWORD` in `.env`, and set `sources.email.enabled: true` in
   `config.yaml`. Without the credentials the email source is skipped with a warning.
3. Subscribe that address to LinkedIn and Indeed job alerts, or forward them to it with a Gmail filter
   ("Forward to" keeps the original sender, which is what the parsers match on).
4. Settings are under `sources.email.mailboxes` in `config.yaml` (`lookback_days`, `max_messages_per_run`, and
   `senders` to map extra From domains to a parser).

The mailbox is opened read-only (`EXAMINE` + `BODY.PEEK`): nothing is marked read, moved or deleted. jobbot never
fetches the linked job pages and never resolves the tracking links. On the first run, existing mail is recorded
as seen without producing jobs. Mail from a known sender whose parser isn't written yet is left alone and picked up
once the parser exists; unknown senders are listed by `report`.

### Daily heartbeat and digest

* **Heartbeat:** once a day, at the first run after `notify.heartbeat.time` (local time, default `09:00`), jobbot
  sends a short summary: runs in the last 24 h (and crashes), postings fetched, new postings, rule passes, scored,
  alerted, failing feeds, and time spent paused or GPU-busy. **If no heartbeat arrives, the bot isn't running.**
  Silence otherwise looks the same as "no matching jobs". To send one now: `python -m jobbot heartbeat`.
* **Digest:** email jobs never alert on their own and are never scored, because an alert email carries no
  description and a model would have to guess the requirements. Jobs that pass the rules are queued and sent once a
  day at `notify.digest.time` (default `18:00`): title, company, location, source and link, at most
  `max_items` (60) per day. With `require_entry_level_title: true` (the default), an email job must have an
  entry-level word in its title (junior, intern, graduate, trainee, …), since the title is the only signal. If an
  ATS feed lists the same role, the ATS copy is scored and may alert, and the email copy is dropped. Preview it with
  `python -m jobbot digest` (read-only, sends nothing).

## Running it on Windows (autostart at logon)

Scoring uses the local GPU, so jobbot runs on your PC while you're logged on. A Task Scheduler task starts
`scripts\jobbot-autostart.ps1` hidden at logon. The script:

* starts the Ollama tray app if Ollama isn't answering (or `ollama serve` if there is no tray app), and waits for it.
  It first waits 60 s after logon and until the NVIDIA driver answers (`nvidia-smi`), because an Ollama started
  too early finds no GPU and silently runs on the CPU. If the running Ollama (also one started by its own Startup
  shortcut) reports only a CPU in its `server.log`, the script restarts it, at most twice;
* runs `python -m jobbot run`, logging to `data\logs\jobbot.log` (rotated at 5 MB, 5 old files kept);
* restarts jobbot after a crash, waiting 10 s at first and up to 10 min after repeated crashes;
* waits and checks again every 5 min if jobbot exits with a configuration problem (e.g. no `.env` yet), so fixing
  `.env` is enough to get it going.

Only one `jobbot run` can run at a time (a lock in `data\`). The wrapper's own log is `data\logs\autostart.log`.

Install (no administrator rights needed), from the project folder in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install-autostart.ps1 -StartNow
```

Check, stop, start and remove it:

```powershell
Get-ScheduledTask -TaskName jobbot | Select-Object TaskName, State     # Running / Ready
Get-Content data\logs\jobbot.log -Tail 20 -Wait                        # follow the log (Ctrl+C to stop following)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\uninstall-autostart.ps1 -KeepTask   # stop only
Start-ScheduledTask -TaskName jobbot                                   # start again
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\uninstall-autostart.ps1             # stop and remove
```

A running `jobbot run` keeps the code it started with, so restart it after updating. While the PC sleeps the bot
pauses too; the heartbeat shows the gap as fewer runs, and `report` counts jobs posted during the gap as "bot
offline" rather than as slow detections.

## Pausing the bot, and the control panel

The bot keeps running hidden under Task Scheduler. To free the GPU (a game, another project) without stopping it:

| Command | Effect |
|---|---|
| `python -m jobbot pause [--hours N]` | Nothing runs (no fetching, scoring or sending) and the model is unloaded. Auto-resumes after `control.pause_auto_resume_hours` (12) unless you pass `--hours`; `--hours 0` = until `resume` |
| `python -m jobbot resume` | Ends a pause and any deferral; the bot runs at once to catch up |
| `python -m jobbot defer [--minutes N \| --next-scan \| --until-resume]` | Keeps fetching and filtering but doesn't load the model; matching jobs wait as `pending` and are scored when the deferral ends. After `control.defer_max_hold_minutes` (120) a waiting job is sent `[unscored]` so nothing urgent is lost |
| `python -m jobbot busy [--set "reason" [--hours N] \| --clear]` | Shows, sets or clears the busy flag (and shows the lock file) |
| `python -m jobbot unload` | Unloads jobbot's own Ollama model now (never another one) |
| `python -m jobbot status [--json]` | State, bot, model, free VRAM, pending jobs |
| `python -m jobbot panel` (or double-click `panel.bat`) | Terminal control panel (Textual), refreshes every second: state with timers, whether the bot is alive and its next scan, model loaded / VRAM, free VRAM, "GPU busy: <reason>" or "GPU quiet for N min", why scoring is deferred, pending jobs and the oldest one's wait against the hold limit, last run, failing feeds, the last 5 alerts with their mark ids. Keys: **P** pause/resume, **D** defer to next scan, **M** defer N minutes, **R** resume, **U** unload, **Q** quit. Closing it doesn't affect the bot |

The CLI, the panel and the bot share the same state in SQLite, so any of them can change it while the others run.

**GPU busy detection (`control.auto_defer`, on by default).** The bot samples `nvidia-smi` every 30 s and keeps 15
minutes of history. Scoring waits (jobs stay `pending`; the 120-minute hold then `[unscored]` still applies; fetching
and filtering go on) while any of these is true, and starts as soon as none is:

| Trigger | Meaning |
|---|---|
| another model loaded | `ollama ps` lists a model other than jobbot's own (jobbot never unloads or touches it) |
| GPU in use | utilization above `gpu_busy_util` (20 %) in any sample of the last `gpu_quiet_minutes` (10), not counting jobbot's own scoring, so scoring starts only after the GPU has been quiet that long |
| low free VRAM | below `min_free_vram_gb` (8) while jobbot's model isn't loaded |
| busy process | a running process whose name or command line contains one of `busy_process_patterns` (case-insensitive; empty by default) |
| busy flag / lock file | `python -m jobbot busy --set "reason" [--hours N]` (ends with `busy --clear`), or a lock file at `busy_lock_path` (`data/gpu-busy.lock`) that another project creates and deletes. Plain text (first line = reason, `pid=1234` optional) or JSON `{"reason": "...", "pid": 1234, "expires_at": "<ISO time>"}`. Ignored when its PID is dead or it is older than `busy_max_hours` (12); a flag set without `--hours` also stops counting after that |

The reason is logged once per change, shown in `status` and the panel, and the heartbeat lists the time spent busy per
reason. After a restart the bot has no history, so it treats the GPU as quiet until a busy sample shows up.
If nvidia-smi or Ollama can't be read, that trigger is skipped. `control.auto_defer: false` turns off the automatic
triggers; the explicit flag and lock file still count.

To mark a script as using the GPU, put this around it (the lock is ignored if the script dies and its PID disappears):

```python
import json, os, pathlib
LOCK = pathlib.Path(r"<path to jobbot>\data\gpu-busy.lock")
LOCK.write_text(json.dumps({"reason": "benchmark", "pid": os.getpid()}))  # 1: start
try:
    ...  # the benchmark / build
finally:
    LOCK.unlink(missing_ok=True)  # 2: end, also when the script fails
```

**Exit codes** (for scripts that call `pause` before using the GPU and `resume` after): `status` returns 0 running,
10 deferred, 11 paused. `pause` and `unload` return 0 on success, 4 if Ollama isn't reachable (`pause` still returns 0:
nothing holds the GPU) and 5 if the model is still loaded afterwards. 2 is a configuration problem, 3 means another
`run` already holds the lock, 130 means stopped (Ctrl+C). No feed warnings are sent while paused.

## Commands

| Command | What it does |
|---|---|
| `python -m jobbot test-notify` | Sends a sample alert, to check the webhook |
| `python -m jobbot heartbeat [--console]` | Sends the daily heartbeat message now |
| `python -m jobbot digest [--llm]` | Previews the email digest from the live mailbox (read-only, sends nothing) |
| `python -m jobbot dry-run [--llm] [--all]` | Evaluates everything listed right now and shows why each job passes or drops. Sends and stores nothing |
| `python -m jobbot once` / `run` | Runs the pipeline once, or loops on `schedule.interval_minutes`. Add `--console` to print alerts instead of sending them |
| `python -m jobbot report [--days 7] [--by feed]` | Detection-latency percentiles per source, score stability, feed health, email senders |
| `python -m jobbot mark <id> good\|bad [note]` | Marks an alert good or bad; the 6-character id is in every Discord and email alert |
| `python -m jobbot report --marks [--days 7]` | Alerts with your marks, and the ones you have not marked yet (with their ids) |
| `python -m jobbot recent [--status filtered]` | Recent jobs and what happened to them (and why) |
| `python -m jobbot calibrate [--model M ...] [--cv file]` | Scores the labelled calibration jobs and suggests a threshold |
| `python -m jobbot calibrate-add --feed F --job-id ID --label yes\|maybe\|no` | Saves a currently listed job into the calibration set |
| `python -m jobbot calibrate-import-marks` | Copies your good/bad alert marks into the calibration set (good = yes, bad = no) |
| `pause`, `resume`, `defer`, `busy`, `unload`, `status`, `panel` | See [Pausing the bot](#pausing-the-bot-and-the-control-panel) |

Global flags: `--config` (or `JOBBOT_CONFIG`), `--log-level`, `--log-file` (or `JOBBOT_LOG_FILE`). Logs are in UTC.

## How it behaves

* **First run:** On a feed's first successful fetch, every existing posting is marked as seen without alerting. The same
  happens for any company you add later.
* **Stage 1 (rules):** A title must contain a role keyword (or an entry-level word such as intern, working student,
  trainee, stagiar or junior, in any field) and no seniority word, and the location must match a location rule.
  Descriptions are downloaded only for jobs that pass that check. The description is then checked for "N+ years of
  experience" (≥ 3 drops the job, in English or Romanian; "3+ years in software engineering", "3 years as a data
  engineer" and "minim 3 ani în domeniu" count too, while company history such as "for 30+ years" doesn't) and for
  unpaid-internship patterns. `profile.hard_requirements` then drops requirements you can't meet: fluency in a
  language other than English or Romanian, Master's/PhD-only roles, and N+ on-site days a week abroad (e.g. "3 days
  per week in our London office"). The patterns are in English and Romanian, and you can edit them. The reason is
  logged and shown by `recent --status filtered` and `dry-run`.
* **Cross-posts:** If the same company posts the same normalised title again within 30 days, the later posting is
  dropped. When copies arrive together, the posting in the higher-priority location wins.
* **Stage 2 (scorer):** The model receives the CV (with contact details redacted) and the job, and returns a 0–100
  score, a 2–3 line reason and a list of missing requirements, as JSON that is validated against a schema.
  * Ollama runs with `temperature: 0`, a JSON-schema `format` and `think: false`. After the first score of each run,
    jobbot checks `/api/ps` (what `ollama ps` shows) and logs a warning unless the model is 100% on the GPU.
  * The job text starts with today's date, so the model can tell an expected graduation date from a finished degree.
  * Unpaid internships are excluded by the rules only. With "unpaid internships" in the prompt's exclusions, the
    model held it against every internship. Seniority and years of experience stay in the prompt as well as in the
    rules, because the model needs them for titles the rules can't judge ("Intermediate", "Mid-Senior level").
  * Malformed output gets one retry, with the bad answer and the validation error fed back to the model.
  * Results are cached in SQLite. The cache key includes the prompt version, the backend and the model name, so
    switching models or changing the prompt re-scores jobs.
  * A score ≥ `notify.min_score` (65) triggers an alert. Each location rule can override this with its own
    `min_score`.
  * Near the threshold (within `llm.rescore_margin`, 15), the job is scored a second time with the model's prompt
    cache cold, and the mean decides. Both scores are logged and shown in the alert. `report` shows how often this
    happens.
  * If your CV names the job's employer, the employer's name is hidden from the model (as `[Company]`), so having
    worked there doesn't inflate the score. To drop some titles at one employer entirely, use
    `profile.exclude_title_at_employer`.
* **Long descriptions** are cut, at a paragraph or sentence boundary, to `llm.max_description_tokens`. The budget also
  shrinks so that the instructions, CV, job and answer fit in `llm.ollama.num_ctx`. Tokens are estimated at 3.5
  characters each. Calibration measured this as about 1.3× the real count, so the estimate errs on the safe side.
* **[unscored] alerts:** A job that passes the rules is never dropped because scoring isn't possible. This covers
  `backend: off`, the model not pulled (one warning per run), a missing CV, and output that is still malformed after
  the retry. In all these cases the alert is sent with an `[unscored]` tag and the reason. If Ollama isn't reachable
  at all, jobbot tries to start it once, holds the jobs as `pending`, and sends them `[unscored]` after the hold limit.
* **Failures:** HTTP requests are retried with backoff and respect `Retry-After`. A non-JSON answer is logged with its
  status, content type, redirects and the start of the body. A failed alert is retried next run (up to 5 times).
  After 3 consecutive failed runs, a feed triggers one warning and is polled less often (up to once an hour); it
  sends a "working again" notice when it recovers. Warnings for several feeds of the same kind are grouped into one
  message.
* **Latency:** `report` shows p50, p90, p95 and the share of jobs detected within 30 minutes, measured from
  `posted_at` to `first_seen_at`. Workday gives only a posting *date*, so Workday jobs appear under "no posted_at".
* **Marks:** every alert carries a short id. `mark <id> good|bad` records your verdict, `report --marks` lists them,
  and `calibrate-import-marks` turns them into calibration labels.

## Calibration

A calibration set is a list of saved real postings (`calibration/jobs/<id>.json`), each labelled `yes`, `maybe` or
`no` in `calibration/labels.yaml`. The set depends on what you are looking for, so it isn't part of the repo; build
your own (the format is in `calibration/labels.example.yaml`):

1. Run `python -m jobbot dry-run --all` or `recent` to find postings worth labelling, including clear no's.
2. Save each one with `python -m jobbot calibrate-add --feed <kind:slug> --job-id <id> --label yes|maybe|no`
   (add `--location "<your city>"` to judge a role posted elsewhere as if it were local; the original location is
   kept in the JSON). From PowerShell for Workday ids, since Git Bash rewrites arguments that start with `/`.
3. Once alerts arrive, `mark` them good or bad and run `python -m jobbot calibrate-import-marks`.
4. Read each posting's requirements before labelling; aim for a few dozen jobs with both yes and no labels.

`python -m jobbot calibrate --model qwen3.5:9b --model ministral-3:8b` scores each job with each model and prints:
* each score next to your label, and whether the rules would drop the job
* the score distribution per label
* the AUC, i.e. how well a model separates yes from no
* what "rules + threshold" would actually alert on
* failure counts and time per job
* a suggested threshold: the best trade-off between alerting on yes jobs and not alerting on no jobs, leaning towards
  recall

Per-job reasons go to `calibration/results.json` (gitignored: it quotes the model's reasoning about your CV).
Calibration uses a fresh in-memory cache, so timings and answers are real. Re-run it after changing the prompt, the
model, the CV, the labels or the rules. `calibration/sample_cv.txt` is a stand-in profile for testing without a
real CV.

Model comparison on a 22-job set of entry-level data/Python postings (5 yes, 6 maybe, 11 no; RTX 5070):

| | qwen3.5:9b | ministral-3:8b |
|---|---|---|
| AUC (yes vs no) | **0.96** | 0.77 |
| yes / maybe / no mean score | 88 / 52 / 35 | 58 / 33 / 38 |
| lowest yes, highest no | 82, 85 (German required) | 30, 80 |
| at threshold 65 | 5/5 yes, 1/11 no alerted | 2/5 yes, 3/11 no alerted |
| malformed outputs | 0/22 | 0/22 |
| median time per job | 1.9 s | 2.9 s |

On a 31-job set with the current prompt, qwen3.5:9b reaches AUC 0.94, and rules + threshold 65 alert on 6/6 yes,
0/8 maybe and 0/17 no.

What changed the results along the way:
* **Labels mattered most.** Labels based on job titles alone were wrong: reading the postings showed that several
  "obvious" matches require a Master's, French, German, a London office or 6+ years.
* **Adding today's date to the job text** fixed the model misreading "2024 – 2027 (expected)" as an already-finished
  degree. AUC went from 0.79 to 0.96.
* **Hiding the employer beat telling the model to ignore it.** Prompt instructions such as "the employer must not
  raise the score" either did nothing or made the model reason in odd ways; replacing the name with `[Company]`
  fixed the bias without changing any other score.
* **A rule capping the score at 30 for unmet requirements made things worse.** The model anchored nearly every job
  at 30, so the rule was removed.

Small models sometimes list a hard requirement as "missing" without lowering the score, which is why hard
requirements are enforced by rules. Scores are identical across repeat runs in the same order, but Ollama reuses its
prompt cache between calls, so a borderline job's score can shift when jobs are processed in a different order (one
moved from 45 to 68). That is what the near-threshold second opinion is for.

## Sources

Only public job APIs and feeds are used. Before adding a source, check its robots.txt and terms; skip it if they
disallow automated access.

| Kind | API | In `config.example.yaml` |
|---|---|---|
| `greenhouse` | `boards-api.greenhouse.io/v1/boards/<slug>/jobs` | Thoughtworks, Elastic |
| `smartrecruiters` | `api.smartrecruiters.com/v1/companies/<id>/postings?country=ro` | Endava |
| `workday` | `<tenant>.wd<N>.myworkdayjobs.com/wday/cxs/<tenant>/<site>/jobs` | Adobe |
| `lever` | `api.lever.co/v0/postings/<slug>?mode=json` | – |
| `workable` | `apply.workable.com/api/v1/widget/accounts/<account>?details=true` | – |
| `teamtailor` | `<site>/jobs.rss` | – |
| `recruitee` | `<slug>.recruitee.com/api/offers/` | – |
| `personio` | `<slug>.jobs.personio.com/xml` | – |
| `email` | IMAP, read-only | LinkedIn and Indeed job-alert emails (off until enabled) |

How to add a company:
* **Greenhouse:** check that `https://boards-api.greenhouse.io/v1/boards/<slug>/jobs` returns JSON.
* **Lever:** check `https://api.lever.co/v0/postings/<slug>?mode=json`, or use `region: eu` for `api.eu.lever.co`.
* **SmartRecruiters:** check `https://api.smartrecruiters.com/v1/companies/<id>/postings?country=ro`.
* **Workable:** the account is the part after `apply.workable.com/`. Check
  `https://apply.workable.com/api/v1/widget/accounts/<account>?details=true`, and check that its `name` is the
  company you mean. The widget lists a job once per location; jobbot merges them.
* **Teamtailor:** `site` is `<slug>.teamtailor.com` or the company's own career domain; check `<site>/jobs.rss`.
  Add `default_location` if its postings name no city.
* **Recruitee:** check `https://<slug>.recruitee.com/api/offers/`.
* **Personio:** check `https://<slug>.jobs.personio.com/xml` (`domain: de` for `.jobs.personio.de`), and read
  Personio's terms first. Jobs published only in another language are fetched again without `language=en`.
* **Workday:** take tenant, `wd` number and site from `https://<tenant>.wd<N>.myworkdayjobs.com/<site>`. jobbot finds
  the tenant's country facet automatically. If the tenant has no country facet, it falls back to office locations
  that name a city in your area.
* Slugs collide: `yonder.teamtailor.com` and `tazz.jobs.personio.com` belong to different companies than you might
  expect. Check the company name and locations in the feed before adding it.

**Job boards not used:**
* **LinkedIn** is never scraped; its alert emails are read instead.
* **BestJobs.eu:** its terms ([/legal/terms](https://www.bestjobs.eu/legal/terms), section 3) ban "colectarea
  neautorizată de date (scraping)". They allow automated access only through BestJobs' official API or webhooks.
* **Hipo.ro:** its terms ([/locuri-de-munca/termeni](https://www.hipo.ro/locuri-de-munca/termeni)) allow
  "o singura copie ... doar in scopuri personale" (a single copy, for personal use only). They also state that
  "compilatia (adica adunarea si rearanjarea) continutului" (compiling, i.e. gathering and rearranging, the content) is
  Hipo's exclusive right. Hipo offers no RSS feed or API.
* **eJobs.ro:** robots.txt advertises an official RSS feed (`https://www.ejobs.ro/rss-listings.xml`), which would be the
  right way in, but its terms haven't been reviewed yet.

## Project layout

```
config.example.yaml  template for config.yaml (profile, locations, llm, notify, control, sources)
.env.example         template for secrets (.env itself is gitignored)
jobbot/              the package: cli, app, pipeline, filters, db, http, notify, control, guard, panel, ...
jobbot/sources/      one module per source kind, plus email parsers
jobbot/scoring/      Scorer interface, Ollama and Anthropic backends
calibration/         labels.example.yaml (format) and sample_cv.txt; your own set stays local
scripts/             Windows autostart wrapper and install/uninstall scripts
tests/               pytest suite with recorded fixtures
data/                SQLite database, logs, lock file (created at runtime, gitignored)
```

## Tests

```bash
python -m pytest -q
```

The tests make no live calls:
* The source parsers run against recorded, trimmed real responses in `tests/fixtures/` (email fixtures are
  redacted).
* HTTP goes through `httpx.MockTransport`.
* Ollama is a fake server that returns valid, malformed and error responses; the Anthropic client is a fake too.
* The control panel is tested with Textual's `run_test` pilot, and the GPU guard with a fake `nvidia-smi`.
