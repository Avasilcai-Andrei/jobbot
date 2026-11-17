"""Extract the CV text once and cache it next to the database."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from pypdf import PdfReader

log = logging.getLogger(__name__)

EXTRACTOR_VERSION = "2"  # bump to re-extract cached CVs after changing the extraction
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d \-().]{7,}\d")


def _glued_words(text: str) -> int:
    """Long letter runs mean the extractor lost the spaces between words."""
    return len(re.findall(r"[^\W\d_]{25,}", text))


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    candidates = []
    for mode in ("layout", "plain"):
        try:
            candidates.append("\n\n".join(p.extract_text(extraction_mode=mode) or "" for p in reader.pages))
        except Exception as e:  # layout mode can fail on unusual PDFs
            log.debug("pypdf %s extraction failed: %s", mode, e)
    if not candidates:
        return ""
    return min(candidates, key=_glued_words)


def clean_cv_text(text: str) -> str:
    # Contact details don't help matching; keep them out of every prompt.
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_cv_text(cv_path: Path, cache_dir: Path) -> str | None:
    if not cv_path.exists():
        log.warning("CV not found at %s; the scoring stage needs it", cv_path)
        return None
    if cv_path.suffix.lower() in (".txt", ".md"):
        return clean_cv_text(cv_path.read_text(encoding="utf-8"))
    data = cv_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()[:16]
    cache = cache_dir / f"cv_v{EXTRACTOR_VERSION}_{digest}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    text = clean_cv_text(extract_pdf_text(cv_path))
    if len(text) < 200:
        log.warning("Only %d characters extracted from %s; is it a scanned PDF?", len(text), cv_path)
    if (glued := _glued_words(text)) > 5:
        log.warning("CV text from %s has %d run-together words; consider a .txt CV instead", cv_path, glued)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    log.info("Extracted %d characters from %s -> %s", len(text), cv_path.name, cache)
    return text
