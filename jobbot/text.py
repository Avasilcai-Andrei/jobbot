"""Text helpers: HTML to plain text, accent folding, cross-post keys."""
from __future__ import annotations

import html
import re
import unicodedata
from html.parser import HTMLParser

_BLOCK_TAGS = {"p", "br", "li", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol", "section"}


class _Stripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(markup: str) -> str:
    s = _Stripper()
    s.feed(markup)
    s.close()
    text = "".join(s.parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def unescape_html_to_text(escaped: str) -> str:
    """For APIs (e.g. Greenhouse) that return entity-escaped HTML."""
    return html_to_text(html.unescape(escaped))


def fold(s: str) -> str:
    """Lowercase and strip diacritics (ș -> s, ă -> a)."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


_COMPANY_NOISE = re.compile(
    r"\b(s\.?\s?r\.?\s?l|s\.?\s?a|ltd|limited|inc|gmbh|llc|plc|bv|ag|corp|corporation|group|romania)\b\.?"
)
_GENDER = re.compile(r"\(\s*[mfwdx]\s*(/\s*[mfwdx]\s*)+\)")
_TITLE_NOISE = re.compile(
    r"\b(remote|hybrid|on-?site|cluj|napoca|bucharest|bucuresti|iasi|timisoara|brasov|romania|ro)\b"
)


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", s)).strip()


def norm_company(company: str) -> str:
    return _squash(_COMPANY_NOISE.sub(" ", fold(company)))


def norm_title(title: str) -> str:
    t = _GENDER.sub(" ", fold(title))
    return _squash(_TITLE_NOISE.sub(" ", t))


def crosspost_key(company: str, title: str) -> str:
    return f"{norm_company(company)}|{norm_title(title)}"
