"""Helpers shared by the email parsers."""
from __future__ import annotations

from email.message import EmailMessage

from ...text import html_to_text


def body_text(msg: EmailMessage) -> str:
    """The text/plain body, or the HTML body turned into text when there is no plain part."""
    part = msg.get_body(preferencelist=("plain",))
    if part is not None:
        return part.get_content()
    part = msg.get_body(preferencelist=("html",))
    return html_to_text(part.get_content()) if part is not None else ""


def clean(line: str) -> str:
    return " ".join(line.replace(" ", " ").split())
