"""User-visible output formatting contracts."""
from __future__ import annotations

import re


def to_plain_text(value: object) -> str:
    """Normalize model output to readable plain text while preserving citations."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```[^\n]*\n?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1（\2）", text)
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-_:| ]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s?\|\s?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s?\|\s?", "　", text)
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
