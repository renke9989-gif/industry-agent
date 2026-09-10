"""Unified, web-only research tools.

The production path intentionally has no vector database. Search responses are
normalized, optionally enriched with page text, and converted into evidence
records that can be cited by the final report.
"""
from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import re
import socket
from html.parser import HTMLParser
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx
try:
    from bs4 import BeautifulSoup
except ImportError:  # Python 3.11 minimal install fallback
    BeautifulSoup = None

from state import ToolEvent
from contracts import Evidence
from learning_loop import log_error


MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "12000"))
CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "3600"))
REQUEST_TIMEOUT = float(os.getenv("WEB_REQUEST_TIMEOUT", "12"))
USER_AGENT = "IndustryResearchAgent/2.0 (+educational portfolio project)"

_cache: Dict[str, tuple[float, List[dict]]] = {}
_cache_lock = threading.Lock()

AUTHORITATIVE_SUFFIXES = (
    ".gov.cn", ".gov", ".edu.cn", ".edu", ".org.cn",
)
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"忽略.{0,8}(之前|以上|系统).{0,8}(指令|提示)",
    r"system\s*prompt",
    r"<\|(?:system|assistant|user)\|>",
    r"do\s+not\s+follow\s+the\s+user",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_source(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    if any(host.endswith(suffix) for suffix in AUTHORITATIVE_SUFFIXES):
        return "official"
    if any(key in host for key in ("stats", "gov", "cninfo", "sse.com", "szse.cn", "ndrc", "mofcom")):
        return "official"
    if any(key in host for key in ("research", "iresearch", "askci", "qianzhan", "iiimedia")):
        return "research"
    if any(key in host for key in ("company", "annualreports", "hkex", "eastmoney")):
        return "company"
    if any(key in host for key in ("reuters", "bloomberg", "caixin", "36kr", "yicai")):
        return "media"
    return "web"


def source_confidence(url: str, score: float = 0.0) -> float:
    base = {"official": 0.92, "company": 0.84, "research": 0.80, "media": 0.78, "web": 0.62}[classify_source(url)]
    if score:
        base = min(0.98, base * 0.75 + max(0.0, min(score, 1.0)) * 0.25)
    return round(base, 2)


def sanitize_web_text(text: str) -> str:
    """Strip markup and neutralize common prompt-injection phrases."""
    cleaned = html.unescape(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "[untrusted instruction removed]", cleaned, flags=re.I)
    return cleaned[:MAX_PAGE_CHARS]


class _TextParser(HTMLParser):
    """Small stdlib fallback when beautifulsoup4 is not installed."""
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript", "nav", "footer", "form"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript", "nav", "footer", "form"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _cache_key(query: str, max_results: int, depth: str) -> str:
    raw = json.dumps([query.strip().lower(), max_results, depth], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_result(item: dict) -> Optional[dict]:
    url = item.get("url") or item.get("href") or ""
    if not url.startswith(("http://", "https://")):
        return None
    return {
        "title": sanitize_web_text(item.get("title", ""))[:300],
        "url": url,
        "snippet": sanitize_web_text(item.get("content") or item.get("body") or "")[:2500],
        "score": float(item.get("score") or 0.0),
        "published_at": str(item.get("published_date") or item.get("published_at") or item.get("date") or ""),
        "source_type": classify_source(url),
    }


def _search_tavily(query: str, max_results: int, depth: str) -> List[dict]:
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    payload = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=max_results,
        search_depth=depth,
        include_answer=False,
        include_raw_content=False,
    )
    return [item for raw in payload.get("results", []) if (item := _normalize_result(raw))]


def _search_ddgs(query: str, max_results: int) -> List[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    with DDGS() as client:
        raw_results = list(client.text(query, region="cn-zh", safesearch="moderate", max_results=max_results))
    return [item for raw in raw_results if (item := _normalize_result(raw))]


def search_web(query: str, max_results: int = 5, depth: str = "advanced") -> tuple[List[dict], ToolEvent]:
    """Search Tavily and fall back to DDGS. Returns normalized results + trace."""
    started = time.perf_counter()
    key = _cache_key(query, max_results, depth)
    with _cache_lock:
        cached = _cache.get(key)
        if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
            results = [dict(item) for item in cached[1]]
            return results, {
                "tool": "search", "query": query, "success": True, "cached": True,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "result_count": len(results), "timestamp": utc_now(),
            }

    errors: List[str] = []
    results: List[dict] = []
    for provider in (_search_tavily, _search_ddgs):
        try:
            if provider is _search_tavily:
                results = provider(query, max_results, depth)
            else:
                results = provider(query, max_results)
            if results:
                break
        except Exception as exc:  # provider degradation is intentional
            errors.append(f"{provider.__name__}: {type(exc).__name__}: {str(exc)[:120]}")
            log_error("search_provider", str(exc), context={"provider": provider.__name__, "query": query[:300]})

    if results:
        with _cache_lock:
            _cache[key] = (time.time(), [dict(item) for item in results])
    event: ToolEvent = {
        "tool": "search", "query": query, "success": bool(results), "cached": False,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "result_count": len(results), "error": "; ".join(errors), "timestamp": utc_now(), "tool_source": "direct",
    }
    return results, event


def fetch_page(url: str) -> tuple[dict, ToolEvent]:
    """Fetch readable page text with strict size/time limits."""
    started = time.perf_counter()
    event: ToolEvent = {"tool": "fetch_page", "url": url, "success": False, "timestamp": utc_now(), "tool_source": "direct"}
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("only public http/https URLs are allowed")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, port)}
        if any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ValueError("private, loopback and link-local destinations are blocked")
        with httpx.Client(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml")):
                raise ValueError(f"unsupported content type: {content_type}")
            if BeautifulSoup is not None:
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "noscript", "nav", "footer", "form"]):
                    tag.decompose()
                title = sanitize_web_text(soup.title.get_text(" ") if soup.title else "")
                text = sanitize_web_text(soup.get_text(" ", strip=True))
            else:
                parser = _TextParser()
                parser.feed(response.text)
                title = ""
                text = sanitize_web_text(" ".join(parser.parts))
            if len(text) < 120:
                raise ValueError("page contains too little readable text")
            event.update(success=True, duration_ms=int((time.perf_counter() - started) * 1000))
            return {"url": str(response.url), "title": title, "text": text}, event
    except Exception as exc:
        log_error("fetch_page", str(exc), context={"url": url[:500]})
        event.update(error=f"{type(exc).__name__}: {str(exc)[:180]}", duration_ms=int((time.perf_counter() - started) * 1000))
        return {"url": url, "title": "", "text": ""}, event


def build_evidence(
    *, evidence_id: str, dimension: str, claim: str, source: dict,
    excerpt: str = "", value: str = "", unit: str = "", period: str = "",
    region: str = "全国",
) -> Evidence:
    url = source.get("url", "")
    source_type = classify_source(url)
    excerpt_text = sanitize_web_text(excerpt or source.get("snippet", ""))[:1200]
    title_text = source.get("title", "")[:300]
    reprint_markers = ("转载", "转自", "来源：", "source:")
    is_reprint = any(marker.lower() in f"{title_text} {excerpt_text}".lower() for marker in reprint_markers)
    published_at = source.get("published_at") or None
    confidence = source_confidence(url, float(source.get("score") or 0.0))
    if not published_at:
        confidence = max(0.0, confidence - 0.08)
    if is_reprint:
        confidence = max(0.0, confidence - 0.12)
    return Evidence(
        id=evidence_id,
        dimension=dimension,
        claim=claim.strip(),
        value=value.strip() or None,
        unit=unit.strip() or None,
        period=period.strip() or None,
        region=region,
        source_title=title_text,
        source_url=url,
        source_type=source_type,
        published_at=published_at,
        retrieved_at=utc_now(),
        excerpt=excerpt_text,
        is_primary_source=source_type == "official" and not is_reprint,
        is_reprint=is_reprint,
        confidence=round(confidence, 2),
    ).model_dump(mode="json")


def independent_source_count(evidence: Iterable[Evidence]) -> int:
    return len({urlparse(item.get("source_url", "")).netloc.lower() for item in evidence
                if item.get("source_url") and not item.get("is_reprint", False)})


def evidence_is_sufficient(evidence: List[Evidence]) -> bool:
    return any(item.get("source_type") == "official" and not item.get("is_reprint", False) for item in evidence) or independent_source_count(evidence) >= 2
