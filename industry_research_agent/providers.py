"""Web research provider abstraction: MCP primary, direct SDK fallback, fake for tests."""
from __future__ import annotations

import json
import os
import time
from typing import Protocol

from contracts import PageResponse, SearchResponse
from research_tools import fetch_page as direct_fetch_page
from research_tools import search_web as direct_search_web


class ResearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5, depth: str = "advanced") -> SearchResponse: ...
    def fetch_page(self, url: str) -> PageResponse: ...


def _parse_mcp_json(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception as exc:
        raise ValueError(f"invalid MCP JSON: {exc}") from exc


class DirectResearchProvider:
    name = "direct_fallback"

    def search(self, query, max_results=5, depth="advanced"):
        results, event = direct_search_web(query, max_results, depth)
        normalized = []
        for item in results:
            try:
                from contracts import SearchResult
                normalized.append(SearchResult(**item))
            except Exception:
                continue
        return SearchResponse(results=normalized, provider=self.name, cached=bool(event.get("cached")), event=event)

    def fetch_page(self, url):
        page, event = direct_fetch_page(url)
        return PageResponse(url=page["url"], title=page.get("title", ""), text=page.get("text", ""), provider=self.name, event=event)


class McpResearchProvider:
    name = "mcp"

    def __init__(self):
        from mcp_servers.mcp_client import load_web_research_tools
        self.tools = {tool.name: tool for tool in load_web_research_tools()}
        if "search" not in self.tools or "fetch_page" not in self.tools:
            raise RuntimeError("MCP server did not expose search and fetch_page")
        self.last_retry_count = 0

    def _invoke(self, name, args):
        last = None
        self.last_retry_count = 0
        for attempt in range(2):
            try:
                return _parse_mcp_json(self.tools[name].invoke(args))
            except Exception as exc:
                last = exc
                self.last_retry_count = attempt + 1
                time.sleep(0.2)
        raise RuntimeError(f"MCP {name} failed: {last}")

    def search(self, query, max_results=5, depth="advanced"):
        payload = self._invoke("search", {"query": query, "max_results": max_results, "depth": depth})
        from contracts import SearchResult
        results = [SearchResult(**item) for item in payload.get("results", [])]
        event = dict(payload.get("event") or {})
        event.setdefault("tool_source", self.name)
        event["retry_count"] = self.last_retry_count
        return SearchResponse(results=results, provider=self.name, cached=bool(event.get("cached")), event=event)

    def fetch_page(self, url):
        payload = self._invoke("fetch_page", {"url": url})
        page = payload.get("page", {})
        event = dict(payload.get("event") or {})
        event.setdefault("tool_source", self.name)
        event["retry_count"] = self.last_retry_count
        return PageResponse(url=page.get("url", url), title=page.get("title", ""), text=page.get("text", ""), provider=self.name, event=event)


class ResilientResearchProvider:
    """Lazy MCP primary with one direct fallback; exposes the active provider."""
    name = "resilient"

    def __init__(self, prefer_mcp: bool = True):
        self.prefer_mcp = prefer_mcp
        self._primary = None
        self._fallback = DirectResearchProvider()
        self.active_provider = "uninitialized"
        self.last_failure = None

    def _get_primary(self):
        if self._primary is None and self.prefer_mcp:
            try:
                self._primary = McpResearchProvider()
            except Exception:
                self._primary = False
        return self._primary

    def search(self, query, max_results=5, depth="advanced"):
        primary = self._get_primary()
        try:
            if primary:
                self.active_provider = primary.name
                return primary.search(query, max_results, depth)
        except Exception as exc:
            self.last_failure = {"provider": "mcp", "error": type(exc).__name__, "message": str(exc)[:300]}
        self.active_provider = self._fallback.name
        return self._fallback.search(query, max_results, depth)

    def fetch_page(self, url):
        primary = self._get_primary()
        try:
            if primary:
                self.active_provider = primary.name
                return primary.fetch_page(url)
        except Exception as exc:
            self.last_failure = {"provider": "mcp", "error": type(exc).__name__, "message": str(exc)[:300]}
        self.active_provider = self._fallback.name
        return self._fallback.fetch_page(url)


_DEFAULT_PROVIDER = ResilientResearchProvider(prefer_mcp=os.getenv("RESEARCH_PROVIDER", "mcp") != "direct")


def get_default_provider() -> ResilientResearchProvider:
    return _DEFAULT_PROVIDER


class FakeResearchProvider:
    """Deterministic provider for unit tests and offline ablation scaffolding."""
    name = "fake"

    def __init__(self, results=None, pages=None):
        self.results = results or []
        self.pages = pages or {}

    def search(self, query, max_results=5, depth="advanced"):
        from contracts import SearchResult
        return SearchResponse(
            results=[SearchResult(**item) for item in self.results[:max_results]],
            provider=self.name,
            cached=False,
            event={"tool_source": self.name, "success": bool(self.results)},
        )

    def fetch_page(self, url):
        page = self.pages.get(url, {"title": "Fake page", "text": "Fake evidence text with market size 100亿元 in 2026."})
        return PageResponse(url=url, title=page.get("title", "Fake page"), text=page.get("text", ""), provider=self.name,
                            event={"tool_source": self.name, "success": bool(page.get("text"))})
