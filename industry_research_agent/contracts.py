"""Validated contracts shared by providers, state, and evaluation."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator


Dimension = Literal["市场", "竞争", "商业模式", "机会", "风险", "趋势"]
SourceType = Literal["official", "company", "research", "media", "web"]
DataType = Literal["fact", "estimate", "user_input", "calculation"]


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = ""
    url: AnyHttpUrl
    snippet: str = ""
    score: float = 0.0
    published_at: Optional[str] = None
    source_type: SourceType = "web"


class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(default_factory=list)
    provider: str
    cached: bool = False
    event: dict = Field(default_factory=dict)


class PageResponse(BaseModel):
    url: AnyHttpUrl
    title: str = ""
    text: str = ""
    provider: str
    event: dict = Field(default_factory=dict)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    dimension: Dimension
    claim: str
    value: Optional[str] = None
    unit: Optional[str] = None
    period: Optional[str] = None
    region: Optional[str] = None
    source_title: str
    source_url: AnyHttpUrl
    source_publisher: Optional[str] = None
    source_type: SourceType
    published_at: Optional[str] = None
    retrieved_at: str
    excerpt: str = Field(min_length=20)
    data_type: DataType = "fact"
    is_primary_source: bool = False
    is_reprint: bool = False
    confidence: float = Field(ge=0, le=1)

    @field_validator("value")
    @classmethod
    def value_not_blank(cls, value):
        return value.strip() if isinstance(value, str) else value


class EvidenceExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[Evidence] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)


class ReportClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    evidence_ids: List[str] = Field(default_factory=list)
    claim_type: Literal["fact", "estimate", "recommendation", "calculation"]


class UserCondition(BaseModel):
    id: str
    key: str
    value: str
    source: Literal["interview", "request", "profile"] = "interview"


class RecommendationTrace(BaseModel):
    id: str
    text: str
    user_condition_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    validation_metric: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)
    trace_status: Literal["validated", "insufficient"] = "validated"


class ClaimJudgment(BaseModel):
    claim_text: str
    evidence_ids: List[str] = Field(default_factory=list)
    verdict: Literal["entailed", "contradicted", "unknown"]
    rationale: str = ""
    judge_source: Literal["rules", "llm", "unavailable"] = "rules"
    confidence: float = Field(default=0.0, ge=0, le=1)


class ProviderError(BaseModel):
    code: str
    message: str
    provider: str
    retryable: bool = False
