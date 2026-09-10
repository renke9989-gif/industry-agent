"""Validated workflow blueprints that compile only to registered graph nodes."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


NodeName = Literal["market", "competition", "business"]
DimensionName = Literal["市场", "竞争", "商业模式", "机会", "风险", "趋势"]
NODE_REGISTRY = {
    "market": {"graph_node": "market_analyst", "dimensions": {"市场", "趋势", "机会"}},
    "competition": {"graph_node": "competition_analyst", "dimensions": {"竞争", "风险"}},
    "business": {"graph_node": "business_analyst", "dimensions": {"商业模式"}},
}


class WorkflowBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["1"] = "1"
    scenario: str = Field(min_length=1, max_length=80)
    nodes: list[NodeName] = Field(min_length=1, max_length=3)
    required_dimensions: list[DimensionName] = Field(min_length=1, max_length=6)
    max_tool_calls: int = Field(default=12, ge=1, le=20)
    timeout_seconds: int = Field(default=90, ge=15, le=300)
    requires_user_confirmation: bool = True
    tools: list[Literal["search", "fetch_page", "extract_evidence"]] = Field(
        default_factory=lambda: ["search", "fetch_page", "extract_evidence"]
    )

    @model_validator(mode="after")
    def dimensions_must_be_covered(self):
        covered = set().union(*(NODE_REGISTRY[name]["dimensions"] for name in self.nodes))
        missing = set(self.required_dimensions) - covered
        if missing:
            raise ValueError("required dimensions are not covered by registered nodes: " + ", ".join(sorted(missing)))
        if len(self.nodes) != len(set(self.nodes)):
            raise ValueError("duplicate workflow nodes are not allowed")
        return self


def compile_blueprint(blueprint: WorkflowBlueprint | dict) -> dict:
    """Compile a validated blueprint to declarative existing-node routing."""
    model = blueprint if isinstance(blueprint, WorkflowBlueprint) else WorkflowBlueprint.model_validate(blueprint)
    return {
        "version": model.version,
        "scenario": model.scenario,
        "graph_nodes": [NODE_REGISTRY[name]["graph_node"] for name in model.nodes],
        "research_plan": [{"dimensions": list(model.required_dimensions), "mode": "evidence"}],
        "limits": {"max_tool_calls": model.max_tool_calls, "timeout_seconds": model.timeout_seconds},
        "requires_user_confirmation": model.requires_user_confirmation,
        "tools": list(model.tools),
    }


def blueprint_from_state(state: dict) -> WorkflowBlueprint:
    dimensions = list(dict.fromkeys(
        dimension for step in state.get("research_plan", []) for dimension in step.get("dimensions", [])
    )) or ["市场", "竞争", "商业模式"]
    nodes = []
    for name, contract in NODE_REGISTRY.items():
        if set(dimensions) & contract["dimensions"]:
            nodes.append(name)
    return WorkflowBlueprint(
        scenario=state.get("scenario") or "unknown",
        nodes=nodes or ["market"],
        required_dimensions=dimensions,
        requires_user_confirmation=bool(state.get("needs_interview", True)),
    )
