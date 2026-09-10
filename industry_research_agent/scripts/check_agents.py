"""Fast static consistency checks that do not call external services."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "market_analyst.py": ["research_dimensions", "市场", "趋势", "机会"],
    "competition_analyst.py": ["research_dimensions", "竞争", "风险"],
    "business_analyst.py": ["research_dimensions", "calculate_roi", "awaiting_user"],
    "supervisor.py": ["dimension_status", "WAIT_USER", "MAX_AGENT_STEPS"],
}
FORBIDDEN = ["knowledge_upsert", "retrieve_knowledge", "market_data_mock", "query_market_data", "chromadb", "sentence_transformers", "EMBEDDING_MODEL", "CHROMA_PERSIST_DIR"]


def main() -> int:
    errors = []
    for filename, required in EXPECTED.items():
        text = (ROOT / "agents" / filename).read_text(encoding="utf-8")
        for marker in required:
            if marker not in text:
                errors.append(f"{filename}: missing {marker}")
        for marker in FORBIDDEN:
                if marker in text:
                    errors.append(f"{filename}: forbidden production dependency {marker}")
    for legacy in ("chroma_db", "knowledge_packs"):
        if (ROOT / legacy).exists():
            errors.append(f"legacy local knowledge directory still exists: {legacy}")
    if errors:
        print("\n".join(errors))
        return 1
    print("Agent consistency check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
