"""Five-session concurrency smoke test for the local Web API."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx


DEFAULT_INPUTS = [
    "调研杭州咖啡店市场，预算50万",
    "调研宠物烘焙行业，关注竞争风险",
    "调研AI教育产品的市场机会",
    "调研机器人焊接产线扩展可行性",
    "调研跨境电商行业趋势",
]
SMOKE_INPUTS = [
    "我想做生意",
]


async def run_one(client: httpx.AsyncClient, base_url: str, text: str) -> dict:
    started = time.perf_counter()
    response = await client.post(f"{base_url}/api/sessions", timeout=10)
    response.raise_for_status()
    session_id = response.json()["session_id"]
    response = await client.post(
        f"{base_url}/api/sessions/{session_id}/messages",
        json={"message": text},
        timeout=None,
    )
    return {
        "session_id": session_id,
        "status_code": response.status_code,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "has_error_event": '"type":"error"' in response.text.replace(" ", ""),
    }


async def main_async(base_url: str, inputs: list[str]) -> dict:
    limits = httpx.Limits(max_connections=len(inputs), max_keepalive_connections=len(inputs))
    async with httpx.AsyncClient(limits=limits) as client:
        results = await asyncio.gather(*(run_one(client, base_url, text) for text in inputs))
    successful = [item for item in results if item["status_code"] == 200 and not item["has_error_event"]]
    durations = [item["duration_seconds"] for item in results]
    return {
        "concurrency": len(inputs),
        "task_success_rate": len(successful) / len(results) if results else 0,
        "unique_sessions": len({item["session_id"] for item in results}) == len(results),
        "average_latency": round(statistics.mean(durations), 2) if durations else None,
        "p95_latency": sorted(durations)[max(0, int(len(durations) * 0.95) - 1)] if durations else None,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--mode", choices=["smoke", "live"], default="smoke",
                        help="smoke 不调用搜索/LLM；live 运行真实研究并产生费用")
    args = parser.parse_args()
    source = SMOKE_INPUTS if args.mode == "smoke" else DEFAULT_INPUTS
    inputs = (source * ((args.count + len(source) - 1) // len(source)))[:args.count]
    try:
        output = asyncio.run(main_async(args.base_url, inputs))
    except httpx.ConnectError:
        output = {"error": f"无法连接 {args.base_url}，请先启动 web_server.py"}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
