"""Independent Redis Streams worker for production execution mode."""
from __future__ import annotations
import asyncio, json, os, time

from queue_backend import RedisStreams


async def process(queue: RedisStreams, stream_id: str, payload: dict):
    import web_server
    run_id = payload["run_id"]; session_id = payload["session_id"]
    run = web_server.RUN_STORE.get(run_id)
    if not run or run.get("status") in {"succeeded", "cancelled"}:
        queue.ack(stream_id); return
    acquired = None
    try:
        acquired = web_server.RUN_STORE.acquire(run_id, owner_id=f"redis-worker-{os.getpid()}")
        web_server.RUN_STORE.update(run_id, fencing_token=acquired["fencing_token"], checkpoint_ref=session_id, current_node="supervisor")
        async for event in web_server.event_generator(session_id, payload.get("message", ""), payload.get("user_id", "anonymous"), run_id, force_local=True):
            raw = event.get("data", "{}") if isinstance(event, dict) else "{}"
            try: decoded = json.loads(raw)
            except Exception: decoded = {"type": event.get("event", "message"), "data": raw}
            queue.publish_event(run_id, decoded)
        queue.ack(stream_id)
    except Exception as exc:
        current = web_server.RUN_STORE.get(run_id) or {}
        if current.get("status") not in {"succeeded", "cancelled", "timed_out"}:
            web_server.RUN_STORE.update(run_id, fencing_token=acquired.get("fencing_token") if acquired else None,
                                        status="failed", finished_at=time.time(), error_code=type(exc).__name__, error_message=str(exc)[:500])
        queue.publish_event(run_id, {"type": "error", "message": str(exc)[:500]})
        queue.ack(stream_id)


async def main():
    queue = RedisStreams()
    consumer = os.getenv("REDIS_CONSUMER_NAME") or f"worker-{os.getpid()}"
    while True:
        entries = queue.claim_stale(consumer=consumer)
        if not entries:
            entries = queue.read(consumer=consumer, count=1, block_ms=5000)
        for stream_id, payload in entries:
            await process(queue, stream_id, payload)


if __name__ == "__main__":
    asyncio.run(main())
