"""Redis Streams task/event transport with local dependency guards."""
from __future__ import annotations
import json, os, socket, time

TASK_STREAM = os.getenv("REDIS_TASK_STREAM", "research:tasks")
GROUP = os.getenv("REDIS_CONSUMER_GROUP", "industry-agent-workers")

class RedisStreams:
    def __init__(self, url=None):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Redis execution requires redis[hiredis]") from exc
        self.client = redis.Redis.from_url(url or os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        self.client.ping()
        try: self.client.xgroup_create(TASK_STREAM, GROUP, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc): raise

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def enqueue(self, payload):
        return self.client.xadd(TASK_STREAM, {"payload": json.dumps(payload, ensure_ascii=False)}, maxlen=10000, approximate=True)

    def read(self, consumer=None, count=1, block_ms=5000):
        consumer = consumer or os.getenv("REDIS_CONSUMER_NAME") or socket.gethostname()
        rows=self.client.xreadgroup(GROUP, consumer, {TASK_STREAM:'>'}, count=count, block=block_ms)
        return [(sid, json.loads(fields.get("payload", "{}"))) for _, entries in rows for sid, fields in entries]

    def ack(self, stream_id): return self.client.xack(TASK_STREAM, GROUP, stream_id)

    def claim_stale(self, consumer=None, min_idle_ms=None, count=20):
        consumer = consumer or os.getenv("REDIS_CONSUMER_NAME") or socket.gethostname()
        try:
            result=self.client.xautoclaim(TASK_STREAM, GROUP, consumer, min_idle_ms or int(os.getenv("REDIS_CLAIM_IDLE_MS","60000")), "0-0", count=count)
            entries=result[1] if isinstance(result, tuple) else []
            return [(sid, json.loads(fields.get("payload","{}"))) for sid, fields in entries]
        except Exception: return []

    def publish_event(self, run_id, event):
        data=dict(event); data.setdefault("run_id", run_id); data.setdefault("created_at", time.time())
        return self.client.xadd(f"research:events:{run_id}", {"payload": json.dumps(data, ensure_ascii=False)}, maxlen=int(os.getenv("REDIS_EVENT_RETENTION","1000")), approximate=True)

    def events(self, run_id, after="0-0", block_ms=5000, count=50):
        rows=self.client.xread({f"research:events:{run_id}": after}, count=count, block=block_ms)
        return [(sid, json.loads(fields.get("payload","{}"))) for _, entries in rows for sid, fields in entries]
