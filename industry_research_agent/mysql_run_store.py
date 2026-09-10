"""MySQL Run Store backend.

The public API intentionally mirrors :class:`run_store.RunStore`.  SQLAlchemy's
asyncmy engine runs on a private event-loop thread so existing synchronous
Starlette/worker call sites remain compatible.
"""
from __future__ import annotations

import asyncio, hashlib, os, threading, time, uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from run_store import RUN_STATES, TERMINAL_STATES


class _LoopRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._serve, daemon=True, name="mysql-run-store")
        self.thread.start()

    def _serve(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, factory):
        return asyncio.run_coroutine_threadsafe(factory(), self.loop).result()

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)


class MySQLRunStore:
    def __init__(self, dsn: str | None = None, *, lease_seconds: int = 45,
                 pool_size: int | None = None, max_overflow: int | None = None):
        self.dsn = dsn or os.getenv("MYSQL_DSN", "mysql+asyncmy://industry:industry@mysql:3306/industry_agent")
        self.lease_seconds = max(5, int(lease_seconds))
        self.runner = _LoopRunner()
        self.engine = self.runner.call(lambda: self._init_engine(pool_size, max_overflow))

    async def _init_engine(self, pool_size, max_overflow):
        engine = create_async_engine(
            self.dsn, pool_size=int(pool_size or os.getenv("MYSQL_POOL_SIZE", "10")),
            max_overflow=int(max_overflow or os.getenv("MYSQL_MAX_OVERFLOW", "20")),
            pool_pre_ping=True, pool_recycle=1800,
        )
        async with engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS runs (
                  run_id VARCHAR(64) PRIMARY KEY, session_id VARCHAR(128) NOT NULL,
                  user_id VARCHAR(128) NOT NULL DEFAULT 'anonymous', request_hash CHAR(64) NOT NULL,
                  request_text TEXT NOT NULL, status VARCHAR(24) NOT NULL, attempt INT NOT NULL DEFAULT 0,
                  owner_id VARCHAR(128), lease_expires_at DOUBLE, fencing_token BIGINT NOT NULL DEFAULT 0,
                  progress DOUBLE NOT NULL DEFAULT 0, current_node VARCHAR(128), checkpoint_ref VARCHAR(255),
                  artifact_ref VARCHAR(512), error_code VARCHAR(128), error_message TEXT,
                  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE, created_at DOUBLE NOT NULL,
                  started_at DOUBLE, heartbeat_at DOUBLE, finished_at DOUBLE,
                  UNIQUE KEY uq_runs_session_hash (session_id, request_hash),
                  KEY ix_runs_status_created (status, created_at),
                  KEY ix_runs_lease_status (lease_expires_at, status),
                  KEY ix_runs_session_created (session_id, created_at)
                ) ENGINE=InnoDB
            """))
        return engine

    @staticmethod
    def request_hash(message: str) -> str:
        return hashlib.sha256(" ".join(str(message).split()).encode()).hexdigest()

    @staticmethod
    def _row(row):
        return dict(row) if row else None

    def _call(self, fn):
        return self.runner.call(fn)

    def create_or_get(self, session_id: str, *, user_id: str, message: str) -> dict:
        async def op():
            digest = self.request_hash(message); now = time.time()
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT * FROM runs WHERE session_id=:s AND request_hash=:h FOR UPDATE"), {"s": session_id, "h": digest})
                existing = result.mappings().first()
                if existing and existing["status"] not in TERMINAL_STATES:
                    return dict(existing)
                if existing:
                    digest = hashlib.sha256(f"{digest}:{now}:{uuid.uuid4()}".encode()).hexdigest()
                run_id = str(uuid.uuid4())
                await conn.execute(text("INSERT INTO runs (run_id,session_id,user_id,request_hash,request_text,status,created_at) VALUES (:r,:s,:u,:h,:m,'queued',:t)"), {"r":run_id,"s":session_id,"u":user_id or "anonymous","h":digest,"m":str(message),"t":now})
                row = (await conn.execute(text("SELECT * FROM runs WHERE run_id=:r"), {"r":run_id})).mappings().first()
                return dict(row)
        return self._call(op)

    def get(self, run_id):
        async def op():
            async with self.engine.connect() as conn:
                row = (await conn.execute(text("SELECT * FROM runs WHERE run_id=:r"), {"r":run_id})).mappings().first()
                return self._row(row)
        return self._call(op)

    def latest_for_session(self, session_id): return self._query_one("SELECT * FROM runs WHERE session_id=:s ORDER BY created_at DESC LIMIT 1", {"s":session_id})
    def active_for_session(self, session_id): return self._query_one("SELECT * FROM runs WHERE session_id=:s AND status IN ('queued','running','retrying') AND cancel_requested=FALSE ORDER BY created_at DESC LIMIT 1", {"s":session_id})
    def _query_one(self, sql, params):
        async def op():
            async with self.engine.connect() as conn:
                row = (await conn.execute(text(sql), params)).mappings().first(); return self._row(row)
        return self._call(op)

    def list_recent(self, *, limit=50): return self._query_many("SELECT * FROM runs ORDER BY created_at DESC LIMIT :n", {"n":max(1,min(int(limit),100))})
    def list_queued(self, *, limit=100): return self._query_many("SELECT * FROM runs WHERE status='queued' AND cancel_requested=FALSE ORDER BY created_at LIMIT :n", {"n":max(1,min(int(limit),100))})
    def _query_many(self, sql, params):
        async def op():
            async with self.engine.connect() as conn:
                return [dict(x) for x in (await conn.execute(text(sql), params)).mappings().all()]
        return self._call(op)

    def acquire(self, run_id, *, owner_id=None):
        async def op():
            now=time.time(); owner=owner_id or f"worker-{uuid.uuid4().hex[:12]}"
            async with self.engine.begin() as conn:
                row=(await conn.execute(text("SELECT * FROM runs WHERE run_id=:r FOR UPDATE"),{"r":run_id})).mappings().first()
                if not row: raise KeyError(f"run not found: {run_id}")
                if row["cancel_requested"]:
                    await conn.execute(text("UPDATE runs SET status='cancelled',finished_at=:t,error_code='cancelled' WHERE run_id=:r"),{"t":now,"r":run_id})
                else:
                    await conn.execute(text("UPDATE runs SET status='running',attempt=:a,owner_id=:o,lease_expires_at=:e,heartbeat_at=:t,fencing_token=:f,started_at=COALESCE(started_at,:t) WHERE run_id=:r"),{"a":int(row["attempt"] or 0)+1,"o":owner,"e":now+self.lease_seconds,"t":now,"f":int(row["fencing_token"] or 0)+1,"r":run_id})
                out=(await conn.execute(text("SELECT * FROM runs WHERE run_id=:r"),{"r":run_id})).mappings().first(); return dict(out)
        return self._call(op)

    def update(self, run_id, *, fencing_token=None, **values):
        allowed={"status","progress","current_node","checkpoint_ref","artifact_ref","error_code","error_message","cancel_requested","finished_at","heartbeat_at","lease_expires_at","owner_id","attempt"}
        values={k:v for k,v in values.items() if k in allowed}
        if "status" in values and values["status"] not in RUN_STATES: raise ValueError(f"invalid run status: {values['status']}")
        async def op():
            if fencing_token is not None: values["fencing_token"]=fencing_token
            if not values: return False
            assignments=", ".join(f"{k}=:{k}" for k in values); params=dict(values); params["r"]=run_id
            guard=" AND fencing_token=:fencing_token AND status NOT IN ('succeeded','failed','timed_out','cancelled')" if fencing_token is not None else ""
            async with self.engine.begin() as conn:
                result=await conn.execute(text(f"UPDATE runs SET {assignments} WHERE run_id=:r{guard}"),params); return result.rowcount==1
        return self._call(op)

    def heartbeat(self, run_id, *, fencing_token, progress=None, current_node=None):
        values={"heartbeat_at":time.time(),"lease_expires_at":time.time()+self.lease_seconds}
        if progress is not None: values["progress"]=max(0,min(1,float(progress)))
        if current_node is not None: values["current_node"]=current_node
        return self.update(run_id, fencing_token=fencing_token, **values)

    def request_cancel(self, run_id):
        row=self.get(run_id)
        if not row: return None
        vals={"cancel_requested":True,"error_code":"cancel_requested"}
        if row["status"]=="queued": vals.update(status="cancelled",finished_at=time.time())
        self.update(run_id, **vals); return self.get(run_id)

    def recover_stale(self):
        async def op():
            now=time.time(); recovered=failed=0
            async with self.engine.begin() as conn:
                rows=(await conn.execute(text("SELECT run_id,checkpoint_ref FROM runs WHERE status IN ('running','retrying') AND lease_expires_at<:t FOR UPDATE"),{"t":now})).mappings().all()
                for row in rows:
                    if row["checkpoint_ref"]:
                        await conn.execute(text("UPDATE runs SET status='queued',owner_id=NULL,lease_expires_at=NULL,error_code='worker_lease_expired' WHERE run_id=:r"),{"r":row["run_id"]}); recovered+=1
                    else:
                        await conn.execute(text("UPDATE runs SET status='failed',finished_at=:t,error_code='stale_run_without_checkpoint' WHERE run_id=:r"),{"r":row["run_id"],"t":now}); failed+=1
            return {"recovered":recovered,"failed":failed}
        return self._call(op)

    def close(self):
        try: self._call(self.engine.dispose)
        finally: self.runner.close()

    def ping(self) -> bool:
        async def op():
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        try:
            return bool(self._call(op))
        except Exception:
            return False
