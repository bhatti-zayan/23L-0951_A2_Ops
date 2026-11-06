"""Database trail (Task 5): every prediction becomes one permanent, queryable row."""
import time
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, select,
                        text)

metadata = MetaData()
predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),   # WHEN
    Column("filename", String(255)),
    Column("verdict", String(10), nullable=False),                   # WHAT was decided
    Column("band", String(40)),
    Column("p_ai", Float, nullable=False),                           # HOW confidently
    Column("confidence", Float),
    Column("threshold", Float),
    Column("model_name", String(100)),                               # BY WHICH model
    Column("model_version", String(20)),
    Column("source", String(200)),                                   # local model or remote API
    Column("latency_ms", Float),
)


class Store:
    def __init__(self, url, retries=15, wait=2):
        self.engine = create_engine(url, pool_pre_ping=True)
        for attempt in range(1, retries + 1):        # the DB container may still be starting
            try:
                metadata.create_all(self.engine)
                break
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(wait)

    def add(self, **row):
        with self.engine.begin() as conn:
            conn.execute(predictions.insert().values(created_at=datetime.now(timezone.utc), **row))

    def recent(self, limit=200):
        with self.engine.connect() as conn:
            rows = conn.execute(select(predictions).order_by(predictions.c.id.desc()).limit(limit))
            return [dict(r._mapping) for r in rows]

    def ping(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
