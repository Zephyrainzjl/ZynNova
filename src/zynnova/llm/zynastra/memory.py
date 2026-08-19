"""Small persistent SQLite conversation store."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .types import Message, ToolCall


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS messages (session TEXT, seq INTEGER, payload TEXT, PRIMARY KEY(session, seq))")

    def _connect(self): return sqlite3.connect(self.path)

    def load(self, session: str) -> list[Message]:
        with self._connect() as db:
            rows = db.execute("SELECT payload FROM messages WHERE session=? ORDER BY seq", (session,)).fetchall()
        result=[]
        for (raw,) in rows:
            obj=json.loads(raw)
            calls=tuple(ToolCall(c["id"], c["name"], c.get("arguments",{})) for c in obj.get("tool_calls",[]))
            result.append(Message(obj["role"], obj.get("content"), obj.get("name"), obj.get("tool_call_id"), calls))
        return result

    def save(self, session: str, messages: list[Message]) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM messages WHERE session=?", (session,))
            for i,m in enumerate(messages):
                payload={
                    "role":m.role,"content":m.content,"name":m.name,"tool_call_id":m.tool_call_id,
                    "tool_calls":[{"id":c.id,"name":c.name,"arguments":dict(c.arguments)} for c in m.tool_calls],
                }
                db.execute("INSERT INTO messages VALUES(?,?,?)", (session,i,json.dumps(payload,ensure_ascii=False)))


__all__ = ["SessionStore"]
