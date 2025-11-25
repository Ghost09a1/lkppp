from typing import Any, Dict, List, Optional, Tuple

from . import db


class MemoryManager:
    def __init__(self, conn, llm_client, max_history: int, summary_every: int):
        self.conn = conn
        self.llm_client = llm_client
        self.max_history = max_history
        self.summary_every = max(1, summary_every)

    def get_context(self, character_id: int) -> List[Dict[str, str]]:
        rows = db.get_recent_messages(self.conn, character_id, self.max_history)
        messages = [{"role": role, "content": content} for role, content in rows]
        latest_summary = db.get_latest_summary(self.conn, character_id)
        if latest_summary:
            messages.insert(0, {"role": "system", "content": f"Long-term memory:\n{latest_summary}"})
        return messages

    def add_message(self, character_id: int, role: str, content: str) -> None:
        db.store_message(self.conn, character_id, role, content)

    def _user_message_count(self, character_id: int) -> int:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE character_id = ? AND role = 'user'",
            (character_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    async def maybe_summarize(self, character_id: int) -> Optional[str]:
        user_count = self._user_message_count(character_id)
        if user_count == 0 or user_count % self.summary_every != 0:
            return None
        # collect recent dialogue for summarization
        recent = db.get_recent_messages(self.conn, character_id, 40)
        text = "\n".join([f"{role}: {content}" for role, content in recent])
        summary = await self.llm_client.summarize(text)
        if summary:
            db.store_summary(self.conn, character_id, summary)
        return summary
