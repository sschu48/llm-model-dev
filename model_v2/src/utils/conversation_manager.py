from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history.pop(0)
        logger.debug(f"Added message to history. Current history length: {len(self.history)}")

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

    def get_history_as_string(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])

    def clear_history(self):
        self.history.clear()
        logger.debug("Conversation history cleared")

    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        return self.history[-n:]