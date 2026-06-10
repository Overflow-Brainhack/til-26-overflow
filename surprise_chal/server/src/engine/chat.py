"""chat messages and log"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    turn: int
    sender_id: str
    text: str
    recipient_id: str | None = None  # None = global broadcast

    @property
    def is_global(self) -> bool:
        return self.recipient_id is None

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "text": self.text,
        }


@dataclass
class ChatLog:
    messages: list[ChatMessage] = field(default_factory=list)

    def post(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def post_system(self, turn: int, text: str) -> None:
        self.messages.append(ChatMessage(turn=turn, sender_id="__system__", text=text))

    def global_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if m.is_global]

    def private_messages_for(self, player_id: str) -> list[ChatMessage]:
        return [
            m
            for m in self.messages
            if not m.is_global
            and (m.sender_id == player_id or m.recipient_id == player_id)
        ]

    def to_list(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]
