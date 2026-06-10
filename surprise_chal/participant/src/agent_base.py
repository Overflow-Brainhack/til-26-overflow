"""The one interface your agent must implement.

The competition server sends your server a JSON **observation** every turn and
expects a JSON **action payload** back within the response deadline. `server.py`
handles all the HTTP; you just implement `decide()`.

- Input: the raw observation dict (see RULES.md → "Observation Payload").
- Output: an `ActionPayload` (build it from the typed actions in
  `engine.actions`; the server serialises it to JSON for you).

`engine/` is the *actual game engine* — the source of truth for every rule and
number. Read it freely; you do not need to run it to play.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.actions import ActionPayload


class PlayerAgent(ABC):
    @abstractmethod
    async def decide(self, observation: dict) -> ActionPayload:
        """Given this turn's observation dict, return your ActionPayload."""
        ...
