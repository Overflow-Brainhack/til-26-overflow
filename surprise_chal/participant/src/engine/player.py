"""player state"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.resources import ResourceBag


@dataclass
class Player:
    id: str
    name: str
    resources: ResourceBag = field(default_factory=ResourceBag)
    alive: bool = True
    # turns since last base was lost; None means player still has bases
    decay_turns: int | None = None
    # ids of players this player has "met" (seen or been seen by)
    known_player_ids: set[str] = field(default_factory=set)
    # reserved for future tech tree
    researched: set[str] = field(default_factory=set)

    def mark_eliminated(self) -> None:
        if self.alive:
            self.alive = False
            self.decay_turns = 0

    def is_eliminated(self) -> bool:
        return not self.alive

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "resources": self.resources.to_dict(),
            "alive": self.alive,
            "decay_turns": self.decay_turns,
        }
