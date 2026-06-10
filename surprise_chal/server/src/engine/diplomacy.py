"""treaty system and diplomacy manager — extensible to multiple treaty types"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TreatyType(Enum):
    PEACE = auto()
    # future types: OPEN_BORDERS = auto(), VISION_SHARING = auto(), etc.


class TreatyStatus(Enum):
    PROPOSED = auto()  # one party has proposed, awaiting response
    ACTIVE = auto()  # both parties have accepted
    BREAKING = auto()  # one party triggered a break; countdown active


@dataclass
class Treaty:
    treaty_type: TreatyType
    proposer_id: str
    partner_id: str
    status: TreatyStatus = TreatyStatus.PROPOSED
    break_in_turns: int | None = None  # set when breaking begins

    @property
    def parties(self) -> frozenset[str]:
        return frozenset({self.proposer_id, self.partner_id})

    def is_active(self) -> bool:
        return self.status == TreatyStatus.ACTIVE

    def to_dict(self) -> dict:
        return {
            "treaty_type": self.treaty_type.name.lower(),
            "proposer_id": self.proposer_id,
            "partner_id": self.partner_id,
            "status": self.status.name.lower(),
            "break_in_turns": self.break_in_turns,
        }


class DiplomacyManager:
    """manages all treaties between players.

    keyed by (frozenset of player ids, treaty type) to allow multiple
    concurrent treaty types between the same pair.
    """

    def __init__(self) -> None:
        # (parties, treaty_type) → Treaty
        self._treaties: dict[tuple[frozenset[str], TreatyType], Treaty] = {}

    def _key(self, a: str, b: str, tt: TreatyType) -> tuple[frozenset[str], TreatyType]:
        return frozenset({a, b}), tt

    def propose(self, proposer: str, partner: str, tt: TreatyType) -> bool:
        """returns True if the proposal was accepted (no duplicate)"""
        key = self._key(proposer, partner, tt)
        if key in self._treaties:
            return False
        self._treaties[key] = Treaty(tt, proposer, partner, TreatyStatus.PROPOSED)
        return True

    def accept(self, acceptor: str, proposer: str, tt: TreatyType) -> bool:
        """acceptor accepts an incoming proposal; returns True if successful"""
        key = self._key(acceptor, proposer, tt)
        treaty = self._treaties.get(key)
        if treaty is None or treaty.status != TreatyStatus.PROPOSED:
            return False
        if treaty.proposer_id == acceptor:
            return False  # can't accept your own proposal
        treaty.status = TreatyStatus.ACTIVE
        return True

    def reject(self, rejector: str, proposer: str, tt: TreatyType) -> bool:
        key = self._key(rejector, proposer, tt)
        treaty = self._treaties.get(key)
        if treaty is None or treaty.status != TreatyStatus.PROPOSED:
            return False
        del self._treaties[key]
        return True

    def break_treaty(
        self, initiator: str, partner: str, tt: TreatyType, delay: int
    ) -> bool:
        """start breaking a treaty; takes effect after `delay` turns"""
        key = self._key(initiator, partner, tt)
        treaty = self._treaties.get(key)
        if treaty is None or treaty.status != TreatyStatus.ACTIVE:
            return False
        treaty.status = TreatyStatus.BREAKING
        treaty.break_in_turns = delay
        return True

    def tick(self) -> list[Treaty]:
        """advance all breaking treaties by 1 turn; return treaties that expired"""
        expired: list[Treaty] = []
        to_delete: list[tuple[frozenset[str], TreatyType]] = []
        for key, treaty in self._treaties.items():
            if (
                treaty.status == TreatyStatus.BREAKING
                and treaty.break_in_turns is not None
            ):
                treaty.break_in_turns -= 1
                if treaty.break_in_turns <= 0:
                    expired.append(treaty)
                    to_delete.append(key)
        for key in to_delete:
            del self._treaties[key]
        return expired

    def cancel_proposed(self, proposer: str, partner: str, tt: TreatyType) -> None:
        """remove a pending proposal (e.g. proposer changed their mind)"""
        key = self._key(proposer, partner, tt)
        self._treaties.pop(key, None)

    def void_all(self) -> list[Treaty]:
        """Immediately clear EVERY treaty/proposal of any status (the post-cutoff
        forced-war rule). Returns the active/breaking treaties that were torn down
        so the runner can notify their parties; pending proposals are simply dropped."""
        active = [
            t
            for t in self._treaties.values()
            if t.status in (TreatyStatus.ACTIVE, TreatyStatus.BREAKING)
        ]
        self._treaties.clear()
        return active

    def has_active_treaty(self, a: str, b: str, tt: TreatyType) -> bool:
        key = self._key(a, b, tt)
        t = self._treaties.get(key)
        return t is not None and t.is_active()

    def is_peace(self, a: str, b: str) -> bool:
        """convenience: are a and b currently bound by an active peace treaty?"""
        return self.has_active_treaty(a, b, TreatyType.PEACE)

    def incoming_proposals_for(self, player_id: str) -> list[Treaty]:
        return [
            t
            for t in self._treaties.values()
            if t.status == TreatyStatus.PROPOSED and t.partner_id == player_id
        ]

    def active_treaties_for(self, player_id: str) -> list[Treaty]:
        return [
            t
            for t in self._treaties.values()
            if player_id in t.parties
            and t.status in (TreatyStatus.ACTIVE, TreatyStatus.BREAKING)
        ]

    def to_list(self) -> list[dict]:
        return [t.to_dict() for t in self._treaties.values()]

    def active_treaty_pairs(self) -> list[dict]:
        """The treaties currently IN EFFECT (active or counting down to break) as
        unordered party pairs — for the replay's treaty graph. Pending (unaccepted)
        proposals are excluded: a graph edge means 'these two are at peace right now'."""
        return [
            {
                "a": t.proposer_id,
                "b": t.partner_id,
                "treaty_type": t.treaty_type.name.lower(),
                "breaking_in_turns": t.break_in_turns,
            }
            for t in self._treaties.values()
            if t.status in (TreatyStatus.ACTIVE, TreatyStatus.BREAKING)
        ]
