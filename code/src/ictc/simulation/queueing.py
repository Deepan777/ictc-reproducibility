"""Finite FIFO packet queues and exact accounting for M03."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

PacketKind = Literal["foreground", "cross"]


@dataclass(slots=True)
class _PacketCohort:
    kind: PacketKind
    arrival_tick: int
    count: int


@dataclass(frozen=True, slots=True)
class ServiceResult:
    """Packets served during one path tick."""

    foreground: int
    cross: int

    @property
    def total(self) -> int:
        return self.foreground + self.cross


@dataclass(frozen=True, slots=True)
class PacketAccounting:
    """Complete packet conservation record for one path."""

    path_id: str
    foreground_generated: int
    foreground_delivered: int
    foreground_dropped: int
    cross_generated: int
    cross_delivered: int
    cross_dropped: int
    final_queue_occupancy: int

    def validate(self) -> None:
        """Raise if any packet-conservation invariant is violated."""
        if self.foreground_generated != self.foreground_delivered + self.foreground_dropped:
            raise ValueError("foreground packet accounting invariant failed")
        if self.cross_generated != self.cross_delivered + self.cross_dropped:
            raise ValueError("cross-traffic packet accounting invariant failed")
        if self.final_queue_occupancy != 0:
            raise ValueError("queue must be empty after drain")


class PacketQueue:
    """Mutable internal FIFO queue with cross-first, tail-drop arrivals."""

    def __init__(self, path_id: str, service_capacity: int, queue_limit: int) -> None:
        self.path_id = path_id
        self.service_capacity = service_capacity
        self.queue_limit = queue_limit
        self._cohorts: deque[_PacketCohort] = deque()
        self._queued = 0
        self._generated = {"foreground": 0, "cross": 0}
        self._delivered = {"foreground": 0, "cross": 0}
        self._dropped = {"foreground": 0, "cross": 0}
        self._foreground_latency_ticks: list[int] = []

    @property
    def occupancy(self) -> int:
        return self._queued

    @property
    def empty(self) -> bool:
        return self._queued == 0

    @property
    def foreground_latency_ticks(self) -> tuple[int, ...]:
        return tuple(self._foreground_latency_ticks)

    def enqueue(self, kind: PacketKind, count: int, tick: int) -> int:
        """Tail-drop a packet cohort and return its dropped count."""
        if count < 0:
            raise ValueError("arrival count cannot be negative")
        self._generated[kind] += count
        admitted = min(count, self.queue_limit - self._queued)
        dropped = count - admitted
        self._dropped[kind] += dropped
        if admitted:
            self._cohorts.append(_PacketCohort(kind=kind, arrival_tick=tick, count=admitted))
            self._queued += admitted
        return dropped

    def service(self, tick: int, capacity: int | None = None) -> ServiceResult:
        """Serve at most one tick's capacity in FIFO order."""
        remaining = self.service_capacity if capacity is None else capacity
        foreground = 0
        cross = 0
        while remaining and self._cohorts:
            cohort = self._cohorts[0]
            served = min(remaining, cohort.count)
            cohort.count -= served
            remaining -= served
            self._queued -= served
            self._delivered[cohort.kind] += served
            if cohort.kind == "foreground":
                foreground += served
                self._foreground_latency_ticks.extend([tick - cohort.arrival_tick + 1] * served)
            else:
                cross += served
            if cohort.count == 0:
                self._cohorts.popleft()
        return ServiceResult(foreground=foreground, cross=cross)

    def accounting(self) -> PacketAccounting:
        """Return and validate the final conservation record."""
        accounting = PacketAccounting(
            path_id=self.path_id,
            foreground_generated=self._generated["foreground"],
            foreground_delivered=self._delivered["foreground"],
            foreground_dropped=self._dropped["foreground"],
            cross_generated=self._generated["cross"],
            cross_delivered=self._delivered["cross"],
            cross_dropped=self._dropped["cross"],
            final_queue_occupancy=self._queued,
        )
        accounting.validate()
        return accounting
