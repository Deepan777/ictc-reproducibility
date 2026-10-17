"""Visible-telemetry routing proposal and gated execution."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ictc.controller.admission import AdmissionGate, AdmissionOutcome, AdmissionRequest
from ictc.domain.enums import ActionKind, AdmissionDecision, ReasonCode
from ictc.domain.models import NetworkAction
from ictc.simulation.environment import ControllerTrace


@dataclass(frozen=True, slots=True)
class ThresholdRoutingPolicy:
    current_path_id: str
    candidate_path_id: str
    minimum_queue_improvement: int = 1

    def __post_init__(self) -> None:
        if not self.current_path_id or not self.candidate_path_id:
            raise ValueError("routing path IDs are required")
        if self.current_path_id == self.candidate_path_id:
            raise ValueError("current and candidate paths must differ")
        if self.minimum_queue_improvement < 0:
            raise ValueError("queue improvement threshold cannot be negative")

    def propose(self, trace: ControllerTrace, *, decision_tick: int) -> NetworkAction:
        """Propose independently from ICTC using only delivered visible telemetry."""
        if decision_tick < 0:
            raise ValueError("decision tick cannot be negative")
        latest: dict[str, tuple[int, int]] = {}
        for record in trace.available_at(decision_tick):
            if record.queue_occupancy is None or record.missing:
                continue
            latest[record.path_id] = (record.available_tick, record.queue_occupancy)
        current = latest.get(self.current_path_id)
        candidate = latest.get(self.candidate_path_id)
        reroute = (
            current is not None
            and candidate is not None
            and current[1] - candidate[1] >= self.minimum_queue_improvement
        )
        kind = ActionKind.REROUTE_TO_PATH if reroute else ActionKind.KEEP_CURRENT_PATH
        payload = f"{trace.run_id}|{decision_tick}|{kind.value}|{current}|{candidate}".encode()
        return NetworkAction(
            action_id=f"policy:{hashlib.sha256(payload).hexdigest()[:20]}",
            kind=kind,
            path_id=self.candidate_path_id if reroute else None,
        )


class RoutingActuator(Protocol):
    def execute(self, action: NetworkAction) -> None: ...


@dataclass(slots=True)
class RecordingRoutingActuator:
    executed_actions: list[NetworkAction] = field(default_factory=list)

    def execute(self, action: NetworkAction) -> None:
        self.executed_actions.append(action)


@dataclass(frozen=True, slots=True)
class ControllerExecution:
    proposal: NetworkAction
    admission: AdmissionOutcome
    executed_action: NetworkAction
    fallback_used: bool
    execution_reason_codes: tuple[ReasonCode, ...]


@dataclass(slots=True)
class GatedRoutingController:
    policy: ThresholdRoutingPolicy
    gate: AdmissionGate
    actuator: RoutingActuator

    def decide_and_execute(
        self,
        trace: ControllerTrace,
        *,
        decision_tick: int,
        build_admission_request: Callable[[NetworkAction], AdmissionRequest],
    ) -> ControllerExecution:
        proposal = self.policy.propose(trace, decision_tick=decision_tick)
        request = build_admission_request(proposal)
        if request.proposed_action != proposal:
            raise ValueError("admission request must bind the independently proposed action")
        outcome = self.gate.evaluate(request)
        if outcome.executable_action is None:
            raise RuntimeError("routing admission produced no executable action")
        executed = outcome.executable_action
        fallback_used = (
            executed == request.fallback_action and outcome.decision is not AdmissionDecision.ADMIT
        )
        reasons = (
            (*outcome.reason_codes, ReasonCode.FALLBACK_EXECUTED)
            if fallback_used
            else outcome.reason_codes
        )
        self.actuator.execute(executed)
        return ControllerExecution(
            proposal=proposal,
            admission=outcome,
            executed_action=executed,
            fallback_used=fallback_used,
            execution_reason_codes=reasons,
        )
