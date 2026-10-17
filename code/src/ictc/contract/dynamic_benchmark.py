"""Deterministic capability-loss, topology-change, and recovery evidence for N05."""

from __future__ import annotations

import json

from ictc.contract.dynamic import (
    DynamicContractLedger,
    DynamicContractResynthesizer,
    JsonValue,
    ResynthesisTrigger,
)
from ictc.contract.planning_benchmark import capability
from ictc.contract.portfolio import InterventionCandidate
from ictc.contract.portfolio_benchmark import (
    arm_candidate,
    portfolio_candidates,
    portfolio_registry,
)
from ictc.contract.synthesis import CapabilityAdvertisement


def capability_loss_registry() -> tuple[CapabilityAdvertisement, ...]:
    return tuple(item for item in portfolio_registry() if item.variable_id == "W")


def topology_changed_candidates() -> tuple[InterventionCandidate, ...]:
    path_b, _path_c, path_d = portfolio_candidates()
    return (path_b, arm_candidate("path-c", "path-c", "Z"), path_d)


def recovery_registry() -> tuple[CapabilityAdvertisement, ...]:
    return (
        *capability_loss_registry(),
        capability("z-a", "z-source-a", "domain-a", "Z", 6),
        capability("z-b", "z-source-b", "domain-b", "Z", 7),
    )


def build_n05_ledger() -> DynamicContractLedger:
    engine = DynamicContractResynthesizer()
    ledger = engine.start(
        ledger_id="n05-dynamic-ledger-v1",
        candidates=portfolio_candidates(),
        advertisements=portfolio_registry(),
    )
    ledger = engine.resynthesize(
        ledger,
        trigger=ResynthesisTrigger.CAPABILITY_CHANGE,
        candidates=portfolio_candidates(),
        advertisements=capability_loss_registry(),
    )
    ledger = engine.resynthesize(
        ledger,
        trigger=ResynthesisTrigger.TOPOLOGY_CHANGE,
        candidates=topology_changed_candidates(),
        advertisements=capability_loss_registry(),
    )
    return engine.resynthesize(
        ledger,
        trigger=ResynthesisTrigger.CAPABILITY_CHANGE,
        candidates=topology_changed_candidates(),
        advertisements=recovery_registry(),
    )


def build_n05_evidence() -> dict[str, JsonValue]:
    ledger = build_n05_ledger()
    return {
        "schema_version": "n05-dynamic-resynthesis-evidence-v1",
        "milestone": "N05",
        "ledger": ledger.to_payload(),
        "hand_checked_expectations": {
            "generation_triggers": [
                "INITIAL",
                "CAPABILITY_CHANGE",
                "TOPOLOGY_CHANGE",
                "CAPABILITY_CHANGE",
            ],
            "execution_ready_by_generation": [
                ["path-b", "path-c"],
                ["path-c"],
                [],
                ["path-c", "path-d"],
            ],
            "dispositions": [
                "PARTIAL_READY",
                "PARTIAL_READY",
                "FAIL_CLOSED",
                "PARTIAL_READY",
            ],
            "generation_count": 4,
            "contract_ids_are_generation_specific": True,
        },
        "change_boundary": (
            "Triggers are declared inputs. N05 does not infer topology, capacity, compromise, or "
            "policy changes from raw network observations."
        ),
        "forbidden_inputs": {
            "oracle_labels": False,
            "post_action_outcomes": False,
            "controller_decisions": False,
            "m14_reference_flags": False,
        },
        "random_sampling_used": False,
    }


def render_n05_evidence() -> str:
    return json.dumps(build_n05_evidence(), indent=2, sort_keys=True) + "\n"
