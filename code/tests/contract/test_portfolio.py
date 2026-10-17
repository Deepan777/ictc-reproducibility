"""Multi-arm retention, comparability, Pareto, and determinism checks for N04."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ictc.contract import (
    InterventionCandidate,
    InterventionPortfolio,
    InterventionPortfolioPlanner,
    PortfolioStatus,
)
from ictc.contract.portfolio_benchmark import (
    arm_candidate,
    build_n04_evidence,
    portfolio_candidates,
    portfolio_registry,
    render_n04_evidence,
)
from ictc.domain.enums import ActionKind
from ictc.domain.models import NetworkAction

ROOT = Path(__file__).resolve().parents[2]


def plan_portfolio() -> InterventionPortfolio:
    return InterventionPortfolioPlanner(portfolio_registry()).plan(
        portfolio_id="portfolio",
        candidates=portfolio_candidates(),
    )


def test_three_arms_retain_two_contracts_and_one_infeasibility() -> None:
    portfolio = plan_portfolio()

    assert portfolio.status is PortfolioStatus.PARTIAL
    assert tuple(item.candidate_id for item in portfolio.candidate_plans) == (
        "path-b",
        "path-c",
        "path-d",
    )
    assert portfolio.contract_candidate_ids == ("path-b", "path-c")
    assert portfolio.no_contract_candidate_ids == ("path-d",)
    plans = {item.candidate_id: item.plan for item in portfolio.candidate_plans}
    assert plans["path-b"].contract is not None
    assert plans["path-b"].contract.required_variable_ids == ("U",)
    assert plans["path-b"].selection is not None
    assert plans["path-b"].selection.total_cost_microunits == 5
    assert plans["path-c"].contract is not None
    assert plans["path-c"].contract.required_variable_ids == ("W",)
    assert plans["path-c"].selection is not None
    assert plans["path-c"].selection.total_cost_microunits == 9
    assert plans["path-d"].contract is None


def test_resource_pareto_is_not_an_effect_ranking() -> None:
    portfolio = plan_portfolio()
    assert portfolio.resource_pareto_candidate_ids == ("path-b",)
    payload = portfolio.to_payload()
    semantics = payload["pareto_semantics"]
    assert isinstance(semantics, str)
    assert "no causal benefit" in semantics


def test_portfolio_is_order_independent_and_digest_bound() -> None:
    first = plan_portfolio()
    second = InterventionPortfolioPlanner(tuple(reversed(portfolio_registry()))).plan(
        portfolio_id="portfolio",
        candidates=tuple(reversed(portfolio_candidates())),
    )
    assert first.to_payload() == second.to_payload()
    assert first.portfolio_digest == second.portfolio_digest


def test_complete_and_no_contract_statuses_are_typed() -> None:
    candidates = portfolio_candidates()
    complete = InterventionPortfolioPlanner(portfolio_registry()).plan(
        portfolio_id="complete", candidates=candidates[:2]
    )
    none = InterventionPortfolioPlanner(portfolio_registry()).plan(
        portfolio_id="none", candidates=(candidates[2],)
    )
    assert complete.status is PortfolioStatus.COMPLETE
    assert complete.no_contract_candidate_ids == ()
    assert none.status is PortfolioStatus.NO_CONTRACT
    assert none.contract_candidate_ids == ()
    assert none.resource_pareto_candidate_ids == ()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("candidate_id", "candidate IDs"),
        ("query_id", "query IDs"),
        ("action_id", "action IDs"),
        ("path_id", "path IDs"),
        ("proposal_id", "proposal IDs"),
        ("contract_id", "contract IDs"),
    ],
)
def test_duplicate_candidate_identity_fields_are_rejected(field: str, message: str) -> None:
    first, second, _ = portfolio_candidates()
    if field == "candidate_id":
        second = second.model_copy(update={"candidate_id": first.candidate_id})
    elif field == "query_id":
        second = second.model_copy(
            update={"query": second.query.model_copy(update={"query_id": first.query.query_id})}
        )
    elif field == "action_id":
        action = second.query.proposed_action.model_copy(
            update={"action_id": first.query.proposed_action.action_id}
        )
        second = second.model_copy(
            update={"query": second.query.model_copy(update={"proposed_action": action})}
        )
    elif field == "path_id":
        action = second.query.proposed_action.model_copy(
            update={"path_id": first.query.proposed_action.path_id}
        )
        second = second.model_copy(
            update={"query": second.query.model_copy(update={"proposed_action": action})}
        )
    elif field == "proposal_id":
        second = second.model_copy(update={"proposal_id": first.proposal_id})
    else:
        second = second.model_copy(update={"contract_id": first.contract_id})
    with pytest.raises(ValueError, match=message):
        InterventionPortfolioPlanner(portfolio_registry()).plan(
            portfolio_id="duplicates", candidates=(first, second)
        )


@pytest.mark.parametrize(
    "field", ["baseline_action", "treatment_variable_id", "outcome_variable_id", "horizon"]
)
def test_incomparable_arms_are_rejected(field: str) -> None:
    first, second, _ = portfolio_candidates()
    if field == "baseline_action":
        baseline = second.query.baseline_action.model_copy(
            update={
                "action": second.query.baseline_action.action.model_copy(
                    update={"action_id": "other"}
                )
            }
        )
        query = second.query.model_copy(update={field: baseline})
    elif field == "horizon":
        query = second.query.model_copy(
            update={field: second.query.horizon.model_copy(update={"duration_ms": 2000})}
        )
    else:
        query = second.query.model_copy(update={field: "other"})
    second = second.model_copy(update={"query": query})
    with pytest.raises(ValueError, match="must share"):
        InterventionPortfolioPlanner(portfolio_registry()).plan(
            portfolio_id="incomparable", candidates=(first, second)
        )


def test_empty_portfolio_id_candidates_and_non_reroute_are_rejected() -> None:
    planner = InterventionPortfolioPlanner(portfolio_registry())
    with pytest.raises(ValueError, match="portfolio ID"):
        planner.plan(portfolio_id="", candidates=portfolio_candidates())
    with pytest.raises(ValueError, match="at least one"):
        planner.plan(portfolio_id="empty", candidates=())

    candidate = arm_candidate("bad", "bad", "U")
    keep = NetworkAction(action_id="not-reroute", kind=ActionKind.KEEP_CURRENT_PATH)
    payload = candidate.model_dump()
    payload["query"]["proposed_action"] = keep.model_dump()
    with pytest.raises(ValidationError, match="identifiability query proposed action"):
        InterventionCandidate.model_validate(payload)


def test_portfolio_model_rejects_inconsistent_membership() -> None:
    payload = plan_portfolio().model_dump()
    payload["status"] = PortfolioStatus.COMPLETE
    with pytest.raises(ValidationError, match="status"):
        InterventionPortfolio.model_validate(payload)
    payload = plan_portfolio().model_dump()
    payload["resource_pareto_candidate_ids"] = ("path-d",)
    with pytest.raises(ValidationError, match="Pareto"):
        InterventionPortfolio.model_validate(payload)


def test_n04_evidence_is_deterministic_and_tracked() -> None:
    rendered = render_n04_evidence()
    assert json.loads(rendered) == build_n04_evidence()
    evidence = ROOT / "docs/milestones/evidence/n04_multi_arm_portfolio.json"
    assert evidence.read_text(encoding="utf-8") == rendered


def test_portfolio_planner_has_no_truth_or_decision_inputs() -> None:
    source = (ROOT / "src/ictc/contract/portfolio.py").read_text(encoding="utf-8")
    for token in (
        "ictc.oracle",
        "ictc.pilot",
        "m14_scoring",
        "post_action_outcome",
        "reference_admissible",
        "AdmissionGate",
    ):
        assert token not in source
