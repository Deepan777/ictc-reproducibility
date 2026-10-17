"""Explicit temporal boundary for completed historical estimation episodes."""

import math
from dataclasses import dataclass

from ictc.causal.estimation import AdjustmentObservation


@dataclass(frozen=True, slots=True)
class CompletedHistoricalEpisode:
    """All times use one declared clock and the same unit."""

    observation: AdjustmentObservation
    covariates_at: float
    action_at: float
    outcome_completed_at: float
    received_at: float

    def __post_init__(self) -> None:
        times = (self.covariates_at, self.action_at, self.outcome_completed_at, self.received_at)
        if not all(math.isfinite(value) for value in times):
            raise ValueError("episode timestamps must be finite")
        if not self.covariates_at < self.action_at <= self.outcome_completed_at <= self.received_at:
            raise ValueError(
                "historical covariates must precede action and receipt must follow outcome"
            )


def history_available_before(
    episodes: tuple[CompletedHistoricalEpisode, ...], decision_cutoff: float
) -> tuple[AdjustmentObservation, ...]:
    """Reject incomplete or not-yet-received outcomes at the strict decision cutoff."""
    if not math.isfinite(decision_cutoff):
        raise ValueError("decision cutoff must be finite")
    if any(episode.received_at >= decision_cutoff for episode in episodes):
        raise PermissionError(
            "every historical outcome must be complete and received before decision"
        )
    return tuple(episode.observation for episode in episodes)
