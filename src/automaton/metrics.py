from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np


def compute_acceptance_stats(
    automaton: Any,
    data: Sequence,
    labels: Sequence,
    accept_fn: Callable[[Any, Sequence], bool],
) -> dict:
    """
    Compute automaton agreement statistics with teacher model.

    labels:
        1 = automaton should accept
        0 = automaton should reject
    """
    if automaton is None or data is None or labels is None:
        return {
            "agreement": 0.0,
            "true_accept": 0.0,
            "true_reject": 0.0,
            "n_accepted": 0.0,
            "total": 0.0,
        }

    if len(data) == 0 or len(labels) == 0:
        return {
            "agreement": 0.0,
            "true_accept": 0.0,
            "true_reject": 0.0,
            "n_accepted": 0.0,
            "total": 0.0,
        }

    accepts = np.asarray(
        [accept_fn(automaton, seq) for seq in data],
        dtype=bool,
    )
    labels = np.asarray(labels, dtype=int)

    true_accept = float(np.sum((labels == 1) & accepts))
    true_reject = float(np.sum((labels == 0) & ~accepts))
    n_accepted = float(np.sum(accepts))
    total = float(len(labels))
    agreement = (true_accept + true_reject) / total if total > 0 else 0.0

    return {
        "agreement": float(agreement),
        "true_accept": true_accept,
        "true_reject": true_reject,
        "n_accepted": n_accepted,
        "total": total,
    }


def compute_automaton_agreement(
    automaton: Any,
    data: Sequence,
    labels: Sequence,
    accept_fn: Callable[[Any, Sequence], bool],
) -> float:
    """Return only agreement."""
    return compute_acceptance_stats(
        automaton=automaton,
        data=data,
        labels=labels,
        accept_fn=accept_fn,
    )["agreement"]