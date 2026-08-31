"""
Automata beam search for local DFA explanations.

This module keeps the Anchor-style beam search / KL-LUCB backbone, but moves
DFA-specific agreement evaluation and automaton manipulation to project-owned
modules under src/automaton and src/learner.
"""
from __future__ import annotations

import copy
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from automaton.metrics import compute_acceptance_stats, compute_automaton_agreement
from automaton.dfa_utils import (
    check_dfa_path_accepted,
    dfa_to_graphviz,
    is_valid_dfa,
    plot_beam_stats,
    remove_unreachable_states,
)
from learner.factory import get_learner

# ------------------------------------------------------------------
# KL-LUCB utilities
# ------------------------------------------------------------------
def kl_bernoulli(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL divergence between Bernoulli(p) and Bernoulli(q)."""
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def dup_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """Upper confidence bound for Bernoulli means using KL divergence."""
    lm = p.copy()
    um = np.minimum(p + np.sqrt(level / 2.0), 1.0)

    for _ in range(1, n_iter):
        qm = (um + lm) / 2.0
        kl_gt_idx = kl_bernoulli(p, qm) > level
        kl_lt_idx = np.logical_not(kl_gt_idx)
        um[kl_gt_idx] = qm[kl_gt_idx]
        lm[kl_lt_idx] = qm[kl_lt_idx]

    return um


def dlow_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """Lower confidence bound for Bernoulli means using KL divergence."""
    um = p.copy()
    lm = np.clip(p - np.sqrt(level / 2.0), 0.0, 1.0)

    for _ in range(1, n_iter):
        qm = (um + lm) / 2.0
        kl_gt_idx = kl_bernoulli(p, qm) > level
        kl_lt_idx = np.logical_not(kl_gt_idx)
        lm[kl_gt_idx] = qm[kl_gt_idx]
        um[kl_lt_idx] = qm[kl_lt_idx]

    return lm


def compute_beta(n_features: int, t: int, delta: float) -> float:
    """Confidence-budget term used by KL-LUCB."""
    alpha = 1.1
    k = 405.5
    temp = np.log(k * n_features * (t ** alpha) / delta)
    return temp + np.log(temp)


def select_critical_arms(
    means: np.ndarray,
    ub: np.ndarray,
    lb: np.ndarray,
    n_samples: np.ndarray,
    delta: float,
    top_n: int,
    t: int,
):
    """Select the two arms that KL-LUCB should sample next."""
    crit_arms = namedtuple("crit_arms", ["ut", "lt"])

    sorted_means = np.argsort(means)
    beta = compute_beta(len(means), t, delta)

    top_set = sorted_means[-top_n:]
    rest_set = sorted_means[:-top_n]

    if len(rest_set) == 0:
        lt = top_set[np.argmin(lb[top_set])] if len(top_set) > 0 else 0
        return crit_arms._make((lt, lt))

    ub[rest_set] = dup_bernoulli(means[rest_set], beta / n_samples[rest_set])
    lb[top_set] = dlow_bernoulli(means[top_set], beta / n_samples[top_set])

    ut = rest_set[np.argmax(ub[rest_set])]
    lt = top_set[np.argmin(lb[top_set])]
    return crit_arms._make((ut, lt))


def to_sample(
    means: np.ndarray,
    ubs: np.ndarray,
    lbs: np.ndarray,
    agreement_threshold: float,
    epsilon_stop: float,
) -> np.ndarray:
    """Return which selected candidates need additional samples."""
    return np.logical_and(
        np.logical_or(ubs >= agreement_threshold, means >= agreement_threshold),
        lbs < agreement_threshold - epsilon_stop,
    )


def _score_candidate_acceptance(args: tuple[Any, Sequence, Sequence]) -> dict:
    """Process-safe helper for candidate scoring."""
    automaton, raw_data, labels = args
    return compute_acceptance_stats(
        automaton=automaton,
        data=raw_data,
        labels=labels,
        accept_fn=check_dfa_path_accepted,
    )



class AutomataBeamSearch:
    """
    Beam search over automata candidates.
    """

    # Local KL-LUCB utilities. Keeping them as static methods avoids
    # depending on modified_modules.alibi and prevents circular imports.
    dup_bernoulli = staticmethod(dup_bernoulli)
    dlow_bernoulli = staticmethod(dlow_bernoulli)
    compute_beta = staticmethod(compute_beta)
    select_critical_arms = staticmethod(select_critical_arms)
    to_sample = staticmethod(to_sample)

    def __init__(self, samplers: List[Callable] | Callable, predictor=None, **kwargs) -> None:
        self.samplers = samplers if isinstance(samplers, list) else [samplers]
        self.sample_fcn = self.samplers[0]
        self.predictor = predictor
        self.sample_cache_size = kwargs.get("sample_cache_size", 1000)
        # Only KL-LUCB sampling/agreement-stat computation is parallelized.
        # DFA candidate generation remains sequential.
        # Controlled by the unified public parameters: parallel and n_jobs.
        self.parallel = bool(kwargs.get("parallel", False))
        self.n_jobs = max(1, int(kwargs.get("n_jobs", 4)))
        # Optional lightweight profiling: accumulates wall-clock time per
        # search phase (KL-LUCB sampling/evaluation vs candidate proposal),
        # printed as a single summary line at the end of automata_beam() when
        # profile_time=True. Always accumulated (negligible timer overhead);
        # only the printing is gated on the flag.
        self.profile_time = bool(kwargs.get("profile_time", False))
        self._profile_totals: Dict[str, float] = defaultdict(float)
        # Reused across draw_automata_samples_parallel calls (KL-LUCB may
        # call this many times per beam iteration, once per critical-arm
        # sampling round) to avoid paying process-pool startup cost every
        # time. Lazily created on first use; see _get_process_pool().
        self._process_pool = None

        self.learner = None
        self.automatas: List[Any] = []
        self.validation_data: List[Any] = []
        self.validation_labels = np.array([], dtype=np.float64)
        self.iteration = 0
        self.state: dict = {}
        # Monotonic counter, incremented once per parallel sampling task
        # submitted (in the fixed, sequential order tasks are decided and
        # queued -- never touched by worker threads). Used together with the
        # sampler's own seed to give each submitted task its own independent,
        # reproducible random.Random stream, so parallel=True sampling stays
        # bit-for-bit reproducible even though thread scheduling itself is
        # not deterministic. See draw_automata_samples_parallel().
        self._sample_call_counter = 0

        # Every automaton candidate ever proposed, kept alive for the whole
        # search. self.state's t_nsamples/t_positives/... dicts are keyed by
        # id(automaton) (a memory address) and are never cleared, so if a
        # rejected candidate were garbage-collected mid-search, Python could
        # reuse its address for a brand-new, unrelated candidate -- which
        # would then silently "inherit" the old candidate's stale
        # accumulated stats on its first lookup. Retaining a strong
        # reference to every candidate here for the search's lifetime makes
        # id() reuse impossible, so this can't happen. This is a real,
        # pre-existing source of non-determinism (memory allocation timing
        # is not guaranteed identical across runs), not something limited to
        # parallel=True.
        self._id_reuse_guard: List[Any] = []

    def _get_process_pool(self):
        """Return a lazily-created, reusable ProcessPoolExecutor sized to n_jobs."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=max(1, int(self.n_jobs)))
        return self._process_pool

    def __del__(self):
        pool = getattr(self, "_process_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # State and sampling
    # ------------------------------------------------------------------
    def _init_state(self, batch_size: int) -> None:
        """
        State needed by AutomataBeamSearch and DFALearner.propose_automata().

        Removed unused Anchor state:
            t_coverage, t_coverage_idx, t_covered_true, t_covered_false, margin
        Kept required state:
            data, labels, current_idx, t_order, and candidate sample counters.
        """
        prealloc_size = batch_size * self.sample_cache_size
        self.state = {
            "t_nsamples": defaultdict(lambda: 0.0),
            "t_accepted": defaultdict(lambda: 0.0),
            "t_positives": defaultdict(lambda: 0.0),
            "t_negatives": defaultdict(lambda: 0.0),
            "t_order": defaultdict(list),
            "data": [],
            "labels": np.zeros(prealloc_size, dtype=np.float64),
            "batch_size": batch_size,
            "current_idx": 0,
        }
        self.state["t_order"][()] = []

    def _select_sampler_for_learner(self) -> Callable:
        if self.learner is None:
            raise RuntimeError("Automata learner has not been initialized.")

        if hasattr(self.learner, "get_sampler"):
            sampler_cls = self.learner.get_sampler()
            for sampler in self.samplers:
                if isinstance(sampler, sampler_cls):
                    return sampler
        return self.samplers[0]

    def draw_automata_samples(self, automata_list: list, batch_size: int) -> Tuple[tuple, tuple, tuple, tuple]:
        """Draw samples once per automaton and update search statistics."""
        sample_stats = []
        for automaton in automata_list:
            t0 = time.perf_counter()
            raw_data, labels = self.sample_fcn(num_samples=batch_size)
            self._profile_totals["sample_predict"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            sample_stats.append(self.update_automata_state(labels, raw_data, automaton))
            self._profile_totals["accept_check"] += time.perf_counter() - t0

        if not sample_stats:
            return (), (), (), ()

        true_accept, true_reject, total, accepted = tuple(zip(*sample_stats))
        return true_accept, true_reject, total, accepted

    def draw_automata_samples_parallel(self, automata_list: list, batch_size: int, n_jobs: int | None = None) -> Tuple[tuple, tuple, tuple, tuple]:
        """
        Draw one sample batch per candidate automaton in parallel, then score
        the sampled batches in parallel as well.

        This parallelizes the expensive part that used to be sequential:
            candidate_1 -> sample batch + teacher labels
            candidate_2 -> sample batch + teacher labels
            ...

        Sampling stays in threads so the sampler can reuse its own cache and
        model objects. DFA agreement scoring is pushed into processes so the
        Python-level path traversal in check_dfa_path_accepted can run without
        the GIL.

        Reproducibility: perturbation sampling normally draws from the shared
        global `random` module, which is safe for a single sequential caller
        but not for several threads consuming it concurrently -- which
        thread's draw lands where in the shared stream depends on OS thread
        scheduling, so results are not reproducible across runs even with a
        fixed seed. When the sampler was constructed with an explicit seed,
        each task submitted here instead gets its own independent
        random.Random, keyed by (sampler seed, a monotonically increasing
        call counter). That counter is advanced here in the same fixed,
        purely sequential order every run (deciding what to submit is never
        threaded), so task #k always gets the same independent stream
        regardless of which worker thread actually executes it or in what
        wall-clock order threads finish -- giving bit-for-bit reproducible
        sampling under parallel=True. Callers that never set a sampler seed
        keep the old (not reproducibility-guaranteed) behavior unchanged.
        """
        if not automata_list:
            return (), (), (), ()
        if self.learner is None:
            raise RuntimeError("Automata learner has not been initialized.")

        n_jobs = max(1, int(n_jobs or self.n_jobs))

        if n_jobs <= 1 or len(automata_list) <= 1:
            return self.draw_automata_samples(automata_list, batch_size)

        max_workers = min(n_jobs, len(automata_list))
        sampled_batches = []
        base_seed = getattr(self.sample_fcn, "seed", None)

        sample_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _ in automata_list:
                if base_seed is not None:
                    call_rng = random.Random((base_seed, self._sample_call_counter))
                    self._sample_call_counter += 1
                    futures.append(executor.submit(self.sample_fcn, num_samples=batch_size, rng=call_rng))
                else:
                    futures.append(executor.submit(self.sample_fcn, num_samples=batch_size))
            for fut in futures:
                raw_data, labels = fut.result()
                sampled_batches.append((raw_data, labels))
        self._profile_totals["sample_predict"] += time.perf_counter() - sample_start

        worker_results = []
        score_payloads = [
            (automaton, raw_data, labels)
            for automaton, (raw_data, labels) in zip(automata_list, sampled_batches)
        ]

        accept_start = time.perf_counter()
        try:
            pool = self._get_process_pool()
            futures = [pool.submit(_score_candidate_acceptance, payload) for payload in score_payloads]
            for automaton, (raw_data, labels), fut in zip(automata_list, sampled_batches, futures):
                worker_results.append((automaton, raw_data, labels, fut.result()))
        except Exception:
            # Fallback keeps the method usable when a candidate or automaton
            # cannot be pickled cleanly for process-based scoring.
            self._process_pool = None
            for automaton, (raw_data, labels) in zip(automata_list, sampled_batches):
                stats = compute_acceptance_stats(
                    automaton=automaton,
                    data=raw_data,
                    labels=labels,
                    accept_fn=check_dfa_path_accepted,
                )
                worker_results.append((automaton, raw_data, labels, stats))
        self._profile_totals["accept_check"] += time.perf_counter() - accept_start

        sample_stats = []
        for automaton, raw_data, labels, stats in worker_results:
            n_samples = int(stats["total"])
            current_idx = self.state["current_idx"]
            automaton_id = id(automaton)

            self.state["t_nsamples"][automaton_id] += stats["total"]
            self.state["t_accepted"][automaton_id] += stats["n_accepted"]
            self.state["t_positives"][automaton_id] += stats["true_accept"]
            self.state["t_negatives"][automaton_id] += stats["true_reject"]
            self.state["data"].extend(raw_data)

            if current_idx + n_samples > len(self.state["labels"]):
                growth_size = max(n_samples, self.state.get("batch_size", 1) * 10)
                new_size = len(self.state["labels"]) + growth_size
                self.state["labels"] = np.resize(self.state["labels"], new_size)

            self.state["labels"][current_idx: current_idx + n_samples] = labels
            self.state["current_idx"] += n_samples

            sample_stats.append((
                int(stats["true_accept"]),
                int(stats["true_reject"]),
                n_samples,
                int(stats["n_accepted"]),
            ))

        true_accept, true_reject, total, accepted = tuple(zip(*sample_stats))
        return true_accept, true_reject, total, accepted

    def _draw_kllucb_samples(self, automata_list: list, batch_size: int) -> Tuple[tuple, tuple, tuple, tuple]:
        """Route KL-LUCB sampling through the sequential or parallel implementation."""
        if self.parallel:
            return self.draw_automata_samples_parallel(
                automata_list,
                batch_size,
                n_jobs=self.n_jobs,
            )
        return self.draw_automata_samples(automata_list, batch_size)

    def update_automata_state(self, labels: Sequence, samples: Sequence, automaton: Any) -> Tuple[int, int, int, int]:
        """Update cached agreement statistics for one automaton."""
        if self.learner is None:
            raise RuntimeError("Automata learner has not been initialized.")

        n_samples = len(samples)
        current_idx = self.state["current_idx"]

        stats = compute_acceptance_stats(
            automaton=automaton,
            data=samples,
            labels=labels,
            accept_fn=self.learner.check_path_accepted,
        )

        true_accept = int(stats["true_accept"])
        true_reject = int(stats["true_reject"])
        n_accepted = int(stats["n_accepted"])

        automaton_id = id(automaton)
        self.state["t_nsamples"][automaton_id] += n_samples
        self.state["t_accepted"][automaton_id] += n_accepted
        self.state["t_positives"][automaton_id] += true_accept
        self.state["t_negatives"][automaton_id] += true_reject
        self.state["data"].extend(samples)

        if current_idx + n_samples > len(self.state["labels"]):
            growth_size = max(n_samples, self.state.get("batch_size", 1) * 10)
            new_size = len(self.state["labels"]) + growth_size
            self.state["labels"] = np.resize(self.state["labels"], new_size)

        self.state["labels"][current_idx: current_idx + n_samples] = labels
        self.state["current_idx"] += n_samples

        return true_accept, true_reject, n_samples, n_accepted

    def get_init_stats(self, automata_list: list) -> Dict[str, np.ndarray]:
        """Read cached sample statistics for a list of candidate automata."""
        size = len(automata_list)
        stats = {
            "n_samples": np.zeros(size, dtype=float),
            "n_accepted": np.zeros(size, dtype=float),
            "positives": np.zeros(size, dtype=float),
            "negatives": np.zeros(size, dtype=float),
        }
        for i, automaton in enumerate(automata_list):
            automaton_id = id(automaton)
            stats["n_samples"][i] = self.state["t_nsamples"][automaton_id]
            stats["n_accepted"][i] = self.state["t_accepted"][automaton_id]
            stats["positives"][i] = self.state["t_positives"][automaton_id]
            stats["negatives"][i] = self.state["t_negatives"][automaton_id]
        return stats

    @staticmethod
    def _agreements_from_stats(stats: Dict[str, np.ndarray]) -> np.ndarray:
        positives = np.asarray(stats["positives"], dtype=float)
        negatives = np.asarray(stats["negatives"], dtype=float)
        n_samples = np.asarray(stats["n_samples"], dtype=float)
        correct = positives + negatives
        return np.divide(correct, n_samples, out=np.zeros_like(n_samples), where=n_samples != 0)

    @staticmethod
    def _iteration_op_label(iteration: int) -> str:
        """Return the high-level refinement operation used in one beam iteration."""
        if int(iteration) == 0:
            return "INIT"
        return "DELETE+MERGE" if int(iteration) % 2 == 0 else "DELTA"

    # ------------------------------------------------------------------
    # KL-LUCB over automata candidates
    # ------------------------------------------------------------------
    def kllucb_automata(
        self,
        automata_list: list,
        init_stats: dict,
        epsilon: float,
        delta: float,
        batch_size: int,
        top_n: int,
        verbose: bool = False,
        verbose_every: int = 1,
        max_rounds: int = 3000,
    ) -> np.ndarray:
        """Run KL-LUCB to identify the top automata by agreement."""
        n_features = len(automata_list)
        if n_features == 0:
            return np.array([], dtype=int)

        n_samples = init_stats["n_samples"]
        n_accepted = init_stats["n_accepted"]
        positives = init_stats["positives"]
        negatives = init_stats["negatives"]

        unsampled_idx = np.where(n_samples == 0)[0]
        if len(unsampled_idx) > 0:
            unsampled_automata = [automata_list[int(i)] for i in unsampled_idx]
            true_accept, true_reject, total, accepted = self._draw_kllucb_samples(unsampled_automata, 1)
            positives[unsampled_idx] += np.asarray(true_accept, dtype=float)
            negatives[unsampled_idx] += np.asarray(true_reject, dtype=float)
            n_samples[unsampled_idx] += np.asarray(total, dtype=float)
            n_accepted[unsampled_idx] += np.asarray(accepted, dtype=float)

        if n_features <= top_n:
            return np.arange(n_features)

        means = self._agreements_from_stats(init_stats)
        ub = np.zeros(n_samples.shape)
        lb = np.zeros(n_samples.shape)

        t = 1
        crit_a_idx = self.select_critical_arms(means, ub, lb, n_samples, delta, top_n, t)
        bound_gap = ub[crit_a_idx.ut] - lb[crit_a_idx.lt]
        prev_gap = bound_gap
        no_improvement_count = 0

        while bound_gap > epsilon and t < max_rounds:
            selected_automata = [automata_list[idx] for idx in crit_a_idx]
            true_accept, true_reject, total, accepted = self._draw_kllucb_samples(selected_automata, batch_size)
            idx = list(crit_a_idx)

            positives[idx] += np.asarray(true_accept, dtype=float)
            negatives[idx] += np.asarray(true_reject, dtype=float)
            n_samples[idx] += np.asarray(total, dtype=float)
            n_accepted[idx] += np.asarray(accepted, dtype=float)

            means = self._agreements_from_stats(init_stats)
            crit_a_idx = self.select_critical_arms(means, ub, lb, n_samples, delta, top_n, t)
            new_gap = ub[crit_a_idx.ut] - lb[crit_a_idx.lt]

            relative_improvement = (prev_gap - new_gap) / prev_gap if prev_gap > 0 else 0.0
            if relative_improvement < 0.01:
                no_improvement_count += 1
                if no_improvement_count >= 10 and new_gap <= 2 * epsilon:
                    if verbose:
                        print(
                            f"  [KL-LUCB] Early stop at round {t}: "
                            f"B={new_gap:.6f}, eps={epsilon:.6f}"
                        )
                    break
            else:
                no_improvement_count = 0

            prev_gap = new_gap
            bound_gap = new_gap
            t += 1

        return np.argsort(means)[-top_n:]

    # ------------------------------------------------------------------
    # Metadata and result formatting
    # ------------------------------------------------------------------
    def get_automata_metadata(
        self,
        automaton: Any,
        success: bool,
        batch_size: int = 100,
        is_final: bool = False,
        collect_error_examples: bool = False,
    ) -> dict:
        """Return metadata for a candidate automaton."""
        automaton_id = id(automaton)

        if self.state["t_nsamples"][automaton_id] == 0:
            self.draw_automata_samples([automaton], batch_size)

        positives = self.state["t_positives"][automaton_id]
        negatives = self.state["t_negatives"][automaton_id]
        total = self.state["t_nsamples"][automaton_id]
        training_agreement = (positives + negatives) / total if total > 0 else 0.0

        validation_agreement = 0.0
        false_accept = tuple()
        false_reject = tuple()

        if is_final and len(self.validation_data) > 0 and len(self.validation_labels) > 0:
            validation_agreement = compute_automaton_agreement(
                automaton=automaton,
                data=self.validation_data,
                labels=self.validation_labels,
                accept_fn=self.learner.check_path_accepted,
            )

            if collect_error_examples:
                accepts = np.asarray(
                    [self.learner.check_path_accepted(automaton, seq) for seq in self.validation_data],
                    dtype=bool,
                )
                labels = np.asarray(self.validation_labels, dtype=int)
                false_accept = tuple(
                    seq for seq, label, accepted in zip(self.validation_data, labels, accepts)
                    if label == 0 and accepted
                )
                false_reject = tuple(
                    seq for seq, label, accepted in zip(self.validation_data, labels, accepts)
                    if label == 1 and not accepted
                )

        return {
            "automata": automaton,
            "training_agreement": float(training_agreement),
            "validation_agreement": float(validation_agreement),
            "states": len(automaton.states),
            "num_preds": int(self.state["current_idx"]),
            "success": success,
            "false_accept": false_accept,
            "false_reject": false_reject,
            # Backward-compatible typo/old key used by prior code.
            "true_reject": false_reject,
        }

    def _make_result(
        self,
        origin_automaton: Any,
        final_automaton: Any,
        success: bool,
        reason: str,
        budget_used: int,
        init_automaton_time: float,
        batch_size: int,
        output_dir: str,
        save_graphs: bool,
        collect_error_examples: bool,
        final_training_agreement: Optional[float] = None,
    ) -> dict:
        initial_metadata = self.get_automata_metadata(
            origin_automaton,
            success=True,
            batch_size=batch_size,
            is_final=True,
            collect_error_examples=False,
        )

        remove_unreachable_states(final_automaton)
        if save_graphs:
            try:
                dfa_to_graphviz(
                    final_automaton,
                    filename="final_automata",
                    instance=getattr(self.sample_fcn, "instance", None),
                    output_dir=output_dir,
                )
            except Exception as exc:
                print(f"  [WARNING] Could not save final DFA graph: {exc}")

        final_metadata = self.get_automata_metadata(
            final_automaton,
            success=success,
            batch_size=batch_size,
            is_final=True,
            collect_error_examples=collect_error_examples,
        )
        if final_training_agreement is not None:
            # Report the training agreement the candidate was actually selected
            # with (its all_history snapshot), instead of get_automata_metadata's
            # live re-read of self.state, which may have drifted since selection
            # if this automaton was resampled again in later beam iterations
            # (e.g. reused as a DELTA parent). Validation agreement, states, and
            # false_accept/false_reject are unaffected and still come from the
            # live metadata above.
            final_metadata["training_agreement"] = float(final_training_agreement)

        training_agreements = [initial_metadata["training_agreement"], final_metadata["training_agreement"]]
        validation_agreements = [initial_metadata["validation_agreement"], final_metadata["validation_agreement"]]
        states = [initial_metadata["states"], final_metadata["states"]]

        return {
            "automata": [origin_automaton, final_automaton],
            "states": states,
            "initial_state": states[0],
            "final_state": states[-1],
            "initial_training_agreement": training_agreements[0],
            "final_training_agreement": training_agreements[-1],
            "initial_validation_agreement": validation_agreements[0],
            "final_validation_agreement": validation_agreements[-1],
            "success": success,
            "reason": reason,
            "budget_used": budget_used,
            "validation_data": self.validation_data,
            "validation_labels": self.validation_labels,
            "num_preds": final_metadata["num_preds"],
            "false_accept": final_metadata["false_accept"],
            "false_reject": final_metadata["false_reject"],
            "true_reject": final_metadata["true_reject"],
            "init_automaton_time": init_automaton_time,
        }

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    def automata_beam(
        self,
        data_type: str = "Tabular",
        alphabet: Optional[List] = None,
        automaton_type: str = "DFA",
        delta: float = 0.05,
        epsilon: float = 0.1,
        agreement_threshold: Optional[float] = None,
        beam_size: int = 1,
        epsilon_stop: float = 0.05,
        batch_size: int = 100,
        verbose: bool = False,
        verbose_every: int = 1,
        output_dir: str = "test_result/explain",
        init_num_samples: int = 1000,
        max_evaluations: Optional[int] = None,
        prebuilt_init=None,
        use_kllucb: bool = True,
        init_state_range: Tuple[int, int] = (25, 65),
        max_init_attempts: int = 40,
        save_graphs: bool = True,
        save_plots: bool = True,
        collect_error_examples: bool = False,
        **kwargs,
    ) -> dict:
        """
        Search for a compact automaton explanation.

        agreement_threshold is the preferred name. agreement_threshold is kept
        only as a backward-compatible alias for existing experiment scripts.
        """
        if "type" in kwargs:
            data_type = kwargs.pop("type")

        if agreement_threshold is None:
            agreement_threshold = kwargs.pop("agreement_threshold", None)
        threshold = 1.0 if agreement_threshold is None else agreement_threshold

        init_automaton_time = 0.0
        self.iteration = 0
        self._init_state(batch_size)

        if prebuilt_init is not None:
            self.learner = prebuilt_init.learner
            self.sample_fcn = self._select_sampler_for_learner()
            self.samplers = [self.sample_fcn]
            origin_automaton = prebuilt_init.initial_dfa.copy()
            self.validation_data = list(getattr(prebuilt_init, "validation_data", []))
            self.validation_labels = np.asarray(getattr(prebuilt_init, "validation_labels", []), dtype=np.float64)
            if verbose:
                print(
                    f"[automata_beam] using prebuilt initial automaton "
                    f"({len(origin_automaton.states)} states), "
                    f"validation samples={len(self.validation_data)}"
                )
        else:
            self.learner = get_learner(automaton_type, new_instance=True)
            self.sample_fcn = self._select_sampler_for_learner()
            self.samplers = [self.sample_fcn]

            init_start = time.perf_counter()
            min_states, max_states = init_state_range
            origin_automaton = None
            init_samples = None
            init_labels = None
            discarded_candidates = []

            for attempt in range(1, max_init_attempts + 1):
                init_samples, init_labels = self.sample_fcn(num_samples=init_num_samples, compute_labels=True)
                positive_samples = [seq for seq, label in zip(init_samples, init_labels) if int(label) == 1]
                negative_samples = [seq for seq, label in zip(init_samples, init_labels) if int(label) == 0]

                candidate = self.learner.create_init_automata(data_type, positive_samples, negative_samples)
                candidate_states = len(candidate.states)
                discarded_candidates.append(candidate)

                if min_states <= candidate_states <= max_states:
                    origin_automaton = candidate
                    if verbose:
                        print(f"[Init DFA] Attempt {attempt}: {candidate_states} states")
                    break

                if verbose:
                    print(f"[Init DFA] Attempt {attempt}: {candidate_states} states, resampling...")

            if origin_automaton is None:
                attempted_states = [len(candidate.states) for candidate in discarded_candidates]
                print(
                    f"  [ERROR] Initial DFA construction failed after {max_init_attempts} attempts: "
                    f"state counts tried={attempted_states}, required range=[{min_states}, {max_states}]"
                )
                return {
                    "automata": [None, None],
                    "states": [None, None],
                    "training_agreements": [0.0, 0.0],
                    "validation_agreements": [0.0, 0.0],
                    "success": False,
                    "reason": f"Initial automaton state count not in range [{min_states}, {max_states}]",
                    "budget_used": 0,
                    "validation_data": [],
                    "validation_labels": np.array([], dtype=np.float64),
                    "num_preds": 0,
                    "false_accept": [],
                    "false_reject": [],
                    "true_reject": [],
                    "init_automaton_time": 0.0,
                }

            for candidate in discarded_candidates:
                if candidate is not origin_automaton:
                    del candidate

            self.validation_data = list(init_samples) if init_samples is not None else []
            self.validation_labels = np.asarray(init_labels if init_labels is not None else [], dtype=np.float64)

            if hasattr(origin_automaton, "make_input_complete"):
                origin_automaton.make_input_complete("sink_state")

            init_automaton_time = time.perf_counter() - init_start

        self.automatas = [origin_automaton]

        if save_graphs:
            try:
                dfa_to_graphviz(origin_automaton, filename="initial_automata", output_dir=output_dir)
            except Exception as exc:
                print(f"  [WARNING] Could not save initial DFA graph: {exc}")

        # Evaluate initial automaton once to populate state.
        (true_accept,), (true_reject,), (total,), (_accepted,) = self.draw_automata_samples(
            [self.automatas[0]],
            batch_size,
        )
        initial_agreement = (true_accept + true_reject) / total if total > 0 else 0.0
        mean = np.array([initial_agreement], dtype=float)

        if use_kllucb and total > 0:
            beta = np.log(1.0 / delta)
            lb = self.dlow_bernoulli(mean, np.array([beta / total]))
            while mean > threshold and lb < threshold - epsilon:
                (ta,), (tr,), (n,), (_acc,) = self.draw_automata_samples([self.automatas[0]], batch_size)
                true_accept += ta
                true_reject += tr
                total += n
                mean = np.array([(true_accept + true_reject) / total] if total > 0 else [0.0])
                lb = self.dlow_bernoulli(mean, np.array([beta / total]))
        else:
            lb = mean

        # Disabled: per the paper's design (Stop Criteria), the search always
        # runs the full DELETE/MERGE/DELTA refinement loop down to the
        # smallest reachable automaton -- there is no "initial automaton
        # already good enough, skip refinement" shortcut. Verified against
        # the paper's own SecureHandshake tau=0.8 numbers (batch_size=1000,
        # delta=0.01, initial training_agreement=0.8655): dlow_bernoulli
        # already returns lb=0.8305 > 0.8 from the very first sample batch,
        # so this check -- if active -- would have returned the unrefined
        # 37-state automaton immediately, contradicting the paper's reported
        # 37->5 state result. Keep disabled.
        # if lb > threshold and is_valid_dfa(self.automatas[0]):
        #     return self._make_result(
        #         origin_automaton,
        #         self.automatas[0],
        #         success=True,
        #         reason="Initial automaton already meets agreement threshold.",
        #         budget_used=1,
        #         init_automaton_time=init_automaton_time,
        #         batch_size=batch_size,
        #         output_dir=output_dir,
        #         save_graphs=save_graphs,
        #         collect_error_examples=collect_error_examples,
        #     )

        best_of_size: Dict[int, List[Any]] = {}
        all_history: List[dict] = []
        iteration_stats: List[dict] = []
        total_candidates_proposed = 0

        # Algorithm 2 (KL-LUCB Guided Beam Search), lines 6-18: propose
        # candidates from the current beam (Delete/Merge on even iterations,
        # Delta on odd), score them with KL-LUCB, keep the top beam_size, and
        # append every scored candidate to all_history for the final
        # selection step below. Runs until the beam is down to 2 states, the
        # evaluation budget is exhausted, or no candidates remain.
        while True:
            if max_evaluations is not None and total_candidates_proposed >= max_evaluations:
                print(f"[Beam] Budget exhausted: {total_candidates_proposed} >= {max_evaluations}")
                break

            op_label = self._iteration_op_label(self.iteration)

            propose_start = time.perf_counter()
            candidates = self.learner.propose_automata(
                self.automatas,
                self.state,
                self.iteration,
                best_of_size.get(self.iteration, []),
                output_dir,
                beam_size,
                batch_size,
            )
            self._profile_totals["propose_automata"] += time.perf_counter() - propose_start
            # Keep every proposed candidate alive for the rest of the search;
            # see _id_reuse_guard's definition in __init__ for why.
            self._id_reuse_guard.extend(candidates)

            if not candidates:
                print("[Beam] No candidates proposed. Stopping.")
                break

            if max_evaluations is not None:
                remaining_budget = max_evaluations - total_candidates_proposed
                if remaining_budget <= 0:
                    break
                if len(candidates) > remaining_budget:
                    sampled_idx = np.random.choice(len(candidates), size=remaining_budget, replace=False)
                    candidates = [candidates[int(i)] for i in sorted(sampled_idx)]

            raw_candidate_count = len(candidates)
            total_candidates_proposed += raw_candidate_count

            valid_candidates = [candidate for candidate in candidates if is_valid_dfa(candidate)]
            if not valid_candidates:
                print(f"[Beam Iter {self.iteration}] op={op_label}, candidates=0, valid=0. Stopping.")
                break
            candidates = valid_candidates

            budget_text = (
                f", evaluations={total_candidates_proposed}/{max_evaluations}"
                if max_evaluations is not None
                else f", evaluations={total_candidates_proposed}"
            )
            print(
                f"[Beam Iter {self.iteration}] op={op_label}, "
                f"candidates={raw_candidate_count}, valid={len(candidates)}"
                f"{budget_text}"
            )

            kllucb_start = time.perf_counter()
            stats = self.get_init_stats(candidates)

            if use_kllucb:
                _ = self.kllucb_automata(
                    candidates,
                    stats,
                    epsilon,
                    delta,
                    batch_size,
                    min(beam_size, len(candidates)),
                    verbose=verbose,
                    verbose_every=verbose_every,
                )
            else:
                unsampled_idx = np.where(stats["n_samples"] == 0)[0]
                if len(unsampled_idx) > 0:
                    unsampled_automata = [candidates[int(i)] for i in unsampled_idx]
                    ta, tr, n, acc = self.draw_automata_samples(unsampled_automata, batch_size)
                    stats["positives"][unsampled_idx] += np.asarray(ta, dtype=float)
                    stats["negatives"][unsampled_idx] += np.asarray(tr, dtype=float)
                    stats["n_samples"][unsampled_idx] += np.asarray(n, dtype=float)
                    stats["n_accepted"][unsampled_idx] += np.asarray(acc, dtype=float)

            agreements = self._agreements_from_stats(stats)
            sorted_indices = np.argsort(agreements)[::-1]
            candidate_indices = sorted_indices[: min(beam_size, len(sorted_indices))]
            beam = [candidates[int(i)] for i in candidate_indices]
            best_of_size[self.iteration + 1] = beam
            self.automatas = beam

            selected_stats = self.get_init_stats(beam)
            selected_agreements = self._agreements_from_stats(selected_stats)

            if use_kllucb:
                beta = np.log(1.0 / (delta / (1 + (beam_size - 1) * len(candidates))))
                n_samples = selected_stats["n_samples"]
                kl_constraints = np.divide(beta, n_samples, out=np.zeros_like(n_samples), where=n_samples != 0)
                lbs = self.dlow_bernoulli(selected_agreements, kl_constraints)
                ubs = self.dup_bernoulli(selected_agreements, kl_constraints)

                continue_sampling = self.to_sample(selected_agreements, ubs, lbs, threshold, epsilon_stop)
                while continue_sampling.any():
                    selected_to_sample = [beam[int(i)] for i in np.where(continue_sampling)[0]]
                    ta, tr, n, acc = self._draw_kllucb_samples(selected_to_sample, batch_size)

                    sample_idx = np.where(continue_sampling)[0]
                    selected_stats["positives"][sample_idx] += np.asarray(ta, dtype=float)
                    selected_stats["negatives"][sample_idx] += np.asarray(tr, dtype=float)
                    selected_stats["n_samples"][sample_idx] += np.asarray(n, dtype=float)
                    selected_stats["n_accepted"][sample_idx] += np.asarray(acc, dtype=float)

                    selected_agreements = self._agreements_from_stats(selected_stats)
                    n_samples = selected_stats["n_samples"]
                    kl_constraints = np.divide(beta, n_samples, out=np.zeros_like(n_samples), where=n_samples != 0)
                    lbs = self.dlow_bernoulli(selected_agreements, kl_constraints)
                    ubs = self.dup_bernoulli(selected_agreements, kl_constraints)
                    continue_sampling = self.to_sample(selected_agreements, ubs, lbs, threshold, epsilon_stop)
            else:
                lbs = np.zeros_like(selected_agreements, dtype=float)
                ubs = np.ones_like(selected_agreements, dtype=float)
            self._profile_totals["kllucb_eval"] += time.perf_counter() - kllucb_start

            for automaton, agreement, lower_bound in zip(beam, selected_agreements, lbs):
                if not is_valid_dfa(automaton):
                    continue
                all_history.append({
                    "automata": automaton,
                    "training_agreement": float(agreement),
                    "lb": float(lower_bound),
                    "states": len(automaton.states),
                })

            round_stats = {
                "iteration": self.iteration,
                "training_agreements": [
                    self.get_automata_metadata(d, success=True)["training_agreement"]
                    for d in beam
                ],
                "states": [len(d.states) for d in beam],
            }
            iteration_stats.append(round_stats)
            # Minimal per-iteration summary: enough to track search progress
            # without printing large candidate tables or CXP details.
            best_agreement = max(round_stats["training_agreements"]) if round_stats["training_agreements"] else 0.0
            best_states = min(round_stats["states"]) if round_stats["states"] else 0
            # Samples actually spent on each selected candidate's agreement estimate
            # this round (KL-LUCB may have drawn several extra batches beyond the
            # initial one for candidates that were hard to distinguish).
            n_samples_used = [int(n) for n in selected_stats["n_samples"]]
            print(
                f"[Beam Iter {self.iteration}] selected={len(beam)}, "
                f"best_agreement={best_agreement:.4f}, best_states={best_states}, "
                f"n_samples={n_samples_used}"
            )
            self.iteration += 1

        if self.profile_time:
            learner_totals = getattr(self.learner, "_profile_totals", {})
            parts = [f"propose_automata={self._profile_totals['propose_automata']:.2f}s"]
            for key in ("propose_delete", "propose_merge", "propose_delta_total", "delta_cxp_analysis"):
                if key in learner_totals:
                    parts.append(f"{key}={learner_totals[key]:.2f}s")
            parts.append(f"kllucb_eval={self._profile_totals['kllucb_eval']:.2f}s")
            print("[Profile] " + ", ".join(parts))
            print(
                f"[Profile]   kllucb_eval breakdown: "
                f"sample_predict={self._profile_totals['sample_predict']:.2f}s, "
                f"accept_check={self._profile_totals['accept_check']:.2f}s"
            )

        if save_plots and iteration_stats:
            try:
                plot_beam_stats(iteration_stats, beam_size, output_dir=output_dir)
            except Exception as exc:
                print(f"  [WARNING] Could not save beam plot: {exc}")

        # Algorithm 2, lines 19-28: final selection over the WHOLE search
        # history (every candidate scored across all iterations, not just the
        # last beam), matching arg min / arg max over states and agreement
        # respectively -- ties are broken implicitly (first match in
        # all_history's insertion order), which in practice never happens
        # across our experiments. Among candidates with training_agreement
        # >= threshold, return the one with the fewest states; if none
        # qualify, fall back to the single highest-agreement candidate in
        # the whole history (best-effort). This rule is load-bearing for
        # every reported result and must not change.
        if all_history:
            qualified = [record for record in all_history if record["training_agreement"] >= threshold]
            if qualified:
                best = min(qualified, key=lambda record: record["states"])
                reason = "Found candidate meeting agreement threshold."
                return self._make_result(
                    origin_automaton,
                    best["automata"],
                    success=True,
                    reason=reason,
                    budget_used=total_candidates_proposed,
                    init_automaton_time=init_automaton_time,
                    batch_size=batch_size,
                    output_dir=output_dir,
                    save_graphs=save_graphs,
                    collect_error_examples=collect_error_examples,
                    final_training_agreement=best["training_agreement"],
                )

            best = max(all_history, key=lambda record: record["training_agreement"])
            reason = (
                f"No candidate meets agreement >= {threshold}. "
                "Returning best-effort candidate with highest training agreement."
            )
            return self._make_result(
                origin_automaton,
                best["automata"],
                success=False,
                reason=reason,
                budget_used=total_candidates_proposed,
                init_automaton_time=init_automaton_time,
                batch_size=batch_size,
                output_dir=output_dir,
                save_graphs=save_graphs,
                collect_error_examples=collect_error_examples,
                final_training_agreement=best["training_agreement"],
            )

        return self._make_result(
            origin_automaton,
            origin_automaton,
            success=False,
            reason="No candidates generated during beam search. Returning initial automaton.",
            budget_used=total_candidates_proposed if total_candidates_proposed > 0 else 1,
            init_automaton_time=init_automaton_time,
            batch_size=batch_size,
            output_dir=output_dir,
            save_graphs=save_graphs,
            collect_error_examples=collect_error_examples,
        )
