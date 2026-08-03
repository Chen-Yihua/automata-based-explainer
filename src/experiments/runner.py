"""
Minimal runner for local automata explanations.

Use this when you only want to run one instance with:
    Beam Search + SA + GA + PSO

This file avoids regular-vs-realworld branching, but saves shared_init.pkl
so baseline parameter tuning can reuse the exact same initial DFA and samples. Experiment scripts should prepare:
    predict_fn, alphabet, test_instance, cfg
then call run_search_suite().
"""
from __future__ import annotations

import os
import pickle
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np

from explainer.automata_beam import AutomataBeamSearch
from learner.dfa_learner import DFALearner, DFASampler
from baselines.search_baselines import SharedInit, ga_dfa_search, pso_dfa_search, sa_dfa_search


DEFAULT_METHODS = ("beam", "sa", "ga", "pso")


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return float(value.reshape(-1)[-1])
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return float(value[-1])
    return float(value)


def _state_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[-1]) if value.size else 0
    if isinstance(value, (list, tuple)):
        return int(value[-1]) if value else 0
    return int(value)


def make_sampler(
    predict_fn: Callable[[Sequence[Sequence]], np.ndarray],
    alphabet: Sequence,
    test_instance: Sequence,
    cfg: dict,
) -> DFASampler:
    """Create and bind the DFA sampler for one explained instance."""
    sampler = DFASampler(
        predictor=predict_fn,
        alphabet=list(alphabet),
        seed=cfg.get("seed", 42),
        edit_distance=cfg.get("edit_distance", 1),
        use_prediction_cache=cfg.get("use_prediction_cache", True),
        prediction_cache_max_size=cfg.get("prediction_cache_max_size", 200000),
    )
    sampler.set_instance_label(list(test_instance))
    sampler.set_n_covered(cfg.get("n_covered", 10))
    return sampler


def run_beam(
    sampler: DFASampler,
    cfg: dict,
    output_dir: str,
) -> tuple[AutomataBeamSearch, dict]:
    """Run AutomataBeamSearch directly, without AnchorTabular or cache."""
    search = AutomataBeamSearch(
        samplers=[sampler],
        predictor=sampler.predictor,
        sample_cache_size=cfg.get("sample_cache_size", 1000),
        parallel=cfg.get("parallel", True),
        n_jobs=cfg.get("n_jobs", 4),
        profile_time=cfg.get("profile_time", False),
    )
    result = search.automata_beam(
        data_type=cfg.get("data_type", "Tabular"),
        automaton_type=cfg.get("automaton_type", "DFA"),
        alphabet=list(sampler.alphabet),
        agreement_threshold=cfg.get("agreement_threshold", 1.0),
        delta=cfg.get("delta", 0.05),
        epsilon=cfg.get("tau", cfg.get("epsilon", 0.1)),
        beam_size=cfg.get("beam_size", 1),
        epsilon_stop=cfg.get("epsilon_stop", 0.05),
        batch_size=cfg.get("batch_size", 100),
        init_num_samples=cfg.get("init_num_samples", 1000),
        max_evaluations=cfg.get("max_evaluations"),
        use_kllucb=cfg.get("use_kllucb", True),
        output_dir=os.path.join(output_dir, "beam"),
        save_graphs=cfg.get("save_graphs", True),
        save_plots=cfg.get("save_plots", True),
        collect_error_examples=cfg.get("collect_error_examples", False),
        verbose=cfg.get("verbose", False),
    )
    return search, result


def build_shared_init(
    beam_search: AutomataBeamSearch,
    beam_result: dict,
    sampler: DFASampler,
    cfg: dict,
) -> SharedInit:
    """Build the shared object required by SA / GA / PSO."""
    automata_pair = beam_result.get("automata") or []
    if automata_pair and automata_pair[0] is not None:
        initial_dfa = automata_pair[0].copy()
    elif beam_search.automatas:
        initial_dfa = beam_search.automatas[0].copy()
    else:
        raise ValueError("Beam search did not produce an initial DFA.")

    training_data, training_labels = sampler(
        num_samples=cfg.get("baseline_train_samples", cfg.get("batch_size", 100)),
        compute_labels=True,
    )

    return SharedInit(
        initial_dfa=initial_dfa,
        learner=DFALearner(),
        validation_data=list(beam_result.get("validation_data", beam_search.validation_data)),
        validation_labels=np.asarray(beam_result.get("validation_labels", beam_search.validation_labels)),
        training_data=list(training_data),
        training_labels=np.asarray(training_labels),
    )


def save_shared_init(shared: SharedInit, output_dir: str) -> str:
    """Persist SharedInit for later baseline parameter tuning."""
    shared_dir = os.path.join(output_dir, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    shared_init_path = os.path.join(shared_dir, "shared_init.pkl")
    with open(shared_init_path, "wb") as f:
        pickle.dump(shared, f)
    print(
        f"  [SharedInit] Saved to {shared_init_path} "
        f"({len(shared.initial_dfa.states)} states, "
        f"{len(shared.training_data)} training samples, "
        f"{len(shared.validation_data)} validation samples)"
    )
    return shared_init_path



def _normalise_method_result(
    method: str,
    result: dict,
    elapsed: float,
    initial_train_agreement: float = 0.0,
    initial_validation_agreement: float = 0.0,
    initial_states: int = 0,
) -> dict:
    dfa = result.get("automata")
    states = result.get("size")
    if states is None and dfa is not None and hasattr(dfa, "states"):
        states = len(dfa.states)

    init_train = _scalar(
        result.get("initial_train_agreement", result.get("initial_training_agreement", initial_train_agreement)),
        initial_train_agreement,
    )
    init_val = _scalar(
        result.get("initial_val_agreement", result.get("initial_validation_agreement", initial_validation_agreement)),
        initial_validation_agreement,
    )
    init_states = _state_count(result.get("initial_states", initial_states))

    return {
        "method": method,
        "initial_train_agreement": init_train,
        "initial_validation_agreement": init_val,
        "initial_states": init_states,
        "train_agreement": _scalar(result.get("training_agreement")),
        "validation_agreement": _scalar(result.get("validation_agreement")),
        "states": int(states or 0),
        "time": float(elapsed),
        "success": bool(result.get("success", False)),
        "reason": result.get("reason", ""),
        "raw": result,
    }


def run_baseline(
    method: str,
    shared: SharedInit,
    cfg: dict,
    output_dir: str,
    initial_train_agreement: float = 0.0,
    initial_validation_agreement: float = 0.0,
    initial_states: int = 0,
) -> dict:
    """Run one baseline method."""
    os.makedirs(output_dir, exist_ok=True)
    threshold = cfg.get("agreement_threshold", cfg.get("accuracy_threshold", 1.0))
    common = dict(
        data_type=cfg.get("data_type", "Tabular"),
        shared_init=shared,
        agreement_threshold=threshold,
        init_num_samples=cfg.get("init_num_samples", 1000),
        batch_size=cfg.get("batch_size", 100),
        output_dir=output_dir,
        max_evaluations=cfg.get("max_evaluations", 500),
    )

    if method == "sa":
        fn = sa_dfa_search
        extra = dict(
            beam_size=1,
            steps=cfg.get("sa_steps", 500),
            T_max=cfg.get("sa_t_max", 10.0),
            T_min=cfg.get("sa_t_min", 0.001),
            sa_candidate_pool_size=cfg.get("sa_candidate_pool_size", 10),
        )
    elif method == "ga":
        fn = ga_dfa_search
        extra = dict(
            population_size=cfg.get("ga_population_size", 10),
            tournament_size=cfg.get("ga_tournament_size", 2),
        )
    elif method == "pso":
        fn = pso_dfa_search
        extra = dict(
            beam_size=1,
            n_particles=cfg.get("pso_particles", 5),
            pso_max_ops_per_iteration=cfg.get("pso_max_ops_per_iteration", 1),
            pso_candidate_pool_size=cfg.get("pso_candidate_pool_size", 5),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    start = time.time()
    raw = fn(**common, **extra)
    return _normalise_method_result(
        method,
        raw,
        elapsed=time.time() - start,
        initial_train_agreement=initial_train_agreement,
        initial_validation_agreement=initial_validation_agreement,
        initial_states=initial_states,
    )


def run_search_suite(
    predict_fn: Callable[[Sequence[Sequence]], np.ndarray],
    alphabet: Sequence,
    test_instance: Sequence,
    cfg: dict,
    output_dir: str,
    methods: Iterable[str] = DEFAULT_METHODS,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, dict]:
    """
    Run Beam Search and optional SA / GA / PSO for one instance.

    Parameters
    ----------
    predict_fn
        Teacher model prediction function.
    alphabet
        Symbols used by the sampler.
    test_instance
        The sequence to explain.
    cfg
        Experiment config dictionary.
    output_dir
        Directory for optional method artifacts.
    methods
        Any subset of {"beam", "sa", "ga", "pso"}.
    """
    print(f"\n[RUN SEARCH SUITE] methods={methods}  output_dir={output_dir}")

    methods = tuple(methods)
    os.makedirs(output_dir, exist_ok=True)

    sampler = make_sampler(predict_fn, alphabet, test_instance, cfg)
    results: Dict[str, dict] = {}
    meta = dict(metadata or {})
    meta.setdefault("agreement_threshold", cfg.get("agreement_threshold", 1.0))

    if "beam" not in methods:
        raise ValueError("Beam must run first because SA/GA/PSO share its initial DFA.")

    start = time.time()
    beam_search, beam_raw = run_beam(sampler, cfg, output_dir)
    beam_elapsed = time.time() - start
    initial_train = _scalar(beam_raw.get("initial_training_agreement"))
    initial_val = _scalar(beam_raw.get("initial_validation_agreement"))
    initial_states = _state_count(beam_raw.get("initial_state"))
    meta.setdefault("initial_states", initial_states)
    results["_meta"] = meta

    results["beam"] = {
        "method": "beam",
        "initial_train_agreement": initial_train,
        "initial_validation_agreement": initial_val,
        "initial_states": initial_states,
        "train_agreement": _scalar(beam_raw.get("final_training_agreement")),
        "validation_agreement": _scalar(beam_raw.get("final_validation_agreement")),
        "states": _state_count(beam_raw.get("final_state")),
        "time": beam_elapsed,
        "success": bool(beam_raw.get("success", False)),
        "reason": beam_raw.get("reason", ""),
        "raw": beam_raw,
    }

    shared = build_shared_init(beam_search, beam_raw, sampler, cfg)
    if cfg.get("save_shared_init", True):
        try:
            save_shared_init(shared, output_dir)
        except Exception as exc:
            print(f"  [WARNING] Could not save shared_init.pkl: {exc}")
    for method in methods:
        if method == "beam":
            continue
        results[method] = run_baseline(
            method=method,
            shared=shared,
            cfg=cfg,
            output_dir=os.path.join(output_dir, method),
        )

    return results


METHOD_LABELS = {
    "beam": "BeamSearch",
    "sa": "SA",
    "ga": "GA",
    "pso": "PSO",
}


def _format_delta(initial_value: float, final_value: float, ok: bool | None = None) -> str:
    mark = ""
    if ok is not None:
        mark = " ✓" if ok else " ✗"
    return f"{initial_value:.4f}→{final_value:.4f}{mark}"


def _infer_dataset_name(title: str, meta: Dict[str, Any]) -> str:
    if meta.get("dataset"):
        return str(meta["dataset"])
    if "_instance_" in title:
        return title.split("_instance_")[0]
    return title


def _print_one_suite(title: str, suite_results: Dict[str, dict]) -> None:
    meta = suite_results.get("_meta", {}) if isinstance(suite_results.get("_meta", {}), dict) else {}
    beam = suite_results.get("beam", {}) or {}

    initial_train = _scalar(beam.get("initial_train_agreement"))
    initial_val = _scalar(beam.get("initial_validation_agreement"))
    initial_states = int(beam.get("initial_states") or meta.get("initial_states") or 0)
    threshold = float(meta.get("agreement_threshold", 1.0))
    dataset_name = _infer_dataset_name(str(title), meta)

    header_parts = []
    if meta.get("clf_train_acc") is not None:
        header_parts.append(f"clf_train={float(meta['clf_train_acc']):.4f}")
    if meta.get("clf_test_acc") is not None:
        header_parts.append(f"clf_test={float(meta['clf_test_acc']):.4f}")
    if meta.get("teacher_states") is not None:
        header_parts.append(f"teacher_states={int(meta['teacher_states'])}")
    if initial_states:
        header_parts.append(f"initial_states={initial_states}")

    print("\n")
    if header_parts:
        print(f"  {dataset_name}  ({'  '.join(header_parts)})")
    else:
        print(f"  {dataset_name}")

    print(f"\n  Initial (RPNI):  train={initial_train:.4f}  validation={initial_val:.4f}")
    print("  " + "─" * 96)
    print(
        f"  | {'Method':12s} | {'Train (Init→Final)':20s} | "
        f"{'Validation (Init→Final)':24s} | {'States':10s} | {'Time(s)':10s} |"
    )
    print("  " + "─" * 96)

    for method in ("beam", "sa", "ga", "pso"):
        res = suite_results.get(method)
        label = METHOD_LABELS.get(method, method)
        if res is None:
            print(f"  | {label:12s} | {'N/A':20s} | {'N/A':24s} | {'N/A':>10s} | {'N/A':>10s} |")
            continue

        init_train = _scalar(res.get("initial_train_agreement"), initial_train)
        init_val = _scalar(res.get("initial_validation_agreement"), initial_val)
        final_train = _scalar(res.get("train_agreement"))
        final_val = _scalar(res.get("validation_agreement"))
        final_states = int(res.get("states", 0) or 0)
        elapsed = _scalar(res.get("time"))
        ok = final_train >= threshold

        train_text = _format_delta(init_train, final_train, ok)
        val_text = _format_delta(init_val, final_val, None)
        print(
            f"  | {label:12s} | {train_text:20s} | "
            f"{val_text:24s} | {final_states:10d} | {elapsed:10.1f} |"
        )

    print("  " + "─" * 96)


def print_suite_summary(results: Dict[str, dict]) -> None:
    """
    Print results from either:
    1. one run_search_suite() output:
       {'beam': {...}, 'sa': {...}, 'ga': {...}, 'pso': {...}}

    2. many instances:
       {'SecureHandshake_instance_00': {'beam': {...}, ...},
        'SecureHandshake_instance_01': None,
        ...}
    """
    if not results:
        print("\n[SUMMARY] No results.")
        return

    # Case 1: results is already one suite.
    if any(method in results for method in ("beam", "sa", "ga", "pso")) and all(
        (key in ("beam", "sa", "ga", "pso")) or key.startswith("_")
        for key in results.keys()
    ):
        _print_one_suite("Search suite summary", results)
        return

    # Case 2: results is a dict of instance_key -> suite_results / None.
    for result_key, suite_results in sorted(results.items()):
        if suite_results is None:
            print("\n" + "=" * 80)
            print(f"{result_key}")
            print("=" * 80)
            print("[NO DATA] This instance failed or was skipped.")
            continue

        if not isinstance(suite_results, dict):
            print("\n" + "=" * 80)
            print(f"{result_key}")
            print("=" * 80)
            print(f"[INVALID DATA] Expected dict, got {type(suite_results).__name__}.")
            continue

        # Some callers may wrap the suite under a nested key.
        if not any(method in suite_results for method in ("beam", "sa", "ga", "pso")):
            print("\n" + "=" * 80)
            print(f"{result_key}")
            print("=" * 80)
            print("[NO METHOD RESULTS] Missing beam/sa/ga/pso keys.")
            continue

        _print_one_suite(str(result_key), suite_results)
