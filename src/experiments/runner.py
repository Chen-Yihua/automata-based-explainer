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

import csv
import glob
import os
import pickle
import re
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np

from explainer.automata_beam import AutomataBeamSearch
from learner.dfa_learner import DFALearner, DFASampler
from baselines.search_baselines import SharedInit, ga_dfa_search, pso_dfa_search, sa_dfa_search


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_METHODS = ("beam", "sa", "ga", "pso")


_SA_CONFIG_RE = re.compile(r"pool=(\d+)")
_GA_CONFIG_RE = re.compile(r"pop=(\d+)")
_PSO_CONFIG_RE = re.compile(r"n_particles=(\d+),pool=(\d+),ops=(\d+)")


def _find_latest_tuned_params_csv(search_root: Optional[str] = None) -> Optional[str]:
    """Find the most recently written best_by_algo_cross_task.csv produced by
    baselines.tune_baseline_params under test_result/tune_*/, if any."""
    root = search_root or os.path.join(PROJECT_ROOT, "test_result")
    matches = glob.glob(os.path.join(root, "tune_*", "best_by_algo_cross_task.csv"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _load_tuned_baseline_params(csv_path: Optional[str] = None) -> Dict[str, int]:
    """Load cross-task-tuned SA/GA/PSO hyperparameters written by
    tune_baseline_params.py's write_best_by_algo_cross_task_table(), if a
    result is available. Returns {} (caller keeps its own hardcoded
    defaults) when no tuned-params file exists -- tuning is optional, not a
    prerequisite for running the main experiment.
    """
    path = csv_path or _find_latest_tuned_params_csv()
    if not path or not os.path.isfile(path):
        return {}

    tuned: Dict[str, int] = {}
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                algo = (row.get("algo") or "").upper()
                config = row.get("config") or ""
                if algo == "SA":
                    m = _SA_CONFIG_RE.search(config)
                    if m:
                        tuned["sa_candidate_pool_size"] = int(m.group(1))
                elif algo == "GA":
                    m = _GA_CONFIG_RE.search(config)
                    if m:
                        tuned["ga_population_size"] = int(m.group(1))
                elif algo == "PSO":
                    m = _PSO_CONFIG_RE.search(config)
                    if m:
                        tuned["pso_particles"] = int(m.group(1))
                        tuned["pso_candidate_pool_size"] = int(m.group(2))
                        tuned["pso_max_ops_per_iteration"] = int(m.group(3))
    except Exception as exc:
        print(f"  [WARNING] Could not read tuned baseline params from {path}: {exc}")
        return {}

    if tuned:
        print(f"  [Tuned params] Using {tuned} from {path}")
    return tuned


class NoInitialDFAError(RuntimeError):
    """Beam search never produced a usable initial DFA (see `.reason`).

    SA/GA/PSO can't run without it (they share beam's initial DFA via
    SharedInit), so callers should treat this as "skip this instance",
    not a generic crash.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Beam search did not produce an initial DFA: {reason}")


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
        max_len=cfg.get("max_length"),
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
        init_state_range=cfg.get("init_state_range", (25, 65)),
        max_init_attempts=cfg.get("max_init_attempts", 40),
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
    cfg: dict,
) -> SharedInit:
    """Build the shared object required by SA / GA / PSO.

    training_data/training_labels are the first `batch_size` samples drawn,
    in draw order, from AutomataBeamSearch.state["data"]/["labels"] (which
    accumulate every draw made during the whole search). In the normal case
    this is exactly the one batch Beam Search drew to evaluate the initial
    (origin) automaton, before KL-LUCB's adaptive sampling grows the pool
    further. It is only that single clean batch as long as the sampler
    returns a full batch on that first draw; if the local neighborhood is so
    small that DFASampler.perturbation under-fills a batch, these are instead
    the first `batch_size` entries spanning that draw plus the start of the
    next one (still label-aligned, but no longer literally one sampling call,
    and it may contain duplicate sequences). SA/GA/PSO then hold this batch
    fixed for every candidate they evaluate.
    """
    automata_pair = beam_result.get("automata") or []
    if automata_pair and automata_pair[0] is not None:
        initial_dfa = automata_pair[0].copy()
    elif beam_search.automatas:
        initial_dfa = beam_search.automatas[0].copy()
    else:
        raise NoInitialDFAError(beam_result.get("reason") or "unknown reason")

    batch_size = cfg.get("batch_size", 100)
    training_data = list(beam_search.state["data"][:batch_size])
    training_labels = np.asarray(beam_search.state["labels"][:batch_size])

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
        max_evaluations=cfg.get("max_evaluations"),
    )

    # Cross-task-tuned SA/GA/PSO hyperparameters (baselines.tune_baseline_params
    # output) are used by default when available; an explicit value in cfg
    # still wins, and if no tuned-params file exists at all this falls back
    # to the same hardcoded defaults as before.
    tuned = _load_tuned_baseline_params(cfg.get("tuned_params_csv"))

    if method == "sa":
        fn = sa_dfa_search
        extra = dict(
            beam_size=1,
            steps=cfg.get("sa_steps", 500),
            T_max=cfg.get("sa_t_max", 10.0),
            T_min=cfg.get("sa_t_min", 0.001),
            sa_candidate_pool_size=cfg.get("sa_candidate_pool_size", tuned.get("sa_candidate_pool_size", 10)),
        )
    elif method == "ga":
        fn = ga_dfa_search
        extra = dict(
            population_size=cfg.get("ga_population_size", tuned.get("ga_population_size", 10)),
            tournament_size=cfg.get("ga_tournament_size", 2),
        )
    elif method == "pso":
        fn = pso_dfa_search
        extra = dict(
            beam_size=1,
            n_particles=cfg.get("pso_particles", tuned.get("pso_particles", 5)),
            pso_max_ops_per_iteration=cfg.get("pso_max_ops_per_iteration", tuned.get("pso_max_ops_per_iteration", 1)),
            pso_candidate_pool_size=cfg.get("pso_candidate_pool_size", tuned.get("pso_candidate_pool_size", 5)),
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
    beam_wall_time = time.time() - start
    # Exclude the beam-only overhead that baseline runs don't pay: building the
    # initial DFA via RPNI (up to max_init_attempts resamples) plus rendering
    # the initial-DFA graph and the per-iteration stats plot. Baselines reuse
    # beam's already-built initial DFA (via SharedInit) and only pay for one
    # final graphviz render + validation eval, same as beam's own final steps,
    # so those are left in both and not subtracted here.
    beam_overhead = _scalar(beam_raw.get("init_automaton_time")) + _scalar(beam_raw.get("plot_stats_time"))
    beam_elapsed = max(0.0, beam_wall_time - beam_overhead)
    initial_train = _scalar(beam_raw.get("initial_training_agreement"))
    initial_val = _scalar(beam_raw.get("initial_validation_agreement"))
    initial_states = _state_count(beam_raw.get("initial_state"))
    meta.setdefault("initial_states", initial_states)
    results["_meta"] = meta

    beam_evaluations_used = beam_raw.get("budget_used")
    results["beam"] = {
        "method": "beam",
        "initial_train_agreement": initial_train,
        "initial_validation_agreement": initial_val,
        "initial_states": initial_states,
        "train_agreement": _scalar(beam_raw.get("final_training_agreement")),
        "validation_agreement": _scalar(beam_raw.get("final_validation_agreement")),
        "states": _state_count(beam_raw.get("final_state")),
        "time": beam_elapsed,
        "wall_time": beam_wall_time,
        "excluded_overhead_time": beam_overhead,
        "evaluations_used": beam_evaluations_used,
        "success": bool(beam_raw.get("success", False)),
        "reason": beam_raw.get("reason", ""),
        "raw": beam_raw,
    }

    try:
        shared = build_shared_init(beam_search, beam_raw, cfg)
    except NoInitialDFAError as exc:
        # Not a crash: beam legitimately couldn't build an initial DFA in
        # init_state_range (often because this instance's true local behavior
        # near the teacher is simpler than the range assumes, so resampling
        # within max_init_attempts can't fix it). SA/GA/PSO need beam's
        # initial DFA, so they can't run either; skip them and say why, instead
        # of the caller's generic except-and-traceback treating this the same
        # as an actual bug.
        print(f"  [SKIP] {exc}")
        meta["skipped"] = True
        meta["skip_reason"] = exc.reason
        return results

    if cfg.get("save_shared_init", True):
        try:
            save_shared_init(shared, output_dir)
        except Exception as exc:
            print(f"  [WARNING] Could not save shared_init.pkl: {exc}")

    # Give SA/GA/PSO the same evaluation budget beam actually spent on this
    # instance, instead of a fixed constant from cfg: beam can converge (or
    # give up) well short of cfg["max_evaluations"] via its own stopping
    # conditions (KL-LUCB bound closing, no more states to delete, agreement
    # threshold reached), and a fixed baseline cap disconnected from that
    # would silently let baselines search more (or less) than beam actually
    # did on that particular instance. cfg["max_evaluations"] remains the
    # ceiling beam itself is capped at; only the baselines' copy is
    # overridden here.
    baseline_cfg = dict(cfg)
    if beam_evaluations_used:
        baseline_cfg["max_evaluations"] = int(beam_evaluations_used)
        print(f"  [Baseline budget] Using beam's actual evaluations ({beam_evaluations_used}) as SA/GA/PSO max_evaluations")

    for method in methods:
        if method == "beam":
            continue
        results[method] = run_baseline(
            method=method,
            shared=shared,
            cfg=baseline_cfg,
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
    if meta.get("clf_test_acc_novel") is not None:
        header_parts.append(f"clf_test_novel={float(meta['clf_test_acc_novel']):.4f}")
    if meta.get("teacher_states") is not None:
        header_parts.append(f"teacher_states={int(meta['teacher_states'])}")
    if initial_states:
        header_parts.append(f"initial_states={initial_states}")

    print("\n")
    if header_parts:
        print(f"  {dataset_name}  ({'  '.join(header_parts)})")
    else:
        print(f"  {dataset_name}")

    if meta.get("skipped"):
        print(f"\n  [SKIPPED] {meta.get('skip_reason', 'beam produced no initial DFA')}")
        return

    print(f"\n  Initial (RPNI):  train={initial_train:.4f}  validation={initial_val:.4f}")
    beam_evals = beam.get("evaluations_used")
    if beam_evals:
        print(f"  Beam evaluations used: {int(beam_evals)}  (SA/GA/PSO max_evaluations budget)")
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
