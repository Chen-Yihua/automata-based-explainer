"""
Baseline Comparison: DFA Optimization with Beam Search, GA, SA, PSO
====================================================================

Fair comparison of DFA search methods:
- Shared starting condition: same RPNI initial DFA + same perturbation samples
- Shared candidate generation: same propose_automata (DELETE / MERGE / DELTA)
- Variable: search algorithm (Beam Search, GA, SA, PSO)

This script is self-contained: all DFA operations (clone, delete-state,
merge-states, evaluate) are implemented inline so the file runs without the
external_modules that are not part of the repository.

The beam search implementation here is a standalone re-implementation that
mirrors the logic in AnchorBaseBeam.anchor_beam / DFALearner.propose_automata.
The original beam search in anchor_base.py is preserved and untouched.

Requirements:
    pip install aalpy deap simanneal pyswarms
"""

import sys
import os
import time
import random
import copy
import gc
import itertools
from collections import defaultdict

import numpy as np

# ── project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
MODIFIED_MODULES = os.path.join(PROJECT_ROOT, 'modified_modules')

for path in [MODIFIED_MODULES, SRC_PATH, EXTERNAL_MODULES, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from aalpy.learning_algs import run_RPNI
from aalpy.automata.Dfa import Dfa, DfaState

# ── optional: tee logging ─────────────────────────────────────────────────────
try:
    from tee import Tee
    _TEE_AVAILABLE = True
except ImportError:
    _TEE_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (edit here to change the experiment)
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Alphabet & test instance ──────────────────────────────────────────────────
ALPHABET = ['a', 'b', 'c', 'd']
TEST_INSTANCE = ['a', 'a', 'a', 'c', 'a', 'a', 'b', 'b']  # L6 example

# ── Perturbation / RPNI sample budget ────────────────────────────────────────
INIT_NUM_SAMPLES = 500   # how many perturbed sequences to generate
EDIT_DISTANCE = 3        # max edit distance for perturbation

# ── Classifier (simple membership oracle) ────────────────────────────────────
# For the standalone comparison we use the L6 language oracle directly.
# Replace `predict_fn` with your trained NN classifier when needed.
def _l6_accept(seq):
    """L6: |#a - #b| mod 3 == 0"""
    a_count = list(seq).count('a')
    b_count = list(seq).count('b')
    return int(abs(a_count - b_count) % 3 == 0)

def predict_fn(seqs):
    return np.array([_l6_accept(s) for s in seqs])

# ── Search budget ─────────────────────────────────────────────────────────────
MAX_ITERATIONS = 6        # outer loop iterations for each algorithm
BEAM_SIZE = 3             # beam width; also GA population size / SA neighbors
ACCURACY_THRESHOLD = 0.85
STATE_THRESHOLD = 8

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_result', 'baseline_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED DFA UTILITIES
#  (mirrors dfa_learner.py but without the external_modules dependency)
# ══════════════════════════════════════════════════════════════════════════════

def clone_dfa(dfa: Dfa) -> Dfa:
    """Deep clone a DFA."""
    old_to_new = {}
    for old_s in dfa.states:
        s_new = DfaState(old_s.state_id, is_accepting=old_s.is_accepting)
        s_new.prefix = []
        old_to_new[old_s] = s_new
    for old_s in dfa.states:
        for sym, tgt in old_s.transitions.items():
            if tgt not in old_to_new:
                ghost = DfaState(
                    getattr(tgt, 'state_id', str(tgt)),
                    is_accepting=getattr(tgt, 'is_accepting', False)
                )
                ghost.prefix = []
                old_to_new[tgt] = ghost
            old_to_new[old_s].transitions[sym] = old_to_new[tgt]
    init_state = dfa.initial_state
    if isinstance(init_state, list):
        init_state = init_state[0]
    if init_state not in old_to_new:
        matched = next(
            (s for s in dfa.states
             if getattr(s, 'state_id', None) == getattr(init_state, 'state_id', None)),
            None
        )
        if matched:
            init_state = matched
        else:
            ghost = DfaState(
                getattr(init_state, 'state_id', 'q0'),
                is_accepting=getattr(init_state, 'is_accepting', False)
            )
            old_to_new[init_state] = ghost
    return Dfa(states=list(old_to_new.values()), initial_state=old_to_new[init_state])


def remove_unreachable(dfa: Dfa) -> Dfa:
    """Remove unreachable states from DFA in-place."""
    reachable = set()
    queue = [dfa.initial_state]
    while queue:
        s = queue.pop()
        if s not in reachable:
            reachable.add(s)
            for nxt in s.transitions.values():
                queue.append(nxt)
    dfa.states = [s for s in dfa.states if s in reachable]
    return dfa


def serialize_dfa(dfa: Dfa) -> int:
    """Return a hash that uniquely identifies a DFA's structure."""
    items = []
    for s in sorted(dfa.states, key=lambda x: str(x.state_id)):
        trans = sorted([(str(sym), str(dst.state_id))
                        for sym, dst in s.transitions.items()])
        items.append((str(s.state_id), s.is_accepting, tuple(trans)))
    return hash(tuple(items))


def check_path_accepted(dfa: Dfa, path) -> bool:
    """Return True if *path* is accepted by *dfa*."""
    cur = dfa.initial_state
    for sym in path:
        if sym not in cur.transitions:
            return False
        cur = cur.transitions[sym]
    return cur.is_accepting


def evaluate_dfa(dfa: Dfa, data, labels) -> float:
    """Accuracy of *dfa* on (data, labels)."""
    accepts = np.array([check_path_accepted(dfa, p) for p in data])
    correct = np.sum(accepts == (labels == 1))
    return float(correct) / len(labels) if len(labels) > 0 else 0.0


# ── Perturbation (mirrors DFASampler.perturbation in dfa_learner.py) ─────────

def perturbation(instance, alphabet, edit_distance, num_samples, seed=None):
    """
    Generate *num_samples* sequences by randomly perturbing *instance*.

    Operations: replace / insert / delete (same as DFASampler.perturbation).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    max_trials = num_samples * 10
    results = []
    trials = 0
    while len(results) < num_samples and trials < max_trials:
        trials += 1
        new_inst = list(instance)
        op = rng.choice(['replace', 'insert', 'delete'])
        max_edit = min(edit_distance, len(new_inst)) if len(new_inst) > 0 else 0
        edit_dist = rng.randint(0, max_edit) if max_edit > 0 else 0

        if op == 'replace' and len(new_inst) > 0:
            idxs = rng.sample(range(len(new_inst)), edit_dist)
            for idx in idxs:
                choices = [s for s in alphabet if s != new_inst[idx]]
                new_inst[idx] = rng.choice(choices) if choices else new_inst[idx]
        elif op == 'insert':
            for _ in range(edit_dist):
                insert_idx = rng.randint(0, len(new_inst))
                new_inst.insert(insert_idx, rng.choice(alphabet))
        elif op == 'delete' and len(new_inst) > 0:
            del_idxs = sorted(
                rng.sample(range(len(new_inst)), min(edit_dist, len(new_inst))),
                reverse=True
            )
            for idx in del_idxs:
                del new_inst[idx]

        results.append(tuple(new_inst))

    # Pad if needed
    if len(results) < num_samples and results:
        extra = rng.choices(results, k=num_samples - len(results))
        results.extend(extra)
    return [list(s) for s in results]


# ── Candidate DFA generation (mirrors DFALearner._propose_delete/merge) ──────

def _propose_delete(dfa: Dfa, data, labels, seen_sigs):
    """Propose candidate DFAs by deleting a single non-initial state."""
    candidates = []
    for s in list(dfa.states):
        if s is dfa.initial_state:
            continue
        # Preserve last accepting state
        if s.is_accepting and sum(x.is_accepting for x in dfa.states) <= 1:
            continue

        new_dfa = clone_dfa(dfa)
        target = next(x for x in new_dfa.states if x.state_id == s.state_id)
        outgoing = dict(target.transitions)

        for st in list(new_dfa.states):
            for sym, nxt in list(st.transitions.items()):
                if nxt is target:
                    st.transitions[sym] = outgoing.get(sym, st)

        if target in new_dfa.states:
            new_dfa.states.remove(target)

        # Fix dangling transitions
        state_set = set(new_dfa.states)
        for st in new_dfa.states:
            for sym, nxt in list(st.transitions.items()):
                if nxt not in state_set:
                    st.transitions[sym] = st

        remove_unreachable(new_dfa)

        if not any(st.is_accepting for st in new_dfa.states):
            del new_dfa
            continue

        sig = serialize_dfa(new_dfa)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            candidates.append(new_dfa)
        else:
            del new_dfa

    return candidates


def _collect_merge_pairs(dfa: Dfa, data, labels, max_pairs=20):
    """
    Collect state pairs to try merging.

    Strategy (mirrors collect_merge_pairs_simple in dfa_learner.py):
    - Compute per-state majority label from data.
    - Only merge states whose majority label matches.
    - Limit to max_pairs pairs.
    """
    state_label_dist = defaultdict(lambda: defaultdict(int))
    for path, lbl in zip(data, labels):
        cur = dfa.initial_state
        ok = True
        for sym in path:
            if sym not in cur.transitions:
                ok = False
                break
            cur = cur.transitions[sym]
        if ok:
            state_label_dist[cur.state_id][int(lbl)] += 1

    def main_label(sid):
        dist = state_label_dist.get(sid, {})
        if not dist:
            return None
        return max(dist, key=dist.get)

    states = list(dfa.states)
    pairs = []
    for s1, s2 in itertools.combinations(states, 2):
        l1 = main_label(s1.state_id)
        l2 = main_label(s2.state_id)
        if l1 is None or l2 is None or l1 != l2:
            continue
        pairs.append((s1, s2))
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def _propose_merge(dfa: Dfa, data, labels, seen_sigs, max_pairs=20):
    """Propose candidate DFAs by merging pairs of states."""
    candidates = []
    pairs = _collect_merge_pairs(dfa, data, labels, max_pairs=max_pairs)

    for s1, s2 in pairs:
        new_dfa = clone_dfa(dfa)
        s1n = next(x for x in new_dfa.states if x.state_id == s1.state_id)
        s2n = next(x for x in new_dfa.states if x.state_id == s2.state_id)

        for st in list(new_dfa.states):
            for sym, nxt in list(st.transitions.items()):
                if nxt is s2n:
                    st.transitions[sym] = s1n

        for sym, nxt in s2n.transitions.items():
            s1n.transitions[sym] = nxt

        s1n.is_accepting = s1n.is_accepting or s2n.is_accepting
        try:
            new_dfa.states.remove(s2n)
        except ValueError:
            pass

        remove_unreachable(new_dfa)

        if not any(st.is_accepting for st in new_dfa.states):
            del new_dfa
            continue

        sig = serialize_dfa(new_dfa)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            candidates.append(new_dfa)
        else:
            del new_dfa

    return candidates


def generate_neighbors(dfa: Dfa, data, labels, beam_size: int = 5):
    """
    Generate candidate neighbor DFAs from *dfa* using DELETE and MERGE.

    This mirrors the even-iteration branch of DFALearner.propose_automata
    (DELETE + MERGE) without depending on the external CXP module.
    """
    seen = {serialize_dfa(dfa)}
    candidates = []
    candidates += _propose_delete(dfa, data, labels, seen)
    candidates += _propose_merge(dfa, data, labels, seen, max_pairs=beam_size * 5)
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED INITIAL CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_shared_setup():
    """
    1. Perturb TEST_INSTANCE to generate labelled training samples.
    2. Run RPNI to learn the initial DFA.

    Returns
    -------
    data         : list of sequences (perturbation samples)
    labels       : np.ndarray of int labels (0/1)
    initial_dfa  : Dfa learned by RPNI
    """
    print("\n" + "=" * 70)
    print("SHARED SETUP: Perturbation samples + RPNI initial DFA")
    print("=" * 70)

    # ── 1. Perturbation ───────────────────────────────────────────────────────
    data = perturbation(TEST_INSTANCE, ALPHABET, EDIT_DISTANCE,
                        INIT_NUM_SAMPLES, seed=SEED)
    labels = predict_fn(data).astype(int)

    pos_count = int(np.sum(labels == 1))
    neg_count = int(np.sum(labels == 0))
    print(f"  Perturbation samples : {len(data)}  (pos={pos_count}, neg={neg_count})")

    # ── 2. RPNI initial DFA ───────────────────────────────────────────────────
    init_passive = []
    for path, lbl in zip(data, labels):
        init_passive.append([tuple(path), bool(lbl)])

    initial_dfa = run_RPNI(init_passive, automaton_type='dfa', print_info=False)
    n_states = len(initial_dfa.states)
    init_acc = evaluate_dfa(initial_dfa, data, labels)
    print(f"  RPNI initial DFA     : {n_states} states, accuracy={init_acc:.4f}")

    return data, labels, initial_dfa


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 1 – BEAM SEARCH  (the original method, re-implemented standalone)
# ══════════════════════════════════════════════════════════════════════════════

def run_beam_search(initial_dfa: Dfa, data, labels):
    """
    Greedy beam search over DFA candidates.

    At each iteration:
    - Expand current beam with generate_neighbors()
    - Keep top-K candidates scored by (accuracy, -num_states)

    This mirrors the logic in AnchorBaseBeam.anchor_beam but operates on the
    fixed perturbation data so the comparison is fair.

    NOTE: The original beam search in
          modified_modules/alibi/explainers/anchors/anchor_base.py
          is preserved and unchanged.
    """
    print("\n" + "─" * 70)
    print("BEAM SEARCH")
    print("─" * 70)
    t0 = time.time()

    beam = [clone_dfa(initial_dfa)]
    best_dfa = beam[0]
    best_acc = evaluate_dfa(best_dfa, data, labels)

    for iteration in range(MAX_ITERATIONS):
        candidates = []
        for dfa in beam:
            candidates += generate_neighbors(dfa, data, labels, BEAM_SIZE)

        if not candidates:
            print(f"  Iteration {iteration}: no candidates, stopping.")
            break

        scored = []
        for dfa in candidates:
            acc = evaluate_dfa(dfa, data, labels)
            scored.append((acc, -len(dfa.states), dfa))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        beam = [x[2] for x in scored[:BEAM_SIZE]]
        top_acc, top_neg_states, top_dfa = scored[0]
        print(f"  Iteration {iteration}: best_acc={top_acc:.4f}, states={-top_neg_states}")

        if top_acc > best_acc or (top_acc == best_acc
                                   and -top_neg_states < len(best_dfa.states)):
            best_acc = top_acc
            best_dfa = top_dfa

        if best_acc >= ACCURACY_THRESHOLD and len(best_dfa.states) <= STATE_THRESHOLD:
            print("  Early stop: accuracy and state thresholds met.")
            break

    return {
        'method': 'Beam Search',
        'accuracy': evaluate_dfa(best_dfa, data, labels),
        'states': len(best_dfa.states),
        'time_s': time.time() - t0,
        'dfa': best_dfa,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 2 – GENETIC ALGORITHM  (DEAP)
# ══════════════════════════════════════════════════════════════════════════════

def run_ga(initial_dfa: Dfa, data, labels):
    """
    Genetic Algorithm over DFA candidates using DEAP.

    Representation : each individual wraps a single DFA object.
    Fitness        : (accuracy, -num_states)  — maximise both.
    Mutation       : replace individual with a random neighbor via
                     generate_neighbors().
    Crossover      : swap the two parent DFAs (topology-safe swap).
    Selection      : tournament selection.
    """
    print("\n" + "─" * 70)
    print("GENETIC ALGORITHM (DEAP)")
    print("─" * 70)

    try:
        from deap import base, creator, tools
    except ImportError:
        print("  [SKIP] DEAP not installed. Run: pip install deap")
        return {'method': 'GA (DEAP)', 'accuracy': float('nan'),
                'states': float('nan'), 'time_s': 0.0, 'dfa': None}

    t0 = time.time()

    # Guard: DEAP creator classes are module-level singletons.
    for _cls in ('FitnessMaxDFA', 'IndividualDFA'):
        if hasattr(creator, _cls):
            delattr(creator, _cls)

    creator.create("FitnessMaxDFA", base.Fitness, weights=(1.0, 1.0))
    creator.create("IndividualDFA", list, fitness=creator.FitnessMaxDFA)

    toolbox = base.Toolbox()

    def _evaluate(ind):
        dfa = ind[0]
        acc = evaluate_dfa(dfa, data, labels)
        return acc, -len(dfa.states)   # maximise accuracy, minimise states

    def _mutate(ind):
        neighbors = generate_neighbors(ind[0], data, labels, BEAM_SIZE)
        if neighbors:
            ind[0] = random.choice(neighbors)
        if ind.fitness.valid:
            del ind.fitness.values
        return (ind,)

    def _crossover(ind1, ind2):
        ind1[0], ind2[0] = ind2[0], ind1[0]
        for ind in (ind1, ind2):
            if ind.fitness.valid:
                del ind.fitness.values
        return ind1, ind2

    toolbox.register("evaluate", _evaluate)
    toolbox.register("mate", _crossover)
    toolbox.register("mutate", _mutate)
    toolbox.register("select", tools.selTournament, tournsize=2)

    # Seed population: initial_dfa + its immediate neighbors.
    # Always include the initial DFA so the population starts from the same
    # point as the other algorithms.
    seed_candidates = generate_neighbors(initial_dfa, data, labels, BEAM_SIZE * 2)
    pop_size = max(BEAM_SIZE + 1, len(seed_candidates) + 1)
    population = [creator.IndividualDFA([clone_dfa(initial_dfa)])]   # elite seed
    for c in seed_candidates[:pop_size - 1]:
        population.append(creator.IndividualDFA([copy.deepcopy(c)]))
    while len(population) < pop_size:
        population.append(creator.IndividualDFA([clone_dfa(initial_dfa)]))

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    # Track global best starting from the initial DFA
    best_dfa = clone_dfa(initial_dfa)
    best_acc = evaluate_dfa(best_dfa, data, labels)

    CXPB, MUTPB = 0.3, 0.7
    for gen in range(MAX_ITERATIONS):
        # Elitism: carry over the single best individual unchanged
        elite = copy.deepcopy(max(population, key=lambda i: i.fitness.values))

        offspring = toolbox.select(population, len(population) - 1)
        offspring = [copy.deepcopy(ind) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring + [elite]

        best_ind = max(population, key=lambda i: i.fitness.values)
        gen_acc    = best_ind.fitness.values[0]
        gen_states = -int(best_ind.fitness.values[1])
        print(f"  Generation {gen}: best_acc={gen_acc:.4f}, states={gen_states}")

        if gen_acc > best_acc or (gen_acc == best_acc
                                   and gen_states < len(best_dfa.states)):
            best_acc = gen_acc
            best_dfa = best_ind[0]

        if best_acc >= ACCURACY_THRESHOLD and len(best_dfa.states) <= STATE_THRESHOLD:
            print("  Early stop: accuracy and state thresholds met.")
            break

    return {
        'method': 'GA (DEAP)',
        'accuracy': evaluate_dfa(best_dfa, data, labels),
        'states': len(best_dfa.states),
        'time_s': time.time() - t0,
        'dfa': best_dfa,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 3 – SIMULATED ANNEALING  (simanneal)
# ══════════════════════════════════════════════════════════════════════════════

def run_sa(initial_dfa: Dfa, data, labels):
    """
    Simulated Annealing using the simanneal library.

    State  : current DFA.
    Energy : (1 - accuracy) + state_count_penalty  (minimise).
    Move   : randomly pick one neighbor from generate_neighbors().
    """
    print("\n" + "─" * 70)
    print("SIMULATED ANNEALING (simanneal)")
    print("─" * 70)

    try:
        from simanneal import Annealer
    except ImportError:
        print("  [SKIP] simanneal not installed. Run: pip install simanneal")
        return {'method': 'SA (simanneal)', 'accuracy': float('nan'),
                'states': float('nan'), 'time_s': 0.0, 'dfa': None}

    t0 = time.time()
    _data = data
    _labels = labels

    class DFAAnnealer(Annealer):
        """SA problem: minimise 1-accuracy, penalise excess states."""

        def move(self):
            neighbors = generate_neighbors(self.state, _data, _labels, BEAM_SIZE)
            if neighbors:
                self.state = random.choice(neighbors)

        def energy(self):
            acc = evaluate_dfa(self.state, _data, _labels)
            state_penalty = max(0, len(self.state.states) - STATE_THRESHOLD) * 0.01
            return (1.0 - acc) + state_penalty

    annealer = DFAAnnealer(clone_dfa(initial_dfa))
    annealer.Tmax = 1.0
    annealer.Tmin = 0.001
    annealer.steps = MAX_ITERATIONS * 20
    annealer.updates = MAX_ITERATIONS

    best_dfa, _ = annealer.anneal()

    return {
        'method': 'SA (simanneal)',
        'accuracy': evaluate_dfa(best_dfa, data, labels),
        'states': len(best_dfa.states),
        'time_s': time.time() - t0,
        'dfa': best_dfa,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 4 – PARTICLE SWARM OPTIMIZATION  (pyswarms)
# ══════════════════════════════════════════════════════════════════════════════

def run_pso(initial_dfa: Dfa, data, labels):
    """
    Particle Swarm Optimization adapted for discrete DFA search.

    PSO is defined for continuous spaces; we adapt it using a shared pool:
    - Each particle holds an integer index into a growing pool of DFAs.
    - Position update: particles vote on the best pool index; pool is expanded
      each outer iteration by running generate_neighbors() on the current best.
    - PySwarms GlobalBestPSO optimises the pool index as a 1-D real value.
    """
    print("\n" + "─" * 70)
    print("PARTICLE SWARM OPTIMIZATION (pyswarms)")
    print("─" * 70)

    try:
        import pyswarms as ps
    except ImportError:
        print("  [SKIP] pyswarms not installed. Run: pip install pyswarms")
        return {'method': 'PSO (pyswarms)', 'accuracy': float('nan'),
                'states': float('nan'), 'time_s': 0.0, 'dfa': None}

    # Suppress pyswarms verbose output
    import logging
    logging.getLogger('pyswarms').setLevel(logging.ERROR)

    t0 = time.time()

    # ── Shared candidate pool ─────────────────────────────────────────────────
    pool = [clone_dfa(initial_dfa)]
    seen_sigs = {serialize_dfa(initial_dfa)}

    def _expand_pool(source_dfas):
        for dfa in source_dfas:
            for nbr in generate_neighbors(dfa, data, labels, BEAM_SIZE):
                sig = serialize_dfa(nbr)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    pool.append(nbr)

    _expand_pool([initial_dfa])

    # ── PSO fitness (cost to minimise) ────────────────────────────────────────
    def _pso_cost(positions):
        costs = np.zeros(positions.shape[0])
        for i, pos in enumerate(positions):
            idx = int(np.clip(round(pos[0]), 0, len(pool) - 1))
            acc = evaluate_dfa(pool[idx], data, labels)
            state_pen = max(0, len(pool[idx].states) - STATE_THRESHOLD) * 0.01
            costs[i] = (1.0 - acc) + state_pen
        return costs

    best_dfa = clone_dfa(initial_dfa)
    best_acc = evaluate_dfa(best_dfa, data, labels)

    n_particles = max(BEAM_SIZE, 3)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    for iteration in range(MAX_ITERATIONS):
        _expand_pool(pool[-n_particles:] if len(pool) > n_particles else pool)

        pool_size = len(pool)
        bounds = (np.zeros(1), np.full(1, float(pool_size - 1)))

        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=1,
            options=options,
            bounds=bounds,
        )
        _, best_pos = optimizer.optimize(_pso_cost, iters=10, verbose=False)

        best_idx = int(np.clip(round(best_pos[0]), 0, pool_size - 1))
        candidate = pool[best_idx]
        cand_acc = evaluate_dfa(candidate, data, labels)
        cand_states = len(candidate.states)
        print(f"  Iteration {iteration}: best_acc={cand_acc:.4f}, "
              f"states={cand_states}, pool_size={pool_size}")

        if cand_acc > best_acc or (cand_acc == best_acc
                                    and cand_states < len(best_dfa.states)):
            best_acc = cand_acc
            best_dfa = candidate

        if best_acc >= ACCURACY_THRESHOLD and len(best_dfa.states) <= STATE_THRESHOLD:
            print("  Early stop: accuracy and state thresholds met.")
            break

    return {
        'method': 'PSO (pyswarms)',
        'accuracy': evaluate_dfa(best_dfa, data, labels),
        'states': len(best_dfa.states),
        'time_s': time.time() - t0,
        'dfa': best_dfa,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log_path = os.path.join(OUTPUT_DIR, 'baseline_comparison.txt')
    log_file = None
    original_stdout = sys.stdout
    if _TEE_AVAILABLE:
        log_file = open(log_path, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_file)

    print("=" * 70)
    print("DFA BASELINE COMPARISON EXPERIMENT")
    print(f"Test instance  : {TEST_INSTANCE}")
    print(f"Alphabet       : {ALPHABET}")
    print(f"RPNI samples   : {INIT_NUM_SAMPLES}")
    print(f"Max iterations : {MAX_ITERATIONS}")
    print(f"Beam size      : {BEAM_SIZE}")
    print(f"Acc threshold  : {ACCURACY_THRESHOLD}")
    print(f"State threshold: {STATE_THRESHOLD}")
    print("=" * 70)

    # ── Shared setup ──────────────────────────────────────────────────────────
    data, labels, initial_dfa = build_shared_setup()
    init_acc = evaluate_dfa(initial_dfa, data, labels)
    init_states = len(initial_dfa.states)

    results = []

    # ── Run each algorithm ────────────────────────────────────────────────────
    results.append(run_beam_search(clone_dfa(initial_dfa), data, labels))
    results.append(run_ga(clone_dfa(initial_dfa), data, labels))
    results.append(run_sa(clone_dfa(initial_dfa), data, labels))
    results.append(run_pso(clone_dfa(initial_dfa), data, labels))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<25} {'Accuracy':>10} {'States':>8} {'Time(s)':>10}")
    print(f"{'Initial DFA (RPNI)':<25} {init_acc:>10.4f} {init_states:>8d}")
    print("─" * 70)
    for r in results:
        acc = r['accuracy']
        sts = r['states']
        t   = r['time_s']
        acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
        sts_str = f"{int(sts)}" if not np.isnan(sts) else "N/A"
        print(f"{r['method']:<25} {acc_str:>10} {sts_str:>8} {t:>10.2f}s")
    print("=" * 70)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, 'summary.csv')
    with open(csv_path, 'w') as f:
        f.write("method,accuracy,states,time_s\n")
        f.write(f"Initial DFA (RPNI),{init_acc:.6f},{init_states},0.0\n")
        for r in results:
            f.write(f"{r['method']},{r['accuracy']},{r['states']},{r['time_s']:.4f}\n")
    print(f"\nSummary saved to: {csv_path}")

    if log_file:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
