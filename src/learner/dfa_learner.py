"""
DFA (Deterministic Finite Automaton) Learner

This module contains:
1. DFA structure operations (from aalpy)
2. DFA manipulation functions (clone, merge, delete, etc.)
3. DFA learning algorithms
4. DFA visualization and export
"""
import gc
import importlib
import itertools
import random
import re
import sys
import os
from typing import Tuple, Dict, List, Set, Any
from collections import defaultdict, deque

from matplotlib import pyplot as plt
import numpy as np
from aalpy.automata.Dfa import Dfa, DfaState

# Add parent directory to path for scar_rpni import
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if _parent_path not in sys.path:
    sys.path.insert(0, _parent_path)

try:
    from scar_rpni_size_capped_demo import learn_dfa_size_capped
except ImportError:
    # Fallback: Define dummy function if module not available
    def learn_dfa_size_capped(*args, **kwargs):
        raise ImportError("scar_rpni_size_capped_demo module not found")
    learn_dfa_size_capped = None


# Add external_modules path for language module
_external_modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external_modules/Explaining-FA'))
if _external_modules_path not in sys.path:
    sys.path.insert(0, _external_modules_path)

# Try to import ExplainLanguage; if it fails (libmata not available),
# DELTA CXP analysis will return no blamed edge.
try:
    ExplainLanguage = importlib.import_module("language.explain").Language
    EXPLAIN_LANGUAGE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] ExplainLanguage not available at module import: {e}")
    EXPLAIN_LANGUAGE_AVAILABLE = False
    ExplainLanguage = None

# 匯入優化模組
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    from dfa_optimization import MergeOptimizer, _cxp_cache, PerformanceMonitor
    OPTIMIZATION_AVAILABLE = True
    print("[INFO] DFA 優化模組已加載")
except ImportError as e:
    print(f"[WARNING] dfa_optimization module not found: {e}, running without optimizations")
    OPTIMIZATION_AVAILABLE = False
    MergeOptimizer = None
    _cxp_cache = None

from .base import BaseAutomataLearner


# ==============================================================
# DFA Operations (moved from dfa_operation.py)
# ==============================================================

def _alphabet_of(dfa) -> Set[str]:
    """Get the alphabet of the dfa"""
    if hasattr(dfa, "alphabet"):
        return set(dfa.alphabet)
    if hasattr(dfa, "get_input_alphabet"):
        try:
            return set(dfa.get_input_alphabet())
        except Exception:
            pass
    syms = set()
    for s in getattr(dfa, "states", []):
        syms.update(getattr(s, "transitions", {}).keys())
    return syms


def dfa_product(dfa1: Dfa, dfa2: Dfa, final_func) -> Dfa:
    """
    Generic DFA product construction (for intersection/union).
    final_func: function that takes (accept1, accept2) -> bool (accepting)
    """
    alphabet = list(_alphabet_of(dfa1) | _alphabet_of(dfa2))
    state_map: Dict[Tuple[str, str], DfaState] = {}
    queue: list = []

    def get_id(s1, s2):
        return f"({s1.state_id},{s2.state_id})"

    initial_tuple = (dfa1.initial_state, dfa2.initial_state)
    initial_id = get_id(*initial_tuple)
    initial_accept = final_func(initial_tuple[0].is_accepting, initial_tuple[1].is_accepting)
    initial = DfaState(initial_id, initial_accept)
    state_map[(initial_tuple[0], initial_tuple[1])] = initial
    queue.append(initial_tuple)

    while queue:
        curr1, curr2 = queue.pop(0)
        curr = state_map[(curr1, curr2)]
        for a in alphabet:
            if a not in curr1.transitions or a not in curr2.transitions:
                continue
            next1 = curr1.transitions[a]
            next2 = curr2.transitions[a]
            next_tuple = (next1, next2)
            if next_tuple not in state_map:
                accept = final_func(next1.is_accepting, next2.is_accepting)
                node = DfaState(get_id(next1, next2), accept)
                state_map[next_tuple] = node
                queue.append(next_tuple)
            curr.transitions[a] = state_map[next_tuple]
    all_states = list(state_map.values())
    return Dfa(initial, all_states)


def dfa_intersection(dfa1: Dfa, dfa2: Dfa) -> Dfa:
    return dfa_product(dfa1, dfa2, lambda a1, a2: a1 and a2)


def dfa_union(dfa1: Dfa, dfa2: Dfa) -> Dfa:
    return dfa_product(dfa1, dfa2, lambda a1, a2: a1 or a2)


def get_base_dfa(learn_type, alphabet_map) -> Dfa:
    """Generate a DFA that sequentially moves from s0 to the end"""
    state_setup = {}
    steps = sorted(alphabet_map.keys())
    num_states = len(steps)

    for i, step in enumerate(steps):
        state_name = f's{i}'
        next_state = f's{i+1}'
        is_accepting = (i == num_states)
        transitions = {symbol: next_state for symbol in alphabet_map[step]}
        state_setup[state_name] = (is_accepting, transitions)

    final_state = f's{num_states}'
    state_setup[final_state] = (True, {})
    return Dfa.from_state_setup(state_setup)


def merge_linear_edges(dfa, learn_type=None):
    """
    Merge consecutive edges in the dfa.
    For TEXT type: if two consecutive edges are both wildcards (*), merge them.
    """
    def is_wildcard(sym):
        if sym == "*":
            return True
        if isinstance(sym, str) and re.match(r"p\d+_\(\*\)", sym):
            return True
        return False
    
    def get_position(sym):
        if not isinstance(sym, str):
            return None
        match = re.match(r"p(\d+)_", sym)
        return int(match.group(1)) if match else None
    
    states = set(dfa.states)
    changed = True
    while changed:
        changed = False
        for s in list(states):
            if s is dfa.initial_state or s.is_accepting:
                continue

            in_edges = []
            for p in states:
                for sym, q in p.transitions.items():
                    if q is s:
                        in_edges.append((p, sym))
                        if len(in_edges) > 1:
                            break
                if len(in_edges) > 1:
                    break

            out_edges = list(s.transitions.items())
            if len(in_edges) == 1 and len(out_edges) == 1:
                p, sym_in = in_edges[0] 
                sym_out, q = out_edges[0]

                if sym_in == sym_out:
                    p.transitions[sym_in] = q
                    states.remove(s)
                    dfa.states.remove(s)
                    changed = True
                    break
                
                if learn_type == "Text" and is_wildcard(sym_in) and is_wildcard(sym_out):
                    pos_in = get_position(sym_in)
                    pos_out = get_position(sym_out)
                    
                    if pos_in is not None and pos_out is not None and pos_out == pos_in + 1:
                        merged_label = f"p{pos_in}-{pos_out}_(*)"
                        del p.transitions[sym_in]
                        p.transitions[merged_label] = q
                        states.remove(s)
                        dfa.states.remove(s)
                        changed = True
                        break
                    
                    range_match = re.match(r"p(\d+)-(\d+)_\(\*\)", sym_in)
                    if range_match and pos_out is not None:
                        start_pos = int(range_match.group(1))
                        end_pos = int(range_match.group(2))
                        if pos_out == end_pos + 1:
                            merged_label = f"p{start_pos}-{pos_out}_(*)"
                            del p.transitions[sym_in]
                            p.transitions[merged_label] = q
                            states.remove(s)
                            dfa.states.remove(s)
                            changed = True
                            break
    return dfa


def merge_parallel_edges(dfa, learn_type):
    """Merge parallel edges in the dfa."""
    def clean_symbol(sym: str) -> str:
        if learn_type in ("Text", "Image"):
            match = re.search(r"\(([\d\.\-]+)\)", sym)
            if match:
                return match.group(1)
            return sym
        return sym

    def get_prefix(sym):
        if not isinstance(sym, str):
            return ""
        match = re.match(r"(p\d+)_\([\d\.\-\w]+\)", sym)
        return match.group(1) + "_" if match else ""

    def get_position(sym):
        if not isinstance(sym, str):
            return None
        match = re.match(r"p(\d+)_", sym)
        return match.group(1) if match else None

    alphabet = set(clean_symbol(sym) for sym in _alphabet_of(dfa))

    for s in dfa.states:
        target_map = {}
        new_transitions = {}
        prefix_map = {}
        position_map = {}

        for sym, t in s.transitions.items():
            clean_sym = clean_symbol(sym)
            new_transitions[clean_sym] = t
            if t != s:
                target_map.setdefault(t, set()).add(clean_sym)
                prefix = get_prefix(sym)
                if prefix:
                    prefix_map[t] = prefix
                
                if learn_type == "Text":
                    pos = get_position(sym)
                    if pos:
                        key = (t, pos)
                        position_map.setdefault(key, set()).add(sym)

        if learn_type == "Text":
            for (t, pos), symbols in position_map.items():
                if len(symbols) >= 2:
                    has_unk = any("UNK" in sym for sym in symbols)
                    has_word = any("UNK" not in sym for sym in symbols)
                    
                    if has_unk and has_word:
                        prefix = prefix_map.get(t, "")
                        wildcard = f"{prefix}(*)" if prefix else "*"
                        
                        for sym in symbols:
                            if sym in new_transitions:
                                del new_transitions[sym]
                        
                        new_transitions[wildcard] = t
            
            s.transitions = new_transitions
        else:
            for t, symbols in target_map.items():
                if symbols == alphabet:
                    s.transitions = new_transitions.copy()
                    for sym in list(symbols):
                        if sym in s.transitions and s.transitions[sym] is t:
                            del s.transitions[sym]

                    prefix = prefix_map.get(t, "")
                    if prefix:
                        s.transitions[f"{prefix}(*)"] = t
                    else:
                        s.transitions[f"*"] = t
    
    return dfa


def simplify_dfa(dfa, learn_type):
    """Simplify dfa by merging edges"""
    dfa = merge_parallel_edges(dfa, learn_type)
    dfa = merge_linear_edges(dfa, learn_type)
    return dfa


def scar_to_aalpy_dfa(scar_dfa) -> Dfa:
    """Convert scar_rpni_size_capped_demo.DFA to aalpy.automata.Dfa"""
    id2state = {}
    for q in sorted(scar_dfa.states, key=lambda x: str(x)):
        id2state[q] = DfaState(f"s{q}", q in scar_dfa.accepting)

    initial = id2state[scar_dfa.start]

    for (q, a), r in scar_dfa.delta.items():
        id2state[q].transitions[a] = id2state[r]

    return Dfa(initial, list(id2state.values()))


def dfa_intersection_any(d1, d2):
    """Intersection that handles both aalpy and scar DFAs"""
    if not isinstance(d1, Dfa):
        d1 = scar_to_aalpy_dfa(d1)
    if not isinstance(d2, Dfa):
        d2 = scar_to_aalpy_dfa(d2)
    return dfa_intersection(d1, d2)


def clone_dfa(dfa: Dfa) -> Dfa:
    """Deep clone a DFA"""
    old_to_new = {}

    for old_s in dfa.states:
        s_new = DfaState(old_s.state_id, is_accepting=old_s.is_accepting)
        s_new.prefix = []
        old_to_new[old_s] = s_new

    for old_s in dfa.states:
        for sym, tgt in old_s.transitions.items():
            if tgt not in old_to_new:
                ghost = DfaState(getattr(tgt, "state_id", str(tgt)), is_accepting=getattr(tgt, "is_accepting", False))
                ghost.prefix = []
                old_to_new[tgt] = ghost
            old_to_new[old_s].transitions[sym] = old_to_new[tgt]

    init_state = dfa.initial_state
    if isinstance(init_state, list):
        init_state = init_state[0]
    if init_state not in old_to_new:
        matched = next(
            (s for s in dfa.states if getattr(s, "state_id", None) == getattr(init_state, "state_id", None)),
            None
        )
        if matched:
            init_state = matched
        else:
            ghost = DfaState(getattr(init_state, "state_id", "q0"),
                            is_accepting=getattr(init_state, "is_accepting", False))
            old_to_new[init_state] = ghost

    return Dfa(states=list(old_to_new.values()), initial_state=old_to_new[init_state])


def remove_unreachable_states(dfa):
    """Remove unreachable states from DFA"""
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


def serialize_dfa(dfa) -> int:
    """Serialize DFA to hashable signature for deduplication"""
    items = []
    for s in sorted(dfa.states, key=lambda x: str(x.state_id)):
        trans = sorted([(sym, str(dst.state_id)) for sym, dst in s.transitions.items()])
        items.append((str(s.state_id), s.is_accepting, tuple(trans)))
    return hash(tuple(items))


def make_dfa_complete(dfa: Dfa, alphabet: list) -> Dfa:
    """Convert Partial DFA to Complete DFA by adding sink state"""
    sink_id = "sink_state"
    existing_ids = set(s.state_id for s in dfa.states)
    while sink_id in existing_ids:
        sink_id += "_"

    sink_state = DfaState(sink_id, is_accepting=False)
    
    for sym in alphabet:
        sink_state.transitions[sym] = sink_state

    added_sink = False
    for s in dfa.states:
        for sym in alphabet:
            if sym not in s.transitions:
                s.transitions[sym] = sink_state
                added_sink = True
    
    if added_sink:
        dfa.states.append(sink_state)
        
    return dfa


def trim_dfa(dfa):
    """Trim unreachable states from DFA"""
    alphabet = _alphabet_of(dfa)
    start = dfa.initial_state

    reachable = {start}
    queue = [start]

    while queue:
        cur = queue.pop(0)
        for a in alphabet:
            if a in cur.transitions:
                nxt = cur.transitions[a]
                if nxt not in reachable:
                    reachable.add(nxt)
                    queue.append(nxt)

    state_map = {}
    for old in reachable:
        new_state = DfaState(old.state_id)
        new_state.is_accepting = old.is_accepting
        new_state.transitions = {}
        state_map[old] = new_state
    
    for old, new in state_map.items():
        for a in alphabet:
            if a in old.transitions:
                nxt = old.transitions[a]
                if nxt in state_map:
                    new.transitions[a] = state_map[nxt]

    return Dfa(
        states=set(state_map.values()),
        initial_state=state_map[start],
    )

# ==============================================================
# DFA Visualization and Export
# ==============================================================
def dfa_to_mata(dfa, file_path):
    """Export DFA to Mata format for libMata - produces valid NFA-explicit format"""
    def clean_state_name(name, index):
        """Create a valid mata identifier: q + index (stable mapping)"""
        # Use indexed names: q0, q1, q2, etc. for stability and libMata compatibility
        return f"q{index}"

    try:
        raw_states = list(getattr(dfa, 'states', []))
        init_state = getattr(dfa, 'initial_state', None)
        if isinstance(init_state, (list, tuple)):
            init_state = init_state[0] if init_state else None

        # Build a robust state pool using states + transition targets + initial state.
        state_by_id = {}
        for st in raw_states:
            sid = getattr(st, 'state_id', None)
            if sid is not None and sid not in state_by_id:
                state_by_id[sid] = st

        if init_state is not None:
            init_id = getattr(init_state, 'state_id', None)
            if init_id is not None and init_id not in state_by_id:
                state_by_id[init_id] = init_state

        for st in list(state_by_id.values()):
            for _, tgt in getattr(st, 'transitions', {}).items():
                tid = getattr(tgt, 'state_id', None)
                if tid is not None and tid not in state_by_id:
                    state_by_id[tid] = tgt

        # Last-resort synthetic single state if DFA object is severely malformed.
        if not state_by_id:
            synthetic = DfaState('q_synth', is_accepting=True)
            synthetic.transitions = {}
            state_by_id['q_synth'] = synthetic
            init_state = synthetic

        # Ensure we have a valid initial state that exists in state_by_id.
        if init_state is None or getattr(init_state, 'state_id', None) not in state_by_id:
            init_state = next(iter(state_by_id.values()))

        states = list(state_by_id.values())
        finals = [s for s in states if getattr(s, 'is_accepting', False)]
        if not finals:
            # Keep export valid even for intermediate malformed DFAs.
            init_state.is_accepting = True
            finals = [init_state]

        all_syms = sorted({sym for s in states for sym in getattr(s, 'transitions', {}).keys()})
        symbol_map = {sym: i for i, sym in enumerate(all_syms)}
        # Map each state to q0, q1, q2, etc.
        state_list = sorted(states, key=lambda s: str(getattr(s, 'state_id', '')))
        state_map = {s.state_id: clean_state_name(s.state_id, i) for i, s in enumerate(state_list)}
        transition_map = {}

        # Build mata content
        mata_lines = ["@NFA-explicit"]
        init_name = state_map[init_state.state_id]
        mata_lines.append(f"%Initial {init_name}")
        # print(f"  [MATA] Initial state: {init_name}")

        finals_str = " ".join(state_map[s.state_id] for s in finals)
        mata_lines.append(f"%Final {finals_str}")
        # print(f"  [MATA] Final states: {finals_str}")
        # print(f"  [MATA] Symbol mapping: {symbol_map}")

        transition_count = 0
        for s in states:
            src = state_map[s.state_id]
            for sym, tgt in sorted(getattr(s, 'transitions', {}).items(), key=lambda x: str(x[0])):  # Sort for consistency
                tgt_id = getattr(tgt, 'state_id', None)
                if tgt_id is None or tgt_id not in state_map:
                    continue
                tgt_name = state_map[tgt_id]
                sym_id = symbol_map[sym]
                mata_lines.append(f"{src} {sym_id} {tgt_name}")
                transition_map[transition_count] = sym
                transition_count += 1
        
        # Write mata file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mata_lines) + "\n")  # Ensure final newline
        
        print(f"  [MATA] Wrote {transition_count} transitions to {file_path}")
        
        # Print mata file content for debugging
        # print(f"  [MATA] File content:")
        # for line in mata_lines:
        #     print(f"    {line}")
        
        # Create reverse mapping: cleaned_name -> original_id
        reverse_state_map = {v: k for k, v in state_map.items()}
        return state_map, symbol_map, transition_map, reverse_state_map
    
    except Exception as e:
        # Last-resort fallback: always write a minimal valid automaton file.
        # This prevents DELTA from crashing when intermediate DFAs are malformed.
        try:
            fallback_lines = [
                "@NFA-explicit",
                "%Initial q0",
                "%Final q0",
            ]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(fallback_lines) + "\n")
            print(f"  [MATA] WARNING: export fallback used due to error: {e}")
            return ({'q0': 'q0'}, {}, {}, {'q0': 'q0'})
        except Exception as inner_e:
            raise RuntimeError(f"Failed to export DFA to mata format: {e}; fallback failed: {inner_e}")


def explain_axp_cxp(axps, cxps, symbol_map):
    """Print human-readable AXp and CXp explanations"""
    inv_map = {v: k for k, v in symbol_map.items()}
    for i, axp in enumerate(axps):
        print(f"AXp {i+1}: {[inv_map[x] for x in axp]}")
    for i, cxp in enumerate(cxps):
        print(f"Cxp {i+1}: {[inv_map[x] for x in cxp]}")


def get_test_word(mata_path, symbol_map):
    """Generate a test word from symbol map"""
    grouped = defaultdict(list)
    path = []

    for k, v in symbol_map.items():
        match = re.match(r"(p\d+)_", k)
        if not match:
            continue
        pos = match.group(1)
        grouped[pos].append((k, v))

    for pos in sorted(grouped.keys(), key=lambda x: int(x[1:])): 
        symbols = grouped[pos]
        for k, v in symbols:
            if "(1.0)" in k or "*" in k:
                path.append((pos, v))
                break 

    path_ids = [v for _, v in path]
    # print("Path:", path_ids)
    return path_ids


# ==============================================================
# DFA Learner Class
# ==============================================================
class DFASampler:
    """
    DFA-aware sampler using perturbation specific to DFA search.
    
    Supports two teacher types:
    1. Automata DFA: initial_dfa from DOT file (ground truth for explanation)
    2. RNN classifier: predictor is a trained neural network (for real-world noisy data)
    
    When initial_dfa is provided:
    - Used as ground-truth oracle for label generation during perturbation
    - Not modified by the search algorithm
    - Provides 100% accurate labels for the automata regular language
    
    When predictor is provided (real-world data):
    - Used as teacher/oracle for label generation
    - Trained RNN classifier for real-world or other datasets
    """

    def __init__(self, predictor=None, base_sampler=None, alphabet: List = [], seed: int = None, 
                 edit_distance: int = 1, data_type: str = "automata", initial_dfa=None):
        self.predictor = predictor
        self.initial_dfa = initial_dfa  # Ground-truth DFA for automata regular languages
        self.tab_sampler = base_sampler
        self.alphabet = alphabet
        self.edit_distance = edit_distance
        self.data_type = data_type  # 'automata' or 'real_world'

        self.instance_label = None
        self.instance = None
        self.n_covered_ex = 10
        self.task_type = "regular" if initial_dfa is not None else "realworld"
        self._built = True
        
        if seed is not None:
            random.seed(seed)
        else:
            self.seed = seed

    def set_instance_label(self, X):
        """
        Set the target instance for DFA-based perturbation.

        This version does not require TabularSampler.
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()

        self.instance = list(X)

        if self.tab_sampler is not None:
            self.tab_sampler.set_instance_label(X)
            self.instance_label = self.tab_sampler.instance_label
        else:
            self.instance_label = int(self.predictor([self.instance])[0])

    def set_n_covered(self, n):
        self.n_covered_ex = n

        if self.tab_sampler is not None:
            self.tab_sampler.set_n_covered(n)

    def build_lookups(self, data=None):
        """AnchorTabular 預期 sampler 有 build_lookups"""
        self._built = True
        return ({}, {}, {})

    def compare_labels(self, samples):

        preds = self.predictor(samples)

        # ==========================================================
        # Regular language:
        # labels = teacher DFA accept/reject
        # ==========================================================
        if self.task_type == "regular":
            return np.asarray(preds, dtype=int)

        # ==========================================================
        # Real-world:
        # labels = agreement with instance prediction
        # ==========================================================
        return np.asarray(preds == self.instance_label, dtype=int)
    
    def perturbation(self, num_samples: int):
        """
        DFASampler perturbation - uses shared DFALearner._generate_perturbed_samples
        
        Generates unique perturbed samples with intelligent deduplication.
        
        ----------
        num_samples : int
            Target number of unique samples to generate
            
        Returns
        -------
        tuple
            (local_paths, d_samples) - both are lists of unique perturbed sequences
        """
        symbols = self.alphabet if self.alphabet else list(set(self.instance))
        edit_distance = self.edit_distance
        
        local_paths_set = set()  # Use set for automatic deduplication
        no_progress_count = 0  # Track consecutive failures
        max_no_progress = 50  # If no new samples in 100 tries, give up
        
        trials = 0
        max_trials = 10000
        
        while len(local_paths_set) < int(num_samples) and trials < max_trials:
            trials += 1
            
            new_instance = list(self.instance)
            op = random.choice(["replace", "insert", "delete"])
            max_edit = min(edit_distance, len(new_instance))
            edit_dist = random.randint(0, max_edit) if max_edit > 0 else 0

            if op == "replace":
                if len(new_instance) > 0 and edit_dist > 0:
                    replace_indices = random.sample(range(len(new_instance)), min(edit_dist, len(new_instance)))
                    for idx in replace_indices:
                        different_symbols = [s for s in symbols if s != new_instance[idx]]
                        if different_symbols:
                            new_instance[idx] = random.choice(different_symbols)

            elif op == "insert":
                for _ in range(edit_dist):
                    insert_idx = random.randint(0, len(new_instance))
                    new_instance.insert(insert_idx, random.choice(symbols))

            elif op == "delete":
                if len(new_instance) > 0 and edit_dist > 0:
                    delete_count = min(edit_dist, len(new_instance))
                    delete_indices = sorted(random.sample(range(len(new_instance)), delete_count), reverse=True)
                    for idx in delete_indices:
                        del new_instance[idx]

            # Convert to hashable tuple for deduplication
            hashable_instance = tuple(new_instance)
            
            # Track progress for early exit
            prev_size = len(local_paths_set)
            local_paths_set.add(hashable_instance)
            
            if len(local_paths_set) == prev_size:
                no_progress_count += 1
            else:
                no_progress_count = 0
            
            # Early exit if no progress for too long
            if no_progress_count > max_no_progress:
                break

        local_paths = [list(seq) for seq in local_paths_set]
        return local_paths, local_paths
        
    def __call__(self, num_samples, compute_labels=True):
        if self.instance is None:
            raise ValueError("DFASampler instance is not set. Call set_instance_label(X) first.")

        raw_data, d_raw_data = self.perturbation(num_samples)

        # Generate labels by comparing predictions with instance_label:
        # - regular (automata): check_path_accepted result compared with instance_label
        # - non-regular: predictor result compared with instance_label
        if compute_labels:
            if self.instance_label is None:
                self.instance_label = int(self.predictor([self.instance])[0])

            labels = self.compare_labels(raw_data)
            return [raw_data, labels.astype(int)]

        return [d_raw_data]
        
class DFALearner(BaseAutomataLearner):
    """DFA (Deterministic Finite Automaton) Learner"""
    
    def __init__(self,
                predictor=None,
                base_sampler=None,
                alphabet: List = [],
                seed: int = None,
                edit_distance: int = 1,
                data_type: str = "automata",
                initial_dfa=None,
            ):
        self.predictor = predictor
        self.initial_dfa = initial_dfa
        self.tab_sampler = base_sampler
        self.alphabet = alphabet
        self.edit_distance = edit_distance
        self.data_type = data_type

        self.instance_label = None
        self.instance = None
        self.n_covered_ex = 10

        # regular: teacher DFA label = accept/reject
        # realworld: label = whether predictor output agrees with original instance
        self.task_type = "regular" if initial_dfa is not None or data_type == "automata" else "realworld"

        self._built = True
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    # ========== Implement Abstract Methods ==========
    def get_sampler(self):
        from learner.dfa_learner import DFASampler
        return DFASampler

    def check_path_accepted(self, dfa, path) -> bool:
        """Check if path is accepted by DFA"""
        dfa = dfa[0] if isinstance(dfa, (list, tuple)) else dfa
        dfa.reset_to_initial()
        for symbol in path:
            try:
                dfa.step(symbol)
            except KeyError:
                return False
        return dfa.current_state.is_accepting
    
    def check_path_exist(self, dfa, path) -> bool:
        """Check if path exists in DFA"""
        dfa = dfa[0] if isinstance(dfa, (list, tuple)) else dfa
        dfa.current_state = dfa.initial_state
        for symbol in path:
            try:
                dfa.step(symbol)
            except KeyError:
                return False
        return True
    
    def get_accept_paths(self, dfa, max_depth=50) -> List[List[Any]]:
        """Find all accepting paths of the DFA using DFS"""
        dfa_obj = dfa[0] if isinstance(dfa, (list, tuple)) else dfa
        paths = set() 

        def dfs(state, prefix, visited):
            if len(prefix) > max_depth:
                return
            if getattr(state, "is_accepting", False):
                paths.add(tuple(prefix))
            for sym, next_state in state.transitions.items():
                if (id(next_state), sym) not in visited:
                    dfs(next_state, prefix + [sym], visited | {(id(next_state), sym)})

        dfs(dfa_obj.initial_state, [], set())
        return [list(p) for p in sorted(paths, key=len)]
    
    # ========== Evaluation Utilities (Shared with search_baselines, pso_optimizer) ==========
    
    def compute_accuracy(self, dfa, data: list, labels: np.ndarray) -> float:
        """
        Compute training accuracy of a DFA on labeled data.
        
        Parameters
        ----------
        dfa : Dfa
            DFA to evaluate
        data : list
            List of input sequences
        labels : np.ndarray
            Binary labels (1=accept, 0=reject)
            
        Returns
        -------
        float
            Accuracy in [0, 1]
        """
        if len(data) == 0:
            return 0.0
        accepts = np.array([self.check_path_accepted(dfa, p) for p in data])
        lbl = np.asarray(labels)
        correct = int(np.sum((lbl == 1) & accepts) + np.sum((lbl == 0) & ~accepts))
        return correct / len(lbl)
    
    def is_valid_dfa(self, dfa) -> bool:
        """
        Check if DFA is valid (has >= 2 states and at least one accepting state).
        
        Parameters
        ----------
        dfa : Dfa
            DFA to validate
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        if not hasattr(dfa, 'states'):
            return False
        states = list(dfa.states)
        return len(states) >= 2 and any(s.is_accepting for s in states)
    
    def add_to_history(self, all_history: list, seen_ids: set, dfa, 
                       training_accuracy: float, validation_accuracy: float = None,
                       use_automata_key: bool = False) -> None:
        """
        Add a valid, not-yet-seen DFA to history (deduplication).
        
        Parameters
        ----------
        all_history : list
            History list to append to
        seen_ids : set
            Set of already-seen DFA signatures (by automaton structure)
        dfa : Dfa
            DFA to add
        training_accuracy : float
            Training accuracy of the DFA
        validation_accuracy : float, optional
            Validation accuracy (default: None)
        use_automata_key : bool
            If True, use 'automata' key in dict (for PSO); 
            if False, use 'dfa' key (for search_baselines)
        """
        try:
            dfa_sig = self.serialize_automaton(dfa)
        except Exception:
            # Fallback for unexpected/invalid objects to keep history robust.
            dfa_sig = id(dfa)

        if dfa_sig in seen_ids:
            return
        if not self.is_valid_dfa(dfa):
            return
        
        seen_ids.add(dfa_sig)
        
        # Use appropriate key based on caller preference
        dfa_key = 'automata' if use_automata_key else 'dfa'
        record = {
            dfa_key: dfa,
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy,
            'states': len(dfa.states),
        }
        all_history.append(record)
    
    def clone_automaton(self, dfa):
        """Deep clone a DFA"""
        old_to_new = {}

        for old_s in dfa.states:
            s_new = DfaState(old_s.state_id, is_accepting=old_s.is_accepting)
            s_new.prefix = []
            old_to_new[old_s] = s_new

        for old_s in dfa.states:
            for sym, tgt in old_s.transitions.items():
                if tgt not in old_to_new:
                    ghost = DfaState(getattr(tgt, "state_id", str(tgt)), is_accepting=getattr(tgt, "is_accepting", False))
                    ghost.prefix = []
                    old_to_new[tgt] = ghost
                old_to_new[old_s].transitions[sym] = old_to_new[tgt]

        init_state = dfa.initial_state
        if isinstance(init_state, list):
            init_state = init_state[0]
        if init_state not in old_to_new:
            matched = next(
                (s for s in dfa.states if getattr(s, "state_id", None) == getattr(init_state, "state_id", None)),
                None
            )
            if matched:
                init_state = matched
            else:
                ghost = DfaState(getattr(init_state, "state_id", "q0"),
                                is_accepting=getattr(init_state, "is_accepting", False))
                old_to_new[init_state] = ghost

        return Dfa(states=list(old_to_new.values()), initial_state=old_to_new[init_state])
    
    def serialize_automaton(self, dfa) -> int:
        """Serialize DFA to hashable signature for deduplication"""
        items = []
        for s in sorted(dfa.states, key=lambda x: str(x.state_id)):
            trans = sorted([(str(sym), str(dst.state_id)) for sym, dst in s.transitions.items()])
            items.append((str(s.state_id), s.is_accepting, tuple(trans)))
        return hash(tuple(items))
    
    def automaton_to_graphviz(self, dfa, filename=None, show_sink=False, instance=None, output_dir="output") -> str:
        """
        Convert DFA to Graphviz DOT string and save visualization.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to visualize
        filename : str
            Output filename
        show_sink : bool
            Whether to show sink states
        instance : list or tuple, optional
            Original instance to highlight its path in the DFA with color
        output_dir : str
            Output directory
            
        Returns
        -------
        str
            DOT content string
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        def clean_state_name(name):
            name = str(name)
            return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name)

        def clean_label(label):
            return str(label).replace('"', "'")

        # draw path edges if instance is provided
        path_edges = set()
        if instance is not None:
            current = dfa.initial_state
            for i, symbol in enumerate(instance):
                if symbol in current.transitions:
                    next_state = current.transitions[symbol]
                    path_edges.add((clean_state_name(current.state_id), clean_state_name(next_state.state_id), clean_label(symbol)))
                    current = next_state
                else:
                    break

        lines = ["digraph DFA {", "  rankdir=LR;", '  node [shape=circle];']
        start_name = clean_state_name(dfa.initial_state.state_id)
        lines.append('  __start__ [shape=point];')
        lines.append(f'  __start__ -> "{start_name}";')

        for state in dfa.states:
            if not show_sink and hasattr(dfa, "sink") and state == dfa.sink:
                continue
            shape = "doublecircle" if state.is_accepting else "circle"
            lines.append(f'  "{clean_state_name(state.state_id)}" [shape={shape}];')

        for state in dfa.states:
            for symbol, next_state in state.transitions.items():
                if not show_sink and hasattr(dfa, "sink") and (
                    state == dfa.sink or next_state == dfa.sink
                ):
                    continue
                src_name = clean_state_name(state.state_id)
                dst_name = clean_state_name(next_state.state_id)
                label_name = clean_label(symbol)
                
                if (src_name, dst_name, label_name) in path_edges:
                    lines.append(f'  "{src_name}" -> "{dst_name}" [label="{label_name}", color=red, penwidth=2.5, fontcolor=red];')
                else:
                    lines.append(f'  "{src_name}" -> "{dst_name}" [label="{label_name}"];')

        lines.append("}")
        
        # Save to file
        dot_content = "\n".join(lines)
        if filename:
            dot_path = os.path.join(output_dir, filename)
            with open(dot_path, "w") as f:
                f.write(dot_content)

        return dot_content


    def create_automata_sized(self, positive_samples, negative_samples, alphabet):
        """Create DFA with size-capped learning"""
        if learn_dfa_size_capped is None:
            raise ImportError(
                "scar_rpni_size_capped_demo module not found. "
                "Please ensure src/scar_rpni_size_capped_demo.py exists in the project."
            )
        
        print(f'\nPassive learning sample count: {len(positive_samples + negative_samples)}\n')

        results = []
        Ms = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        for M in Ms:
            out = learn_dfa_size_capped(
                positive_samples,
                negative_samples,
                alphabet,
                M,
                include_sink_in_count=False,
                verbose=True,
                beam_enabled=True
            )
            states = out["total_states_including_sink"] if False else out["non_sink_states"]
            results.append((M, states, out["train_accuracy"]))
            print(f"M={M:>3} | states={states:>3} | train_acc={out['train_accuracy']:.3f}")
            
        Ms_list = [m for m, _, _ in results]
        accs = [acc for _, _, acc in results]
        plt.figure()
        plt.plot(Ms_list, accs, marker='o')
        plt.xlabel("Size cap M (non-sink states)" if not False else "Size cap M (including sink)")
        plt.ylabel("Training accuracy")
        plt.title("SCAR-RPNI: accuracy vs. size cap")
        plt.grid(True)
        plt.show()
        return out['dfa']
    
    def create_init_automata(self, data_type, positive_samples, negative_samples):
        """Create initial DFA using RPNI algorithm"""
        from aalpy.learning_algs import run_RPNI
        init_passive_data = []
        
        if data_type != 'Tabular':
            sample_source = positive_samples if positive_samples else negative_samples
            text_length = len(sample_source[0]) if sample_source else 0
            alphabet_map = {}
            
            all_samples = positive_samples + negative_samples
            for pos in range(text_length):
                symbols_at_pos = set()
                for sample in all_samples:
                    if pos < len(sample):
                        symbols_at_pos.add(sample[pos])
                alphabet_map[pos] = list(symbols_at_pos)
        
        for i in positive_samples:
            init_passive_data.append([tuple(i), True])
        for i in negative_samples:
            init_passive_data.append([tuple(i), False])
        dfa = run_RPNI(init_passive_data, automaton_type='dfa', print_info=False)
        # print("dfa", dfa)
        return dfa
    
    def perturbation(self, num_samples, max_trials=None):
        """
        Randomly perturb sequences to generate initial passive samples.
        
        Strategies:
        - replace: replace n symbols with different ones
        - append: insert n new symbols at random positions
        - delete: delete n symbols
        
        Uses adaptive max_trials based on sequence length and required samples:
        - If max_trials is None, compute automatically
        - At least 500 trials or 5x num_samples attempts
        
        Parameters
        ----------
        num_samples : int
            Target number of unique samples to generate
        max_trials : int, optional
            Maximum number of generation attempts. If None, computed adaptively
            
        Returns
        -------
        list
            List of unique perturbed sequences (may be < num_samples if diversity is limited)
        """
        if max_trials is None:
            # Adaptive max_trials based on sequence length and target samples
            seq_len = len(self.instance) if self.instance else 1
            max_trials = max(500, num_samples * 5, seq_len * num_samples)
        
        perturbed = set()  # Automatic deduplication
        trials = 0
        no_progress_count = 0
        max_no_progress = 50  # Stop if no new samples for 50 consecutive tries

        while len(perturbed) < num_samples and trials < max_trials:
            new_instance = self.instance.copy()
            op = random.choice(["replace", "append", "delete"])
            edit_dist = min(self.edit_distance, len(new_instance))  # Ensure not exceeding sequence length

            if op == "replace" and len(new_instance) > 0 and edit_dist > 0:
                try:
                    replace_count = min(edit_dist, len(new_instance))
                    replace_indices = random.sample(range(len(new_instance)), replace_count)
                    for idx in replace_indices:
                        different_symbols = [s for s in self.alphabet if s != new_instance[idx]]
                        if different_symbols:
                            new_instance[idx] = random.choice(different_symbols)
                except (ValueError, IndexError):
                    pass  # Skip if sampling fails

            elif op == "append" and edit_dist > 0:
                for _ in range(edit_dist):
                    insert_idx = random.randint(0, len(new_instance))
                    new_instance.insert(insert_idx, random.choice(self.alphabet))

            elif op == "delete" and len(new_instance) > 0 and edit_dist > 0:
                try:
                    delete_count = min(edit_dist, len(new_instance))
                    delete_indices = sorted(random.sample(range(len(new_instance)), delete_count), reverse=True)
                    for idx in delete_indices:
                        del new_instance[idx]
                except (ValueError, IndexError):
                    pass  # Skip if sampling fails
            
            # Track progress for early stopping
            prev_size = len(perturbed)
            perturbed.add(tuple(new_instance))
            
            if len(perturbed) == prev_size:
                no_progress_count += 1
            else:
                no_progress_count = 0
            
            trials += 1
            
            # Early exit if no progress
            if no_progress_count > max_no_progress and len(perturbed) > 0:
                break

        perturbed = list(perturbed)
        
        # Report sampling results
        if len(perturbed) < num_samples:
            message = (f"[Perturbation] Generated {len(perturbed)} unique samples "
                      f"(target: {num_samples}) in {trials} trials. "
                      f"Diversity may be limited at edit_distance={self.edit_distance}.")
            print(message)
        else:
            print(f"[Perturbation] Generated {len(perturbed)} samples in {trials} trials")

        return perturbed
    
    def _propose_single_neighbor(self, dfa, state, data, labels, seen_signatures, max_attempts: int = 1):
        """
        Generate a single random neighbor DFA by:
        1. Randomly choose one operation (DELETE/MERGE/DELTA)
        2. Randomly select indices to operate on
        3. Call the corresponding _propose_*_single method (which has internal retry logic)
        
        Note: Each operation function has its own internal max_attempts loop for robustness,
        so we keep max_attempts=1 here (no retries needed at this level).
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
        max_attempts : int
            Max retries if operation fails (default 1, since operations retry internally)
            
        Returns
        -------
        Dfa
            A single new DFA, or original DFA if mutation fails
        """
        import gc
        
        # Select a random operation to try
        operation = random.choice(['DELETE', 'MERGE', 'DELTA'])
        
        try:
            new_dfa = None
            
            if operation == 'DELETE':
                # Pick a random state index and try to delete it
                if not dfa.states:
                    gc.collect()
                    return dfa
                target_idx = random.randint(0, len(dfa.states) - 1)
                new_dfa = self._propose_delete_single(dfa, target_idx, data, labels, seen_signatures, max_attempts=10)
                
            elif operation == 'MERGE':
                # Pick two random state indices and try to merge them
                if len(dfa.states) < 2:
                    gc.collect()
                    return dfa
                state1_idx = random.randint(0, len(dfa.states) - 1)
                state2_idx = random.randint(0, len(dfa.states) - 1)
                new_dfa = self._propose_merge_single(dfa, state1_idx, state2_idx, data, labels, seen_signatures, max_attempts=10)
                
            elif operation == 'DELTA':
                # Pick a random source state and try to add/modify a transition
                if not dfa.states:
                    gc.collect()
                    return dfa
                source_idx = random.randint(0, len(dfa.states) - 1)
                transition_idx = random.randint(0, len(dfa.states) - 1)
                new_dfa = self._propose_delta_single(dfa, source_idx, transition_idx, data, labels, seen_signatures, max_attempts=10)
            
            # Operation functions always return a DFA (modified or original as fallback)
            gc.collect()
            return new_dfa if new_dfa is not None else dfa
            
        except Exception:
            gc.collect()
            return dfa

    def _propose_delete_single(self, dfa, target_state_idx: int, data, labels, seen_signatures, max_attempts: int = 10):
        """
        Propose a SINGLE new DFA by deleting a state.
        Tries up to max_attempts times with random state selections.
        Returns first successful deletion or original DFA if all attempts fail.
        """
        import gc
        max_attempts = max(1, int(max_attempts))
        
        for attempt in range(max_attempts):
            # Pick random state index
            idx = random.randint(0, len(dfa.states) - 1) if attempt > 0 else target_state_idx
            idx = max(0, min(idx, len(dfa.states) - 1))
            target_state = dfa.states[idx]
            
            # Skip initial state and sole accepting state
            if target_state == dfa.initial_state or (target_state.is_accepting and sum(x.is_accepting for x in dfa.states) <= 1):
                continue
            
            try:
                new_dfa = dfa.copy()
                target_state_in_new = next((x for x in new_dfa.states if x.state_id == target_state.state_id), None)
                if target_state_in_new is None:
                    continue
                
                outgoing = dict(target_state_in_new.transitions)
                for st in list(new_dfa.states):
                    for sym, next_s in list(st.transitions.items()):
                        if next_s == target_state_in_new:
                            st.transitions[sym] = outgoing.get(sym, st)
                
                if target_state_in_new in new_dfa.states:
                    new_dfa.states.remove(target_state_in_new)
                
                for st in new_dfa.states:
                    for sym, nxt in list(st.transitions.items()):
                        if nxt not in new_dfa.states:
                            st.transitions[sym] = st
                
                remove_unreachable_states(new_dfa)
                
                if not any(st.is_accepting for st in new_dfa.states):
                    print(f"  [SKIP] Deleting {target_state.state_id} leaves no accepting state, skipping.")
                    continue
                
                sig = self.serialize_automaton(new_dfa)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    gc.collect()
                    return new_dfa
            except Exception:
                continue
        
        gc.collect()
        return dfa

    def _propose_merge_single(self, dfa, state1_idx: int, state2_idx: int, data, labels, seen_signatures, max_attempts: int = 10):
        """
        Propose a SINGLE new DFA by merging two states.
        Tries up to max_attempts times with random state pair selections.
        Returns first successful merge or original DFA if all attempts fail.
        
        Used by PSO to avoid generating multiple candidates per operation slot.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state1_idx : int
            Index of first state to merge
        state2_idx : int
            Index of second state to merge
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
        max_attempts : int
            Maximum number of merge attempts (default 10)
            
        Returns
        -------
        Dfa
            The merged DFA if merge is valid, or original DFA if all attempts fail
        """
        import gc
        max_attempts = max(1, int(max_attempts))
        
        for attempt in range(max_attempts):
            # Pick random state pair
            if attempt == 0:
                idx1, idx2 = state1_idx, state2_idx
            else:
                idx1 = random.randint(0, len(dfa.states) - 1)
                idx2 = random.randint(0, len(dfa.states) - 1)
                if idx1 == idx2:
                    continue
            
            # Clamp indices to valid range
            idx1 = max(0, min(idx1, len(dfa.states) - 1))
            idx2 = max(0, min(idx2, len(dfa.states) - 1))
            
            if idx1 == idx2:
                continue
            
            s1 = dfa.states[idx1]
            s2 = dfa.states[idx2]
            
            # Cannot merge initial state
            if s1 == dfa.initial_state or s2 == dfa.initial_state:
                continue
            
            try:
                new_dfa = dfa.copy()
                s1_new = next((x for x in new_dfa.states if x.state_id == s1.state_id), None)
                s2_new = next((x for x in new_dfa.states if x.state_id == s2.state_id), None)
                
                if s1_new is None or s2_new is None:
                    continue
                
                # Merge s2 into s1: redirect all incoming transitions to s1
                for st in new_dfa.states:
                    for sym, nxt in list(st.transitions.items()):
                        if nxt == s2_new:
                            st.transitions[sym] = s1_new
                
                # Add s2's outgoing transitions to s1
                for sym, nxt in s2_new.transitions.items():
                    if nxt == s2_new:
                        s1_new.transitions[sym] = s1_new
                    else:
                        s1_new.transitions[sym] = nxt
                
                # Merge accepting status based on majority label
                # Compute state_label_dist similar to collect_merge_pairs_simple
                state_label_counts_s1 = defaultdict(int)
                state_label_counts_s2 = defaultdict(int)
                for seq, y in zip(data, labels):
                    cur = dfa.initial_state
                    for sym in seq:
                        if sym not in cur.transitions:
                            break
                        cur = cur.transitions[sym]
                        if cur.state_id == s1.state_id:
                            state_label_counts_s1[y] += 1
                        if cur.state_id == s2.state_id:
                            state_label_counts_s2[y] += 1
                
                s1_majority = max(state_label_counts_s1, key=state_label_counts_s1.get) if state_label_counts_s1 else 0
                s2_majority = max(state_label_counts_s2, key=state_label_counts_s2.get) if state_label_counts_s2 else 0
                merged_majority = max([s1_majority, s2_majority], default=0)
                s1_new.is_accepting = (merged_majority == 1)
                
                # Remove s2
                if s2_new in new_dfa.states:
                    new_dfa.states.remove(s2_new)
                
                # Remove unreachable states
                remove_unreachable_states(new_dfa)
                
                # Check if there are still accepting states after merge
                if not any(st.is_accepting for st in new_dfa.states):
                    print(f"  [SKIP] Merging {s2.state_id} into {s1.state_id} leaves no accepting state, skipping.")
                    continue
                
                # Check for duplicates
                sig = self.serialize_automaton(new_dfa)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    gc.collect()
                    return new_dfa
            except Exception:
                continue
        
        # All attempts failed, return original DFA as fallback
        gc.collect()
        return dfa

    def _propose_delta_single(self, dfa, source_idx: int, transition_idx: int, data, labels, seen_signatures, max_attempts: int = 10):
        """
        Propose a SINGLE new DFA by rewiring one existing transition.

        This variant is intentionally lightweight: it does not run CXP analysis.
        Instead, it picks an existing edge from the DFA, then tries to redirect it
        to a different target state. This keeps PSO / single-neighbor search cheap
        and distinct from the beam-search DELTA path.

        Uses source_idx / transition_idx as deterministic hints for the first
        attempt, then falls back to random edge / target selection.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        source_idx : int
            Index of source state
        transition_idx : int
            Index into the alphabet to select a symbol
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
        max_attempts : int
            Maximum number of transition attempts (default 10)
            
        Returns
        -------
        Dfa
            The modified DFA if transition addition is valid, or original DFA if all attempts fail
        """
        max_attempts = max(1, int(max_attempts))

        if not getattr(dfa, "states", None):
            return dfa

        edge_pool = []
        for state in dfa.states:
            for symbol, target_state in state.transitions.items():
                edge_pool.append((state, symbol, target_state))

        if not edge_pool:
            return dfa

        for attempt in range(max_attempts):
            if attempt == 0:
                source_state = dfa.states[max(0, min(source_idx, len(dfa.states) - 1))]
                outgoing = list(source_state.transitions.items())
                if outgoing:
                    symbol = outgoing[max(0, min(transition_idx, len(outgoing) - 1))][0]
                else:
                    source_state, symbol, _ = random.choice(edge_pool)
            else:
                source_state, symbol, _ = random.choice(edge_pool)

            if symbol not in source_state.transitions:
                continue

            old_target = source_state.transitions[symbol]
            candidate_targets = [s for s in dfa.states if s.state_id != old_target.state_id]
            if not candidate_targets:
                continue

            if attempt == 0:
                ordered_targets = sorted(candidate_targets, key=lambda s: s.state_id)
                target_state = ordered_targets[max(0, min(transition_idx, len(ordered_targets) - 1))]
                target_choices = [target_state] + [s for s in ordered_targets if s.state_id != target_state.state_id]
            else:
                target_choices = candidate_targets[:]
                random.shuffle(target_choices)

            for target_state in target_choices:
                try:
                    new_dfa = dfa.copy()
                    src_new = next((x for x in new_dfa.states if x.state_id == source_state.state_id), None)
                    target_new = next((x for x in new_dfa.states if x.state_id == target_state.state_id), None)

                    if src_new is None or target_new is None:
                        continue

                    src_new.transitions[symbol] = target_new
                    remove_unreachable_states(new_dfa)

                    if not any(st.is_accepting for st in new_dfa.states):
                        continue

                    sig = self.serialize_automaton(new_dfa)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        gc.collect()
                        return new_dfa
                except Exception:
                    continue

        gc.collect()
        return dfa

    def _propose_delete(self, dfa, state, data, labels, seen_signatures, beam_size):
        """
        Propose new DFAs by deleting states from the given DFA.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
            
        Returns
        -------
        list
            List of new DFAs created by deletion
        """
        import heapq, gc
        new_dfas = []
        print(f"Delete state ...")

        for s in list(dfa.states):
            # Skip initial state and the last accepting state
            if s == dfa.initial_state or (s.is_accepting and sum(x.is_accepting for x in dfa.states) <= 1):
                continue
            
            new_dfa = dfa.copy()
            target_state = next(x for x in new_dfa.states if x.state_id == s.state_id)
            
            # Inline delete logic
            # print(f"Deleting state {target_state.state_id}")
            outgoing = dict(target_state.transitions)
            for st in list(new_dfa.states):
                for sym, next_s in list(st.transitions.items()):
                    if next_s == target_state:
                        if sym in outgoing:
                            st.transitions[sym] = outgoing[sym]
                        else:
                            st.transitions[sym] = st
            
            if target_state in new_dfa.states:
                new_dfa.states.remove(target_state)

            for st in new_dfa.states:
                for sym, nxt in list(st.transitions.items()):
                    if nxt not in new_dfa.states:
                        st.transitions[sym] = st

            # Remove unreachable states before checking for accepting states
            remove_unreachable_states(new_dfa)
            
            # check if there are still accepting states after deletion
            if not any(st.is_accepting for st in new_dfa.states):
                print(f"  [SKIP] Deleting {target_state.state_id} leaves no accepting state, skipping.")
                del new_dfa, target_state, outgoing
                gc.collect()
                continue

            sig = self.serialize_automaton(new_dfa)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                self.update_state_metrics(state, dfa, new_dfa, data, labels, "DELETE")
                new_dfas.append(new_dfa)
            else:
                del new_dfa, target_state, outgoing, st
                gc.collect()

        del s
        gc.collect()

        print(f"Generated {len(new_dfas)} new DFAs from DELETE modifications.")
        print("-" * 30)
        return new_dfas

    def collect_merge_pairs_simple(self, dfa, data, labels, max_pairs=20):
        """
        Compute the label distribution for each state based on the training data.
        Then score pairs of states based on how similar their label distributions are, prioritizing pairs with the same main label.
        main label: the label with the highest count in the distribution
        
        Returns
        -------
        tuple
            (merge_pairs, state_label_dist) where state_label_dist is a dict of state_id -> {label: count}
        """
        from collections import defaultdict
        
        #  label distribution for each state
        state_label_dist = defaultdict(lambda: defaultdict(int))
        for seq, y in zip(data, labels):
            cur = dfa.initial_state
            path_states = [cur]  # Include initial state in the path
            for sym in seq:
                if sym not in cur.transitions:
                    break
                cur = cur.transitions[sym]
                path_states.append(cur)
            
            # Record label on all states in the path
            for state in path_states:
                state_label_dist[state.state_id][y] += 1
        
        # compute the main label for each state
        def main_label(state_id):
            dist = state_label_dist[state_id]
            return max(dist, key=dist.get) if dist else None
        
        # select pairs of states with the same main label
        pair_scores = []
        for s1, s2 in itertools.combinations(dfa.states, 2):
            # if s1 == dfa.initial_state or s2 == dfa.initial_state:
            #     continue
            if s1.is_accepting != s2.is_accepting:
                continue
            
            # prioritize pairs with the same main label
            if main_label(s1.state_id) == main_label(s2.state_id):
                pair_scores.append((1.0, s1, s2))
            else:
                dist1 = state_label_dist[s1.state_id]
                dist2 = state_label_dist[s2.state_id]
                # Jaccard similarity of label distributions
                all_labels = set(dist1.keys()) | set(dist2.keys())
                inter = sum(min(dist1.get(y,0), dist2.get(y,0)) for y in all_labels)
                union = sum(max(dist1.get(y,0), dist2.get(y,0)) for y in all_labels)
                sim = inter / union if union > 0 else 0
                if sim > 0:
                    pair_scores.append((sim, s1, s2))
        
        pair_scores.sort(reverse=True, key=lambda x: x[0])
        
        merge_pairs = [(s1, s2) for _, s1, s2 in pair_scores[:max_pairs]]
        return merge_pairs, state_label_dist
    
    def _propose_merge(self, dfa, state, data, labels, seen_signatures, beam_size):
        """
        Propose new DFAs by merging pairs of states in the given DFA.
        Uses intelligent scoring to prioritize high-impact merges.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
            
        Returns
        -------
        list
            List of new DFAs created by merging states
        """
        import heapq, gc
        new_dfas = []
        print(f"Merging state ...")

        feasible_pairs, state_label_dist = self.collect_merge_pairs_simple(dfa, data, labels, max_pairs=20)
        # feasible_pairs = [(s1, s2) for s1, s2 in itertools.combinations(list(dfa.states) , 2)]

        if not feasible_pairs:
            print("  [MERGE] does not find any feasible pairs to merge, skipping MERGE step.")
            return new_dfas

        # Helper function to get majority label for a state
        def get_majority_label(state_id):
            dist = state_label_dist.get(state_id, {})
            if dist:
                return max(dist, key=dist.get)
            return 0

        for s1, s2 in feasible_pairs:
            # do merge
            new_dfa = dfa.copy()
            s1_new = next(x for x in new_dfa.states if x.state_id == s1.state_id)
            s2_new = next(x for x in new_dfa.states if x.state_id == s2.state_id)

            # print(f"Merging state {s2_new.state_id} into {s1_new.state_id}")
            for st in list(new_dfa.states):
                for sym, nxt in list(st.transitions.items()):
                    if nxt == s2_new:
                        st.transitions[sym] = s1_new

            for sym, nxt in s2_new.transitions.items():
                s1_new.transitions[sym] = nxt
            
            # Merge accepting status based on majority label
            s1_majority = get_majority_label(s1.state_id)
            s2_majority = get_majority_label(s2.state_id)
            merged_majority = max([s1_majority, s2_majority], default=0)
            s1_new.is_accepting = (merged_majority == 1)
            
            if isinstance(new_dfa.states, set):
                new_dfa.states.discard(s2_new)
            else:
                try:
                    new_dfa.states.remove(s2_new)
                except ValueError:
                    pass

            # Remove unreachable states before checking for accepting states
            remove_unreachable_states(new_dfa)

            # check if there are still reachable accepting states after merge
            if not any(st.is_accepting for st in new_dfa.states):
                print(f"  [SKIP] Merging {s2.state_id} into {s1.state_id} leaves no accepting state, skipping.")
                del new_dfa, s1_new, s2_new
                gc.collect()
                continue

            sig = self.serialize_automaton(new_dfa)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                self.update_state_metrics(state, dfa, new_dfa, data, labels, "MERGE")
                new_dfas.append(new_dfa)
            else:
                del new_dfa, s1_new, s2_new
                gc.collect()

        gc.collect()

        print(f"Generated {len(new_dfas)} new DFAs from MERGE modifications.")
        print("-" * 30)
        return new_dfas

    def _trace_path(self, dfa, path: list) -> Tuple[DfaState, bool]:
        """
        trace a path through the DFA, returning the final state and whether it was fully traced
        
        Parameters
        ----------
        dfa : Dfa
        path : list
            
        Returns
        -------
        (final_state, fully_traced) : Tuple[DfaState, bool]
            final_state: trace to the final state
            fully_traced: whether the path was fully traced (True) or stopped early (False)
        """
        current = dfa.initial_state
        for symbol in path:
            if symbol not in current.transitions:
                return current, False
            current = current.transitions[symbol]
        return current, True

    def _extract_path_edges(self, dfa, path: list) -> Tuple[List[Tuple[str, str, str]], bool]:
        """
        extract edges from a path through the DFA, returning a list of (src_state_id, symbol, tgt_state_id) and whether the path was fully traced
        
        Parameters
        ----------
        dfa : Dfa
        path : list
            
        Returns
        -------
        (edges, fully_traced) : Tuple[List[Tuple], bool]
            edges: list of (src_state_id, symbol, tgt_state_id) tuples
            fully_traced: whether the path was fully traced (True) or stopped early (False)
        """
        edges = []
        current = dfa.initial_state
        for sym in path:
            next_state = current.transitions.get(sym)
            if next_state is None:
                return edges, False
            edges.append((current.state_id, sym, next_state.state_id))
            current = next_state
        return edges, True

    # def _aggregate_cxp_analysis(self, dfa, misclassified_paths, alphabet_map, mata_path, batch_size=200):
    #     """
    #     Batch CXP analysis for DELTA.

    #     Strategy:
    #     1. sample batch of misclassified paths
    #     2. Compute CXP for each sampled misclassified path
    #     3. Map shortest CXPs to edges
    #     4. Choose the most frequent blamed edge among misclassified paths
    #     """

    #     total_paths = len(misclassified_paths)
    #     if total_paths == 0:
    #         print(f"  [CXP] No misclassified paths provided")
    #         return []

    #     if total_paths > batch_size:
    #         misclassified_paths = random.sample(misclassified_paths, batch_size)
    #         # misclassified_paths = sorted(misclassified_paths, key=len)[:batch_size]

    #     sampled_count = len(misclassified_paths)
    #     print(f"  [CXP] Analyzing misclassified paths... total={total_paths}, sampled={sampled_count}")

    #     cxp_records = []

    #     # Process each misclassified path to compute CXPs
    #     for path_idx, path in enumerate(misclassified_paths):
    #         current = dfa.initial_state
    #         path_edges = []
    #         for sym in path:
    #             next_state = current.transitions.get(sym)
    #             if next_state is None:
    #                 break
    #             path_edges.append((current.state_id, sym, next_state.state_id))
    #             current = next_state

    #         if not path_edges:
    #             print(f"  [CXP]   No edges traced for path, skipping")
    #             continue

    #         path_edges, fully_traced = self._extract_path_edges(dfa, path)
    #         if not path_edges or not fully_traced:
    #             print(f"  [CXP]   Path not fully traced, skipping")
    #             continue

    #         filtered_path_edges = []
    #         encoded_word = []
    #         for edge in path_edges:
    #             _, sym, _ = edge
    #             if sym not in alphabet_map:
    #                 filtered_path_edges = []
    #                 encoded_word = []
    #                 break
    #             filtered_path_edges.append(edge)
    #             encoded_word.append(alphabet_map[sym])

    #         if not encoded_word:
    #             print(f"  [CXP]   Empty encoded word, skipping")
    #             continue

    #         if not EXPLAIN_LANGUAGE_AVAILABLE or ExplainLanguage is None:
    #             print(f"  [CXP]   ExplainLanguage unavailable, skip DELTA (no fallback)")
    #             return []

    #         try:
    #             # Use existing mata_path (passed from caller)
    #             engine = ExplainLanguage()
    #             result_data = engine.explain_word(
    #                 mata_path,
    #                 from_mata=True,
    #                 word=encoded_word,
    #                 ascii=encoded_word,
    #                 target_axp=False,
    #                 bootstrap_cxp_size_1=False,
    #                 print_exp=False,
    #             )

    #         except Exception as e:
    #             print(f"  [CXP]   WARNING: ExplainLanguage error: {type(e).__name__}: {str(e)}")
    #             print(f"  [CXP]   Skip DELTA (no fallback)")
    #             return []

    #         if not isinstance(result_data, dict):
    #             print(f"  [CXP]   WARNING: Invalid ExplainLanguage result type")
    #             print(f"  [CXP]   Skip DELTA (no fallback)")
    #             return []

    #         cxp_raw = result_data.get('cxps', []) or []
    #         valid_cxps = []
    #         for cxp in cxp_raw:
    #             try:
    #                 cxp_tuple = tuple(int(pos) for pos in cxp)
    #             except Exception:
    #                 continue
    #             if cxp_tuple:
    #                 valid_cxps.append(cxp_tuple)

    #         if valid_cxps:
    #             min_len = min(len(c) for c in valid_cxps)
    #             shortest_cxps = [c for c in valid_cxps if len(c) == min_len]

    #             for cxp_tuple in shortest_cxps:
    #                 cxp_records.append({
    #                     "cxp": cxp_tuple,
    #                     "path_edges": filtered_path_edges,
    #                 })

    #     if not cxp_records:
    #         print(f"  [CXP] No CXPs obtained from batch, skip DELTA (no fallback)")
    #         return []

    #     shortest_records = cxp_records

    #     # Map shortest CXPs back to transitions first, then do frequency counting on edges.
    #     shortest_edges = []

    #     for rec in shortest_records:
    #         blamed_edges_in_this_cxp = set()
    #         for pos in rec["cxp"]:
    #             if pos < 0 or pos >= len(rec["path_edges"]):
    #                 continue
    #             src_state_id, sym, _ = rec["path_edges"][pos]
    #             blamed_edges_in_this_cxp.add((src_state_id, sym))
    #         shortest_edges.extend(blamed_edges_in_this_cxp)

    #     if not shortest_edges:
    #         print(f"  [CXP] No valid shortest CXP mapping to transitions, skip DELTA (no fallback)")
    #         return []

    #     edge_freq = defaultdict(int)
    #     for edge in shortest_edges:
    #         edge_freq[edge] += 1

    #     ranked_edges = sorted(
    #         edge_freq.items(),
    #         key=lambda item: (-item[1], str(item[0][0]), str(item[0][1]))
    #     )
    #     if not ranked_edges:
    #         return []

    #     top_edges = [edge for edge, _ in ranked_edges[:2]]
    #     return top_edges
    def _aggregate_cxp_analysis(
        self,
        dfa,
        false_reject_paths,
        false_accept_paths,
        alphabet_map,
        mata_path,
        all_data,
        all_labels,
        batch_size=200,
        top_k=2,
    ):
        """
        Batch CXP analysis for DELTA with separate ranking for:
        1. false reject paths (label=1, DFA rejects)
        2. false accept paths (label=0, DFA accepts)

        Returns
        -------
        dict
            {
                "fr_rewire_edges": [(src_state_id, symbol), ...],
                "fa_rewire_edges": [(src_state_id, symbol), ...],
                "missing_edges":   [(src_state_id, missing_symbol), ...],
            }

        Scoring
        -------
        score(edge) = blamed_freq(edge) / (correct_usage(edge) + 1)

        - FR score is computed only from false-reject shortest CXPs
        - FA score is computed only from false-accept shortest CXPs
        - missing_edges are only collected from partially traced false-reject paths
        """

        def _sample_paths(paths, max_n):
            if len(paths) > max_n:
                return random.sample(paths, max_n)
            return paths

        def _collect_cxp_records(paths, collect_missing=False, tag="FR"):
            """
            For a set of misclassified paths:
            - fully traced paths -> compute CXP and keep each path's shortest CXP(s)
            - partially traced paths:
                * if collect_missing=True, record first missing transition
                * otherwise ignore for CXP
            """
            cxp_records = []
            missing_transition_records = []

            full_count = 0
            partial_count = 0
            explain_fail_count = 0

            for path_idx, path in enumerate(paths):
                current = dfa.initial_state
                path_edges = []
                fully_traced = True
                missing_symbol = None
                missing_src_state_id = None

                for sym in path:
                    next_state = current.transitions.get(sym)
                    if next_state is None:
                        fully_traced = False
                        missing_symbol = sym
                        missing_src_state_id = current.state_id
                        break
                    path_edges.append((current.state_id, sym, next_state.state_id))
                    current = next_state

                if not path_edges and not fully_traced:
                    partial_count += 1
                    if collect_missing and missing_src_state_id is not None and missing_symbol is not None:
                        missing_transition_records.append((missing_src_state_id, missing_symbol))
                    continue

                if fully_traced:
                    full_count += 1
                else:
                    partial_count += 1
                    if collect_missing and missing_src_state_id is not None and missing_symbol is not None:
                        missing_transition_records.append((missing_src_state_id, missing_symbol))

                # only fully traced paths can use ExplainLanguage
                if not fully_traced:
                    continue

                if not path_edges:
                    continue

                filtered_path_edges = []
                encoded_word = []
                for edge in path_edges:
                    _, sym, _ = edge
                    if sym not in alphabet_map:
                        filtered_path_edges = []
                        encoded_word = []
                        break
                    filtered_path_edges.append(edge)
                    encoded_word.append(alphabet_map[sym])

                if not encoded_word:
                    continue

                if not EXPLAIN_LANGUAGE_AVAILABLE or ExplainLanguage is None:
                    print(f"  [CXP-{tag}] ExplainLanguage unavailable, skip CXP part")
                    explain_fail_count += 1
                    continue

                try:
                    engine = ExplainLanguage()
                    result_data = engine.explain_word(
                        mata_path,
                        from_mata=True,
                        word=encoded_word,
                        ascii=encoded_word,
                        target_axp=False,
                        bootstrap_cxp_size_1=False,
                        print_exp=False,
                    )
                except Exception as e:
                    print(f"  [CXP-{tag}] WARNING: ExplainLanguage error: {type(e).__name__}: {str(e)}")
                    explain_fail_count += 1
                    continue

                if not isinstance(result_data, dict):
                    print(f"  [CXP-{tag}] WARNING: Invalid ExplainLanguage result type")
                    explain_fail_count += 1
                    continue

                cxp_raw = result_data.get("cxps", []) or []
                valid_cxps = []

                for cxp in cxp_raw:
                    try:
                        cxp_tuple = tuple(int(pos) for pos in cxp)
                    except Exception:
                        continue
                    if cxp_tuple:
                        valid_cxps.append(cxp_tuple)

                # keep each path's shortest CXP(s)
                if valid_cxps:
                    min_len = min(len(c) for c in valid_cxps)
                    shortest_cxps = [c for c in valid_cxps if len(c) == min_len]

                    for cxp_tuple in shortest_cxps:
                        cxp_records.append({
                            "cxp": cxp_tuple,
                            "path_edges": filtered_path_edges,
                        })

            print(
                f"  [CXP-{tag}] fully traced={full_count}, "
                f"partially traced={partial_count}, explain_fail={explain_fail_count}"
            )
            return cxp_records, missing_transition_records

        def _count_blamed_edges(cxp_records):
            edge_freq = defaultdict(int)
            for rec in cxp_records:
                blamed_edges_in_this_cxp = set()
                for pos in rec["cxp"]:
                    if pos < 0 or pos >= len(rec["path_edges"]):
                        continue
                    src_state_id, sym, _ = rec["path_edges"][pos]
                    blamed_edges_in_this_cxp.add((src_state_id, sym))

                for edge in blamed_edges_in_this_cxp:
                    edge_freq[edge] += 1
            return edge_freq

        def _rank_edges(blamed_edge_freq, correct_usage):
            ranked_items = []
            for edge, bfreq in blamed_edge_freq.items():
                cusage = correct_usage.get(edge, 0)
                score = bfreq / (cusage + 1)
                ranked_items.append((score, bfreq, cusage, edge))

            ranked_items.sort(
                key=lambda item: (-item[0], -item[1], item[2], str(item[3][0]), str(item[3][1]))
            )
            return ranked_items

        # --------------------------------------------------
        # 0. Sample FR / FA paths separately
        # --------------------------------------------------
        false_reject_paths = _sample_paths(false_reject_paths, batch_size)
        false_accept_paths = _sample_paths(false_accept_paths, batch_size)

        print(
            f"  [CXP] FR total={len(false_reject_paths)}, "
            f"FA total={len(false_accept_paths)}"
        )

        # --------------------------------------------------
        # 1. Collect CXP records separately
        #    - FR: compute CXPs + collect missing transitions
        #    - FA: compute CXPs only
        # --------------------------------------------------
        fr_cxp_records, missing_transition_records = _collect_cxp_records(
            false_reject_paths,
            collect_missing=True,
            tag="FR",
        )
        fa_cxp_records, _ = _collect_cxp_records(
            false_accept_paths,
            collect_missing=False,
            tag="FA",
        )

        # --------------------------------------------------
        # 2. Count blamed frequencies separately
        # --------------------------------------------------
        fr_blamed_edge_freq = _count_blamed_edges(fr_cxp_records)
        fa_blamed_edge_freq = _count_blamed_edges(fa_cxp_records)

        # --------------------------------------------------
        # 3. correct usage from correctly classified paths
        # --------------------------------------------------
        accepts = np.array([self.check_path_accepted(dfa, p) for p in all_data])
        all_labels = np.asarray(all_labels)
        correct_indices = np.where(
            ((all_labels == 1) & (accepts == True)) |
            ((all_labels == 0) & (accepts == False))
        )[0]

        correct_usage = defaultdict(int)

        for idx in correct_indices:
            path = all_data[idx]
            path_edges, _ = self._extract_path_edges(dfa, path)

            seen_edges_in_path = set()
            for src_state_id, sym, _ in path_edges:
                seen_edges_in_path.add((src_state_id, sym))

            for edge in seen_edges_in_path:
                correct_usage[edge] += 1

        # --------------------------------------------------
        # 4. Rank FR / FA edges separately + missing edges
        #    Then find the SINGLE best edge across all types
        # --------------------------------------------------
        ranked_fr_items = _rank_edges(fr_blamed_edge_freq, correct_usage)
        ranked_fa_items = _rank_edges(fa_blamed_edge_freq, correct_usage)

        # Rank missing transitions (from FR partial traces)
        missing_edge_freq = defaultdict(int)
        for missing_edge in missing_transition_records:
            missing_edge_freq[missing_edge] += 1

        ranked_missing_items = sorted(
            missing_edge_freq.items(),
            key=lambda item: (-item[1], str(item[0][0]), str(item[0][1]))
        )
        
        # --------------------------------------------------
        # 5. Combine all edges with unified scoring
        #    Form: (score, bfreq, cusage, edge, edge_type)
        # --------------------------------------------------
        all_scored_edges = []
        
        # Add FR rewire edges
        for score, bfreq, cusage, edge in ranked_fr_items:
            all_scored_edges.append((score, bfreq, cusage, edge, "FR_rewire"))
        
        # Add FA rewire edges
        for score, bfreq, cusage, edge in ranked_fa_items:
            all_scored_edges.append((score, bfreq, cusage, edge, "FA_rewire"))
        
        # Add missing edges (score = frequency)
        for missing_edge, freq in ranked_missing_items:
            all_scored_edges.append((float(freq), freq, 0, missing_edge, "missing"))
        
        # Sort by score (descending)
        all_scored_edges.sort(key=lambda x: (-x[0], -x[1], x[2], str(x[3][0]), str(x[3][1])))
        
        # Return only the top-1 best edge
        if all_scored_edges:
            score, bfreq, cusage, best_edge, edge_type = all_scored_edges[0]
            if edge_type == "FR_rewire":
                fr_rewire_edges = [best_edge]
                fa_rewire_edges = []
                missing_edges = []
            elif edge_type == "FA_rewire":
                fr_rewire_edges = []
                fa_rewire_edges = [best_edge]
                missing_edges = []
            else:  # missing
                fr_rewire_edges = []
                fa_rewire_edges = []
                missing_edges = [best_edge]
            
            if all_scored_edges:
                print(f"  [CXP] Best edge: {best_edge} (type={edge_type}, score={score:.4f})")
        else:
            fr_rewire_edges = []
            fa_rewire_edges = []
            missing_edges = []

        return {
            "fr_rewire_edges": fr_rewire_edges,
            "fa_rewire_edges": fa_rewire_edges,
            "missing_edges": missing_edges,
        }

    def _propose_delta(self, dfa, state, data, labels, seen_signatures, batch_size=32, top_k=1):
        """
        Propose new DFAs by separately repairing:

        1. false rejects (label=1 but DFA rejects)
        - rewire FR-blamed edges toward accepting-reachable states
        - add missing transitions toward accepting-reachable states

        2. false accepts (label=0 but DFA accepts)
        - rewire FA-blamed edges toward non-accepting-oriented states
            (prefer states that cannot reach accepting)
        """

        new_dfas = [dfa]  # always include original DFA

        accepts = np.array([self.check_path_accepted(dfa, p) for p in data])

        false_reject_indices = np.where((labels == 1) & (accepts == False))[0]
        false_accept_indices = np.where((labels == 0) & (accepts == True))[0]

        if len(false_reject_indices) == 0 and len(false_accept_indices) == 0:
            print("  [DELTA] No misclassified paths to analyze")
            return new_dfas

        false_reject_paths = [data[i] for i in false_reject_indices.tolist()]
        false_accept_paths = [data[i] for i in false_accept_indices.tolist()]

        # export DFA to mata for CXP
        _, alphabet_map, _, _ = dfa_to_mata(dfa, self.mata_path)

        delta_result = self._aggregate_cxp_analysis(
            dfa,
            false_reject_paths,
            false_accept_paths,
            alphabet_map,
            self.mata_path,
            all_data=data,
            all_labels=labels,
            batch_size=batch_size,
            top_k=top_k,
        )

        # fr_rewire_edges = delta_result.get("fr_rewire_edges", [])
        # fa_rewire_edges = delta_result.get("fa_rewire_edges", [])
        # missing_edges = delta_result.get("missing_edges", [])
        if not delta_result:
            print("  [DELTA] No repair candidates, returning original DFA only")
            return new_dfas

        # if not fr_rewire_edges and not fa_rewire_edges and not missing_edges:
        #     print("  [DELTA] No repair candidates, returning original DFA only")
        #     return new_dfas

        # --------------------------------------------------
        # Build state mapping
        # --------------------------------------------------
        orig_state_by_id = {st.state_id: st for st in dfa.states}

        # --------------------------------------------------
        # Collect all blamed edges and try connecting to all states
        # --------------------------------------------------
        all_blamed_edges = []
        all_blamed_edges.extend(delta_result.get("fr_rewire_edges", []))
        all_blamed_edges.extend(delta_result.get("fa_rewire_edges", []))
        all_blamed_edges.extend(delta_result.get("missing_edges", []))
        
        if not all_blamed_edges:
            print("  [DELTA] No blamed edges found")
        else:
            # Try each blamed edge with ALL possible target states
            for src_state_id, symbol in all_blamed_edges:
                src_state = orig_state_by_id.get(src_state_id)
                
                if src_state is None:
                    continue
                
                # Get old target if edge exists
                old_target = src_state.transitions.get(symbol)
                
                # Try ALL other states as targets
                target_candidates = [st for st in dfa.states 
                                    if (old_target is None or st.state_id != old_target.state_id)]
                
                if not target_candidates:
                    continue
                
                # Generate candidates for all target states
                for target_state in target_candidates:
                    try:
                        new_dfa = dfa.copy()
                        src_new = next(x for x in new_dfa.states if x.state_id == src_state_id)
                        target_new = next(x for x in new_dfa.states if x.state_id == target_state.state_id)
                        
                        src_new.transitions[symbol] = target_new
                        remove_unreachable_states(new_dfa)
                        
                        if any(st.is_accepting for st in new_dfa.states):
                            sig = self.serialize_automaton(new_dfa)
                            if sig not in seen_signatures:
                                seen_signatures.add(sig)
                                self.update_state_metrics(
                                    state, dfa, new_dfa, data, labels,
                                    f"DELTA({src_state_id}--{symbol}-->{target_state.state_id})"
                                )
                                new_dfas.append(new_dfa)
                        else:
                            del new_dfa
                            gc.collect()
                    except Exception:
                        pass

        print(f"Generated {len(new_dfas)} new DFAs from DELTA modifications.")
        print("-" * 30)
        return new_dfas


    def propose_automata(self, dfas, state, iteration, previous_best: list, output_dir: str, beam_size: int = 10, batch_size: int = 32):
        """
        Propose new DFA candidates by expanding existing DFAs.
        
        This method coordinates three strategies:
        - DELETE: Remove states from DFA (even iterations)
        - MERGE: Combine pairs of states (even iterations)
        - DELTA: Modify transitions based on CXP analysis (odd iterations)
        
        Parameters
        ----------
        dfas : list
            List of current DFAs
        state : dict
            State dictionary for tracking metrics
        sample_fcn : object
            Sampling function with feature values
        iteration : int
            Current iteration number
        previous_best : list
            List of best DFAs from previous iteration
        data_type : str
            Type of data ('Tabular', 'Text', etc.)
            
        Returns
        -------
        list
            List of proposed new DFAs
        """
        # Store output directory for mata file generation
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Unified mata file path (will be overwritten each time)
        self.mata_path = os.path.join(self.output_dir, "dfa_explicit.mata")
        
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]

        # Initialize metrics for first iteration
        if iteration == 0:
            for dfa in dfas:
                accepts = np.array([self.check_path_accepted(dfa, p) for p in data])
                true_accept = np.sum((labels == 1) & (accepts == True))
                false_reject = np.sum((labels == 0) & (accepts == False))
                
                dfa_id = id(dfa)
                state['t_nsamples'][dfa_id] = float(len(data))
                state['t_accepted'][dfa_id] = float(np.sum(accepts))
                state['t_positives'][dfa_id] = float(true_accept)
                state['t_negatives'][dfa_id] = float(false_reject)
                state['t_order'][dfa_id].append(dfa_id)

                print("--------------------------------------------")
                print(f"Proposed DFA ID: {dfa_id}")
                # print(self.automaton_to_graphviz(dfa, filename="initial_dfa", output_dir=output_dir))
            
            return dfas

        seen_signatures = set()
        new_dfas = []

        for dfa in previous_best:
            if iteration % 2 == 0:
                # Even iterations: DELETE and MERGE
                delete_candidates = self._propose_delete(dfa, state, data, labels, seen_signatures, beam_size)
                merge_candidates = self._propose_merge(dfa, state, data, labels, seen_signatures, beam_size)
                new_dfas.extend(delete_candidates)
                new_dfas.extend(merge_candidates)
            else:
                # Odd iterations: DELTA
                new_dfas.extend(self._propose_delta(dfa, state, data, labels, seen_signatures, batch_size))
        
        unique_dfas = []
        seen = set()
        for dfa in new_dfas:
            sig = self.serialize_automaton(dfa)
            if sig not in seen:
                seen.add(sig)
                unique_dfas.append(dfa)
            else:
                del dfa
                gc.collect()
        return unique_dfas
        # return new_dfas


# ==============================================================
# Module-level exports for backward compatibility
# ==============================================================

__all__ = [
    # Learner class
    'DFALearner',
    # DFA operations
    'dfa_product',
    'dfa_intersection',
    'dfa_union',
    'dfa_intersection_any',
    'get_base_dfa',
    'merge_linear_edges',
    'merge_parallel_edges',
    'simplify_dfa',
    'scar_to_aalpy_dfa',
    'clone_dfa',
    'remove_unreachable_states',
    'delete_state',
    'merge_states',
    'serialize_dfa',
    'make_dfa_complete',
    'trim_dfa',
    # Path checking
    'check_path_exist',
    'check_path_accepted',
    'get_accept_paths',
    # Visualization
    'dfa_to_graphviz',
    'dfa_to_mata',
    'explain_axp_cxp',
    'get_test_word',
    # Helper
    '_alphabet_of',
]
