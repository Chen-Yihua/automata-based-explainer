"""
Automaton module.

This package contains automaton data structures, DFA utilities,
DOT loading helpers, and visualization/export utilities.
"""

from aalpy.automata.Dfa import Dfa as DFA, DfaState as DFAState

from .dfa_utils import (
    get_alphabet,
    dfa_product,
    dfa_intersection,
    dfa_union,
    remove_unreachable_states,
    serialize_dfa,
    make_dfa_complete,
    trim_dfa,
    dfa_to_mata,
    merge_linear_edges,
    merge_parallel_edges,
    simplify_dfa,
    plot_beam_stats,
)

from .load_dfa import (
    load_dfa_from_dot,
    create_automata_dfa_predictor,
    list_available_automata,
)

__all__ = [
    "DFA",
    "DFAState",
    "get_alphabet",
    "dfa_product",
    "dfa_intersection",
    "dfa_union",
    "remove_unreachable_states",
    "serialize_dfa",
    "make_dfa_complete",
    "trim_dfa",
    "dfa_to_mata",
    "merge_linear_edges",
    "merge_parallel_edges",
    "simplify_dfa",
    "plot_beam_stats",
    "load_dfa_from_dot",
    "create_automata_dfa_predictor",
    "list_available_automata",
]