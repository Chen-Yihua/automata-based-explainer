import libmata.parser as parser
import libmata.nfa.nfa as nfa
import libmata.alphabets as alphabets

from pysat.examples.hitman import Hitman

from typing import List

import signal
import os
import time

import string
import json

import random

class Language:

    PRINTABLE_ASCII = [char for char in range(33,127)]
    MAX_TIMED_OUT = 30

    def __init__(self):
        self.ALPHABET = set(string.ascii_letters+string.digits)

    # is_automata means that automata_input has a .mata file or file_automata is a regex
    def explain_word(self, automata_input: str, from_mata: bool, word: str, ascii: bool, target_axp, bootstrap_cxp_size_1: bool, type_drop: int = 0, ascii_drop_char:List[int]=[], max_timed_out: int = MAX_TIMED_OUT, print_exp = False, first_minimal = False, minimal = False):
        start_total_time = time.time()
        if from_mata:
            alpha = alphabets.OnTheFlyAlphabet()
            automata = parser.from_mata(automata_input,alpha)
        else:
            automata = parser.from_regex(automata_input)
        internal_syms = list(alpha.get_alphabet_symbols())

        if not ascii_drop_char:
            ascii_drop_char = list(automata.get_symbols())
        ascii_drop_char.sort()
        if ascii:
            word_ascii = ascii
        else:
            word_ascii = self.translate_string(word)


        enumerate_func = self.enumerate_explanations_axp if target_axp else self.enumerate_explanations_cxp
        (
            status, is_accepted, count_axp, count_cxp, count_cxp_1,
            count_dual_before_1_target, time_first_target_xp, axps, cxps,
            number_minimum_target, number_minimum_dual, time_all_minimum_target_xp,
            time_first_axp, time_first_cxp
        ) = enumerate_func(automata,
                           word_ascii,
                           bootstrap_cxp_size_1,
                           type_drop,
                           ascii_drop_char,
                           max_timed_out,
                           print_exp,
                           first_minimal,
                           minimal)
        
        end_total_time = time.time() - start_total_time
        json_report = {
            "status": status,
            "rtime": end_total_time,
            "rtime_first_target": time_first_target_xp,
            "automata":  automata_input,
            "word": ','.join(map(str, word_ascii)),
            "length_word": len(word_ascii),
            "is_accepted": is_accepted,
            "count_axp": count_axp,
            "count_cxp": count_cxp,
            "count_cxp_1": count_cxp_1,
            "count_dual_before_1_target": count_dual_before_1_target,
            "target_axp": target_axp,
            "bootstrap_cxp_size_1": bootstrap_cxp_size_1,
            "type_drop": ".+" if type_drop == 1 else (".*" if type_drop == 2 else "."),
            "number_minimum_target": number_minimum_target,
            "number_minimum_dual": number_minimum_dual,
            "time_all_minimum_target_xp": time_all_minimum_target_xp,
            "time_first_axp": time_first_axp,
            "time_first_cxp": time_first_cxp,
            "axps": axps, 
            "cxps": cxps
        }
        # print(json.dumps(json_report, indent=2))
        return json_report

    

    def enumerate_explanations_axp(self, automata: nfa, word_ascii: List[int], bootstrap_cxp_size_1: bool, type_drop, ascii_drop_char: List[int] = [], max_timed_out: int = MAX_TIMED_OUT, print_exp = False, first_minimal = False, minimal = False, type_mhs=1):
        cxps, axps = [], []
        is_accepted =  automata.is_in_lang(word_ascii)
        print("automata", automata)
        if not is_accepted:
            alpha_comp = {str(s): s for s in automata.get_symbols()}
            alpha_comp.update({str(s): s for s in ascii_drop_char})
            automata = nfa.complement(automata, alphabets.OnTheFlyAlphabet.from_symbol_map(alpha_comp))
        if type_mhs==1:
            map = Hitman(htype='sorted')
        if type_mhs==2:
            map = Hitman(htype='sorted', mxs_adapt=True, mxs_exhaust=True, mxs_minz=True)
        if type_mhs==3:
            map = Hitman(htype='sorted', mxs_adapt=True, mxs_exhaust=True, mxs_trim=5)
        automata_word = nfa.Nfa(len(word_ascii)+1)
        automata_word.make_initial_state(0)
        automata_word.make_final_state(len(word_ascii))
        for index in range(len(word_ascii)):
            self.recover_char(automata_word, index, word_ascii[index])
        start_time = time.time()
        number_minimum_target = 0
        number_minimum_dual = 0
        count_dual_before_1_target = 0
        time_first_target_xp = max_timed_out
        time_first_axp = max_timed_out
        time_first_cxp = max_timed_out
        time_all_minimum_target_xp = max_timed_out
        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(max_timed_out)
        if bootstrap_cxp_size_1:
            cxps_size_1, time_first_cxp = self.bootstrap_cxp_size_1(automata, automata_word, word_ascii, type_drop, ascii_drop_char, print_exp, max_timed_out)
        else:
            cxps_size_1 = []
        count_cxp_1 = len(cxps_size_1)
        cxps.extend(cxps_size_1)
        count_cxp, count_axp = 0, 0
        for cx in cxps_size_1:
            map.hit(cx)
        skip = [True]*len(word_ascii)
        min_axp_size = float('inf')
        try:
            while True:
                candidate_xp = map.get()
                if candidate_xp is None:
                    return True, is_accepted, count_axp, count_cxp, count_cxp_1, count_dual_before_1_target, time_first_target_xp, axps, cxps, number_minimum_target, number_minimum_dual, time_all_minimum_target_xp, time_first_axp, time_first_cxp
                self.free_chars(automata_word, range(len(word_ascii)), type_drop, ascii_drop_char, skip)
                self.fix_chars(automata_word, candidate_xp, type_drop, word_ascii, skip, ascii_drop_char)
                is_included= nfa.is_included(automata_word, automata)
                
                if not is_included:
                    cxp = self.grow(automata, automata_word, word_ascii, skip, type_drop, ascii_drop_char, print_exp)
                    if time_first_cxp == max_timed_out:
                        time_first_cxp = time.time() - start_time
                    count_cxp += 1
                    map.hit(cxp)
                    cxps.append(cxp)
                else:
                    if print_exp:
                        print(f"Axp, {'_'.join([str(s) for s in candidate_xp])}")
                    if len(candidate_xp)<=min_axp_size:
                        if time_first_target_xp == max_timed_out:
                            time_first_target_xp = time.time() - start_time
                            count_dual_before_1_target=count_cxp
                        if time_first_axp == max_timed_out:
                            time_first_axp = time_first_target_xp
                        min_axp_size=len(candidate_xp)
                        number_minimum_target = count_axp + 1
                        number_minimum_dual = count_cxp
                        time_all_minimum_target_xp = time.time() - start_time
                    elif minimal:
                        return True, is_accepted, count_axp, count_cxp, count_cxp_1, count_dual_before_1_target, time_first_target_xp, axps, cxps, number_minimum_target, number_minimum_dual, time_all_minimum_target_xp, time_first_axp, time_first_cxp
                    count_axp += 1
                    axps.append(candidate_xp)
                    if first_minimal:
                        return True, is_accepted, count_axp, count_cxp, count_cxp_1, count_dual_before_1_target, time_first_target_xp, axps, cxps, number_minimum_target, number_minimum_dual, time_all_minimum_target_xp, time_first_axp, time_first_cxp
                    map.block(candidate_xp)
        except BaseException:
            if print_exp:
                print("Finished because it timed out")
            return False, is_accepted, count_axp, count_cxp, count_cxp_1, count_dual_before_1_target, time_first_target_xp, axps, cxps, number_minimum_target, number_minimum_dual, time_all_minimum_target_xp, time_first_axp, time_first_cxp
        
    def enumerate_explanations_cxp(self, automata: nfa, word_ascii: List[int], bootstrap_cxp_size_1: bool, type_drop, ascii_drop_char: List[int] = [], max_timed_out: int = MAX_TIMED_OUT, print_exp = False, first_minimal = False, minimal = False, type_mhs=1):
        axps, cxps = [], []
        universal_indexes = set(range(len(word_ascii)))
        is_accepted =  automata.is_in_lang(word_ascii)
        if not is_accepted:
            alpha_comp = {str(s): s for s in automata.get_symbols()}
            automata = nfa.complement(automata, alphabets.OnTheFlyAlphabet.from_symbol_map(alpha_comp))
            # print("automata", automata)
        ascii_drop_char.sort()
        if type_mhs==1:
            map = Hitman(htype='sorted')
        if type_mhs==2:
            map = Hitman(htype='sorted', mxs_adapt=True, mxs_exhaust=True, mxs_minz=True)
        if type_mhs==3:
            map = Hitman(htype='sorted', mxs_adapt=True, mxs_exhaust=True, mxs_trim=5)
        automata_word = nfa.Nfa(len(word_ascii)+1)
        automata_word.make_initial_state(0)
        automata_word.make_final_state(len(word_ascii))
        for state, char_ascii in enumerate(word_ascii):
            self.recover_char(automata_word, state, char_ascii)
        start_time = time.time()

        number_minimum_target = 0
        number_minimum_dual = 0
        count_dual_before_1_target = 0
        time_first_target_xp = max_timed_out
        time_first_axp = max_timed_out
        time_first_cxp = max_timed_out
        time_all_minimum_target_xp = max_timed_out

        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(max_timed_out)

        if bootstrap_cxp_size_1:
            cxps_size_1, time_first_cxp = self.bootstrap_cxp_size_1(automata, automata_word, word_ascii, type_drop, ascii_drop_char, print_exp, max_timed_out)
        else:
            cxps_size_1 = []
        count_cxp_1 = len(cxps_size_1)
        count_cxp, count_axp = 0, 0
        for cx in cxps_size_1:
            map.block(cx)
        min_cxp_size = float('inf')
        try:
            while True:
                candidate_xp = map.get()
                if candidate_xp is None:
                    return True, is_accepted, count_axp, count_cxp, count_cxp_1, count_dual_before_1_target, time_first_target_xp, axps, cxps, number_minimum_target, number_minimum_dual, time_all_minimum_target_xp, time_first_axp, time_first_cxp
                for index in candidate_xp:
                    self.drop_char(automata_word, index, type_drop, ascii_drop_char)
                if type_drop==0:
                    is_included = nfa.is_included(automata_word, automata)
                else:
                    is_included = nfa.is_included( automata_word, automata)
                if not is_included:
                    if print_exp:
                        print(f"Cxp, {'_'.join([str(s) for s in candidate_xp])}")
                    if len(candidate_xp)<=min_cxp_size:
                        if time_first_target_xp == max_timed_out:
                            time_first_target_xp = time.time() - start_time
                            count_dual_before_1_target=count_axp
                        if time_first_cxp == max_timed_out:
                            time_first_cxp = time_first_target_xp
                        min_cxp_size=len(candidate_xp)
                        number_minimum_target = count_cxp + 1
                        number_minimum_dual = count_axp
                        time_all_minimum_target_xp = time.time() - start_time
                    elif minimal:
                        return True, is_accepted, count_axp, count_cxp, count_cxp_1, count_dual_before_1_target, time_first_target_xp, axps, cxps, number_minimum_target, number_minimum_dual, time_all_minimum_target_xp, time_first_axp, time_first_cxp
                    count_cxp += 1
                    cxps.append(candidate_xp)
                    if first_minimal:
                        return True, is_accepted, count_axp, count_cxp, count_cxp_1, count_dual_before_1_target, time_first_target_xp, axps, cxps, number_minimum_target, number_minimum_dual, time_all_minimum_target_xp, time_first_axp, time_first_cxp
                    map.block(candidate_xp)
                else:
                    axp = list(universal_indexes - set(candidate_xp))
                    axp.sort()
                    axp = self.shrink(automata, automata_word, word_ascii, axp, type_drop, ascii_drop_char, print_exp)
                    if time_first_axp == max_timed_out:
                        time_first_axp = time.time() - start_time
                    count_axp +=1
                    map.hit(axp)
                    axps.append(axp)
                    # return True, is_accepted, count_axp, count_cxp, count_cxp_1, count_dual_before_1_target, time_first_target_xp, axps, cxps, number_minimum_target, number_minimum_dual, time_all_minimum_target_xp, time_first_axp, time_first_cxp
                for state in candidate_xp:
                    self.recover_char(automata_word, state, word_ascii[state], type_drop=type_drop, ascii_drop_char=ascii_drop_char)
        except BaseException:
            return False, is_accepted, count_axp, count_cxp, count_cxp_1, count_dual_before_1_target, time_first_target_xp, axps, cxps, number_minimum_target, number_minimum_dual, time_all_minimum_target_xp, time_first_axp, time_first_cxp
    

    def bootstrap_cxp_size_1(self, automata, automata_word, word_ascii, type_drop, ascii_drop_char, print_exp, max_timed_out):
        cxps = []
        start_time = time.time()
        time_first_xp = max_timed_out
        for index in range(len(word_ascii)):
            start_cxp = time.time()
            self.drop_char(automata_word, index, type_drop, ascii_drop_char)
            included = nfa.is_included(automata_word, automata)
            if not included:
                if time_first_xp == max_timed_out:
                    time_first_xp = time.time() - start_time
                if print_exp:
                    print(f"Cxp size1, {index}, {self.format_time(time.time() - start_cxp)}")
                cxps.append([index])
            self.recover_char(automata_word, index, word_ascii[index], type_drop=type_drop, ascii_drop_char=ascii_drop_char)
        return cxps, time_first_xp
    
    def grow(self, automata: nfa, candidate_cxp_nfa: nfa, word_ascii: List[int], skip: List[bool], type_drop, ascii_drop_char: List[int] = PRINTABLE_ASCII, print_exp: bool = False):
        try:
            start_cxp = time.time()
            cxp = []
            total_time_emptiness, avg_time_emptiness, iterations = 0,0,0
            min_len_counter_example = len(word_ascii)
            for candidate_index in range(len(word_ascii)):
                if skip[candidate_index]:
                    continue
                if candidate_index>min_len_counter_example:
                    break
                iterations +=1
                self.recover_char(candidate_cxp_nfa, candidate_index, word_ascii[candidate_index], type_drop=type_drop, ascii_drop_char=ascii_drop_char)
                start = time.time()
                included = nfa.is_included(candidate_cxp_nfa, automata)

                total_time_emptiness += time.time() - start
                avg_time_emptiness = (avg_time_emptiness*(iterations-1) + (time.time()-start))/iterations
                if included:
                    cxp.append(candidate_index)
                    self.drop_char(candidate_cxp_nfa, candidate_index, type_drop, ascii_drop_char)
                    skip[candidate_index] = False
                else:
                    skip[candidate_index] = True
        except BaseException:
            if print_exp:
                print(f"Incomplete Cxp, {'_'.join([str(s) for s in cxp])}, {self.format_time(time.time() - start_cxp)}, {self.format_time(total_time_emptiness)}, {self.format_time(avg_time_emptiness)}, {iterations}")
            raise
        if print_exp:
            print(f"Cxp, {'_'.join([str(s) for s in cxp])}, {self.format_time(time.time() - start_cxp)}, {self.format_time(total_time_emptiness)}, {self.format_time(avg_time_emptiness)}, {iterations}")
        return cxp
    

    def shrink(self, automata: nfa, candidate_axp_nfa: nfa, word_ascii: List[int], candidate_xp_complement: List[int], type_drop, ascii_drop_char: List[int] = PRINTABLE_ASCII, print_exp: bool = False):
        try:
            start_axp = time.time()
            axp = []
            complement_axp = []
            total_time_emptiness, avg_time_emptiness, iterations = 0,0,0
            
            for i in range(len(candidate_xp_complement)):
                iterations += 1
                self.drop_char(candidate_axp_nfa, candidate_xp_complement[i], type_drop, ascii_drop_char)

                start = time.time()
                included = nfa.is_included(candidate_axp_nfa, automata)
                total_time_emptiness += time.time() - start
                avg_time_emptiness = (avg_time_emptiness*(iterations-1) + (time.time()-start))/iterations

                if not included:
                    axp.append(candidate_xp_complement[i])
                    self.recover_char(candidate_axp_nfa, candidate_xp_complement[i], word_ascii[candidate_xp_complement[i]], type_drop, ascii_drop_char)
                else:
                    complement_axp.append(candidate_xp_complement[i])
            for index in complement_axp:
                self.recover_char(candidate_axp_nfa, index, word_ascii[index], type_drop, ascii_drop_char)
        except BaseException:
            if print_exp:
                print(f"Incomplete Axp, {'_'.join([str(s) for s in axp])}, {self.format_time(time.time() - start_axp)}, {self.format_time(total_time_emptiness)}, {self.format_time(avg_time_emptiness)}, {iterations}")
            raise
        if print_exp:
            print(f"Axp, {'_'.join([str(s) for s in axp])}, {self.format_time(time.time() - start_axp)}, {self.format_time(total_time_emptiness)}, {self.format_time(avg_time_emptiness)}, {iterations}")
        return axp

    def skipable_indexes(self, word, cex_word, cex_path, type_drop = 1):
        indexes = []
        length_cex = len(cex_path)
        for i, cex_symbol in enumerate(cex_word):
            if type_drop==1 and (i > 0 and cex_path[i] == cex_path[i - 1]) or (i < length_cex - 1 and cex_path[i] == cex_path[i + 1]):
                continue
            if cex_symbol == word[cex_path[i]]:
                indexes.append(cex_path[i])
        return indexes

    def fix_chars(self, automata_word: nfa, indexes, type_drop: int, word_ascii: List[int], skip: List[bool], ascii_drop_char: List[int] = PRINTABLE_ASCII):
        for pos_i, i in enumerate(indexes):
            if skip[i]:
                continue
            if type_drop==1 and (pos_i > 0 and indexes[pos_i] == indexes[pos_i - 1]) or (pos_i < len(indexes) - 1 and indexes[pos_i] == indexes[pos_i + 1]):
                continue
            self.recover_char(automata_word, i, word_ascii[i], type_drop, ascii_drop_char=ascii_drop_char)
            skip[i] = True
            continue
            

    def free_chars(self, automata_word: nfa, indexes, type_drop, ascii_drop_char: List[int], skip: List[bool]):
        for i in indexes:
            if skip[i]:
                self.drop_char(automata_word, i, type_drop, ascii_drop_char)
                skip[i] = False

    

    def drop_char(self, nfa_to_drop: nfa, state_index: int, type_drop, ascii_drop_char = PRINTABLE_ASCII):
        if type_drop != 2:
            for current_transition in nfa_to_drop.get_trans_from_state_as_sequence(state_index):
                nfa_to_drop.remove_trans(current_transition)
        if type_drop == 1: #\Sigma+
            for current_symbol in ascii_drop_char:
                nfa_to_drop.add_transition(state_index, current_symbol, state_index+1)
                nfa_to_drop.add_transition(state_index, current_symbol, state_index)
        if type_drop == 2: #\Sigma*
            for trans_target in nfa_to_drop.get_trans_from_state_as_sequence(state_index):
                for trans_source in nfa_to_drop.get_transitions_to_state(state_index):
                    nfa_to_drop.remove_trans(trans_source)
                    nfa_to_drop.add_transition(trans_source.source, trans_source.symbol, trans_target.target)
                for current_symbol in ascii_drop_char:
                    nfa_to_drop.add_transition(trans_target.target, current_symbol, trans_target.target)
                if state_index in nfa_to_drop.initial_states:
                    nfa_to_drop.initial_states = {trans_target.target}
            for trans_target in nfa_to_drop.get_trans_from_state_as_sequence(state_index):
                nfa_to_drop.remove_trans(trans_target)
        else: # \Sigma
            for current_symbol in ascii_drop_char:
                nfa_to_drop.add_transition(state_index, current_symbol, state_index+1)
    
    def recover_char(self, nfa_to_recover: nfa, state_index: int, recover_symbol_ascii: int, type_drop = 0, ascii_drop_char = PRINTABLE_ASCII):
        if type_drop != 2:
            for current_transition in nfa_to_recover.get_trans_from_state_as_sequence(state_index):
                nfa_to_recover.remove_trans(current_transition)
            nfa_to_recover.add_transition(state_index, recover_symbol_ascii, state_index+1)
        if type_drop == 2:
            next_state = min({x for x in nfa_to_recover.get_reachable_states() if x > state_index}, default=state_index)
            if next_state in nfa_to_recover.initial_states:
                nfa_to_recover.initial_states = {state_index}
            if next_state == state_index+1:
                for trans_target in nfa_to_recover.get_trans_from_state_as_sequence(next_state):
                    if trans_target.target == next_state:
                        nfa_to_recover.remove_trans(trans_target)

            prev_state = max({x for x in nfa_to_recover.get_reachable_states() if x < state_index}, default=state_index)
            if prev_state != state_index-1 and state_index != 0:
                for current_symbol in ascii_drop_char:
                    nfa_to_recover.add_transition(state_index, current_symbol, state_index)
            for trans_to_target in nfa_to_recover.get_transitions_to_state(next_state):
                if trans_to_target.source != trans_to_target.target and trans_to_target.source != state_index:
                    nfa_to_recover.add_transition(trans_to_target.source, trans_to_target.symbol, state_index)
                    nfa_to_recover.remove_trans(trans_to_target)
            if state_index!=next_state:
                nfa_to_recover.add_transition(state_index, recover_symbol_ascii, next_state)

    """
    l: length of words
    w: number of random words
    d: alphabeth
    """
    def generate_regex_exact(self, l: int, m: int, d: List[int]):
        words = []
        for n in range(m):
            single_word = ""
            for i in range(l):
                single_word += chr(random.choice(d))
            words.append(single_word)
        
        return '|'.join(words)
    
    def generate_regex_suffix(self, l: int, m: int, d: List[int]):
        return self.sigma_star_over_alphabet(d) + '(' + self.generate_regex_exact(l,m,d) + ')'
    

    def generate_regex_prefix(self, l: int, m: int, d: List[int]):
        return '(' + self.generate_regex_exact(l,m,d) + ')' + self.sigma_star_over_alphabet(d)
    
    def generate_regex_substring(self, l: int, m: int, d: List[int]):
        return self.sigma_star_over_alphabet(d) + '(' + self.generate_regex_exact(l,m,d) + ')' + self.sigma_star_over_alphabet(d)
    
    def create_benchmark(self, path, size_benchmark, l: int, m: int, d: List[int]):
        self.create_regex(size_benchmark, l, m, d, path, "benchmark_exact_regex_", self.generate_regex_exact)
        self.create_regex(size_benchmark, l, m, d, path, "benchmark_suffix_regex_", self.generate_regex_suffix)
        self.create_regex(size_benchmark, l, m, d, path, "benchmark_prefix_regex_", self.generate_regex_prefix)
        self.create_regex(size_benchmark, l, m, d, path, "benchmark_substring_regex_", self.generate_regex_substring)

    def create_all_substring(self, path):
        lengths = [5,10,15,20]
        m_words = [1,3,5,10]
        d_alphabet = [2,3,5,10]
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        for l in lengths:
            for m in m_words:
                for d in d_alphabet:
                    folder = os.path.join(os.path.dirname(path), f'l{l}m{m}d{d}')
                    os.makedirs(folder, exist_ok=True)
                    file_path_ouput = os.path.join(folder, f'l{l}m{m}d{d}.txt')
                    file_output = open(file_path_ouput, "w")
                    file_output.write(self.generate_regex_substring(l,m,range(97, 97+d)))
                    file_output.close

                    axp_folder = os.path.join(folder, 'target_axp')
                    os.makedirs(axp_folder, exist_ok=True)
                    axp_ouput = os.path.join(axp_folder, 'axp_job.script')
                    axp_file = open(axp_ouput, "w")
                    axp_script = self.create_script(l,m,d, "--target_axp")
                    axp_file.write(axp_script)
                    axp_file.close()

                    axp_bootstrap_folder = os.path.join(folder, 'target_axp_bootstrap_cxp_size_1')
                    os.makedirs(axp_bootstrap_folder, exist_ok=True)
                    axp_bootstrap_ouput = os.path.join(axp_bootstrap_folder, 'axp_cxp_1.script')
                    axp_bootstrap_file = open(axp_bootstrap_ouput, "w")
                    axp_bootstrap_script = self.create_script(l,m,d, "--target_axp --bootstrap_cxp_size_1")
                    axp_bootstrap_file.write(axp_bootstrap_script)
                    axp_bootstrap_file.close()

                    cxp_folder = os.path.join(folder, 'target_cxp')
                    os.makedirs(cxp_folder, exist_ok=True)
                    cxp_ouput = os.path.join(cxp_folder, 'cxp_job.script')
                    cxp_file = open(cxp_ouput, "w")
                    cxp_script = self.create_script(l,m,d, "--target_cxp")
                    cxp_file.write(cxp_script)
                    cxp_file.close()
                    

    def create_script(self, l, m, d, flags):
        return f"#!/bin/bash\n#SBATCH --time=30\n#SBATCH --cpus-per-task=1\n#SBATCH --mem=16384\n#SBATCH --nodelist=critical001\n#SBATCH --array=1-100%4\nfile=../l{l}m{m}d{d}.txt\nregex=$(sed -n '1p' \"$file\")\ninstance_word=$(sed -n \"$((SLURM_ARRAY_TASK_ID + 1))p\" \"$file\")\npython ../../../../xaidfa/main.py --explain \"$regex\" --regex --word \"$instance_word\" --name_instance instance --drop_alphabet [{','.join(map(str, range(97,97+d)))}] {flags} --timed_out 900"

    def create_regex(self, size_benchmark:int, l: int, m: int, d: List[int], path, folder_name, func_generation):
        folder = folder_name+str(int(time.time()))
        
        directory = os.path.join(os.path.dirname(path), folder)

        os.makedirs(directory, exist_ok=True)

        print(f"Create benchmark in {directory}")
        for id_benchmark in range(size_benchmark):
            file_path_ouput = os.path.join(directory, f'{folder_name}_{l}_{m}_{len(d)}_{id_benchmark}.txt')
            file_output = open(file_path_ouput, "w")
            file_output.write(func_generation(l, m, d))
            file_output.close()

    def sigma_star_over_alphabet(self, d: List[int]):
        return '(' + '|'.join(map(chr, d)) +')*'

    @staticmethod
    def translate_ascii(string: List[int]):
        return [chr(c) for c in string]
    
    @staticmethod
    def translate_string(string: str):
        if isinstance(string, list):
            if all(isinstance(x, int) for x in string):
                return string
            else:
                return [int(x) for x in string]
        if isinstance(string, str):
            try:
                return [int(w) for w in string.strip().split()]
            except ValueError:
                return string.strip().split()
        return [ord(c) for c in string]
    
    @staticmethod
    def handler(signum, frame):
        # raise BaseException("Timed out")
        return
    
    @staticmethod
    def format_time(seconds_input):
        minutes = int(seconds_input // 60)
        seconds = int(seconds_input % 60)
        milliseconds = int((seconds_input % 1) * 1000)
        return f"{minutes:02}:{seconds:02}.{milliseconds:03}"
