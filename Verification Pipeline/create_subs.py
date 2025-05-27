from sp_vars import *
import numpy as np
import pandas as pd
import sympy as sp
from check_answer_rowwise import replace_infinite_sums
import json
import sys
sys.set_int_max_str_digits(10_000_000)

N_SUBS = 5
N_TRIES = 200
QUESTIONS_FILE = 'questions_joined.json'
OUTPUT_FILE = 'leftover_subs.json'
VAR_SUBSTITUTIONS = {
    A: lambda: np.random.randint(1, 10),
    B: lambda: np.random.randint(1, 10),
    D: lambda: np.random.randint(1, 10),
    F: lambda: np.random.randint(1, 10),
    G: lambda: np.random.randint(1, 10),
    H: lambda: np.random.randint(1, 10),
    J: lambda: np.random.randint(1, 10),
    K: lambda: np.random.randint(1, 10),
    n: lambda: np.random.randint(1, 10),
    x: lambda: np.abs(np.random.randn()*10),
    sp.var('e'): lambda: sp.exp(1), # e is not a variable, but a constant
    # sp.var('pi'): lambda: sp.pi, # pi is not a variable, but a constant

    C: lambda: 0, # C is the integration constant, so we can set it to 0
}

    
if __name__ == '__main__':
    with open(r"C:\Users\admin2\Downloads\SymQLst-Eq-Num-Sym-07_05-manual (3).json", 'r') as f:
        questions = json.load(f)

    all_questions_subs = {}
    leftover_questions = [101, 102, 11711, 11712, 11713, 11714, 11715, 11716, 11717, 11718, 11719, 11720, 11721, 11722, 11723, 11724, 11725, 11726, 11727, 11728, 11729, 11730, 11731, 11732, 11733, 11734, 11735, 11736, 11737, 11738, 11739, 11740, 11741, 11742, 11743, 11744, 11745, 11746, 11747, 11748, 11749, 11750, 11751, 11752, 11753, 11754, 11755, 11756, 11757, 11758, 11759, 11760, 11761, 11762, 11763, 11764, 11765, 11766, 11767, 11768, 11769, 11770, 11771, 11772, 11773, 11774, 11775, 11776, 11777, 11778, 11779, 11780, 11781, 11782, 11783, 11784, 11785, 11786, 11787, 11788, 11789, 11790, 11791, 11792, 11793, 11794, 11795, 11796, 11797, 11798, 11799, 11800, 11801, 11802, 11803, 11804, 11805, 11806, 11807, 11808, 11809, 11810, 11811, 11812, 11813, 11814, 11815, 11816, 11817, 11818, 11819, 11820, 11821, 11822, 11823, 11824, 11825, 11826, 11827, 11828, 11829, 11830, 11831, 11832, 11833, 11834, 11835, 11836, 11837, 11838, 11839, 11840, 11841, 11842, 11843, 11844, 11845, 11846, 11847, 11848, 11849, 11850, 11851, 11852, 11853, 11854, 11855, 11856, 11857, 11858, 11859, 11860, 11861, 11862, 11863, 11864, 11865, 11866, 11867, 11868, 11869, 11870, 11871, 11872, 11873, 11874, 11875, 11876, 11877, 11878, 11879, 11880, 11881, 11882, 11883, 11884, 11885, 11886, 11887, 11888, 11889, 11890, 11891, 11892, 11893, 11894, 11895, 11896, 11897, 11898, 11899, 11900, 11901, 11902, 11903, 11904, 11905, 11906, 16543, 16544, 16545, 16546, 16547, 16548, 16549, 16550, 16551, 16552, 16553, 16554, 16555, 16556, 16557, 16558, 16559, 16560, 16561, 16562, 16563, 16564, 16565, 16566, 16567, 16568, 16569, 16570, 16571, 16572, 16573, 16574, 16575, 16576, 16577, 16578, 16579, 16580, 16581, 16582, 16583, 16584, 16585, 16586, 16587, 16588, 16589, 16590, 16591, 16592]
    for question in questions:
        q_id = question['Index']
        if int(q_id) not in leftover_questions:
            continue
        print(q_id)
        true_answer = sp.parse_expr(question['Answer in Sympy'], evaluate=False)
        true_answer = true_answer.removeO()
        if true_answer.has(sp.Sum):
            true_answer = replace_infinite_sums(true_answer)
        
        question_subs = []
        for _ in range(N_TRIES):
            vars_to_sub = true_answer.free_symbols
            if sp.var('e') in vars_to_sub:
                # e is not a variable, but a constant
                vars_to_sub.remove(sp.var('e'))
            sub_vals = {
                var: VAR_SUBSTITUTIONS[var]() 
                for var in vars_to_sub
            }
            try:
                numer_answer = true_answer.subs(sub_vals).evalf()
            except Exception as e:
                # bad substitution, try again
                continue
            if pd.isna(numer_answer):
                # bad substitution, try again
                continue
            if numer_answer is sp.nan:
                # bad substitution, try again
                continue
            # check if the answer is finite
            if numer_answer.is_infinite:
                # bad substitution, try again
                continue
            
            subs_vals_strs = {
                str(var): val for var, val in sub_vals.items()
            }
            question_subs.append((subs_vals_strs, str(numer_answer)))
            
            if len(question_subs) >= N_SUBS:
                break
        else:
            # Could not find a valid substitution after N_TRIES
            print(f"Could not find a valid substitution for question {q_id}")
            continue
        all_questions_subs[int(q_id)] = question_subs
        
    print(f'Subs generated: {len(all_questions_subs)}')
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_questions_subs, f, indent=2)