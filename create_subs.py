from sp_vars import *
import numpy as np
import pandas as pd
import sympy as sp
from collect_llm_answers import load_questions
from check_answer_rowwise import replace_infinite_sums
import json
import sys
sys.set_int_max_str_digits(10_000_000)

N_SUBS = 5
N_TRIES = 200
QUESTIONS_FILE = 'questions.json'
OUTPUT_FILE = 'numer_subs_x_positive.json'
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

def _filter_large_numeric_noise(question):
    if 'Numeric' not in question['Variation']:
        return True
    noise_deg = int(question['Variation'].split('-')[-1])
    if noise_deg < 7:
        return True
    return False

def _filter_large_powers(question):
    """
    Large powers are not parsed in a decent time by sympy.
    We load the questions without evaluating the answer, so we can filter them
    out.
    """
    # Michael be healthy
    if int(question['Index']) in [
        5387, 5388, 5389, 5390, 5436, 5437, 5438, 5439, 5440, 6528, 6529, 6530,
        6531, 6532, 7104, 7105, 7106, 7107, 7108, 10427, 10428, 10429, 10430,
        10431, 10432, 10422, 9121, 9122, 9123, 9124, 9125, 9126, 9151, 9152, 
        9153, 9154, 9155, 9156, 7935, 7936, 7937, 7938, 7984, 7985, 7986,
        7987, 7988, 9076, 9077, 9078, 9079, 9080, 9652, 9653, 9654, 9655, 9656,
        12970, 12977, 12978, 12979, 12980, 11670, 11671, 11672, 11673, 11674,
        11700]: 
        return False
    if 'Numeric' not in question['Variation']:
        return True
    numeric_level = question['Variation'].split('-')[-1]
    if numeric_level == 'S':
        # S means symbolic, so we can skip it
        return True
    numeric_level = int(numeric_level)
    if numeric_level >= 7:
        return False
    unevaluated = sp.parse_expr(question['Answer in Sympy'], evaluate=False)
    try:
        if unevaluated.has(sp.Pow):
            # check if the power is large
            for base, exp in unevaluated.as_powers_dict().items():
                if abs(exp) > 1_000:
                    return False
    except Exception as e:
        return True
    return True
    
    
if __name__ == '__main__':
    questions = load_questions(
        QUESTIONS_FILE,
        filter_func=_filter_large_powers,
        )
    print(f'Total questions: {len(questions)}')
    all_questions_subs = {}

    for q_id, _, true_answer in questions:
        print(q_id)
        true_answer = true_answer.removeO()
        if true_answer.has(sp.Sum):
            true_answer = replace_infinite_sums(true_answer)
        
        question_subs = []
        for _ in range(N_TRIES):
            sub_vals = {
                var: VAR_SUBSTITUTIONS[var]() 
                for var in true_answer.free_symbols
            }
            try:
                numer_answer = true_answer.subs(sub_vals).evalf()
            except Exception as e:
                # bad substitution, try again
                continue
            if pd.isna(numer_answer):
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