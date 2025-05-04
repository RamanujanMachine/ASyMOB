from sp_vars import *
import numpy as np
import pandas as pd
import sympy as sp
from collect_llm_answers import load_questions
from check_answers import replace_infinite_sums
import json

N_SUBS = 5
N_TRIES = 200
QUESTIONS_FILE = '11_5K_questions.json'
OUTPUT_FILE = '11_5K_questions_subs.json'
VAR_SUBSTITUTIONS = {
    A: lambda: np.random.randint(1, 10),
    B: lambda: np.random.randint(1, 10),
    D: lambda: np.random.randint(1, 10),
    E: lambda: np.random.randint(1, 10),
    F: lambda: np.random.randint(1, 10),
    G: lambda: np.random.randint(1, 10),
    H: lambda: np.random.randint(1, 10),
    J: lambda: np.random.randint(1, 10),
    K: lambda: np.random.randint(1, 10),
    n: lambda: np.random.randint(1, 10),
    x: lambda: np.random.randn()*10,
    sp.var('e'): lambda: sp.exp(1), # e is not a variable, but a constant
    # sp.var('pi'): lambda: sp.pi, # pi is not a variable, but a constant

    C: lambda: 0, # C is the integration constant, so we can set it to 0
}

if __name__ == '__main__':
    questions = load_questions(QUESTIONS_FILE)
    print(f'Total questions: {len(questions)}')
    all_questions_subs = {}

    for q_id, _, true_answer in questions:
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
                numer_answer = float(true_answer.subs(sub_vals).evalf())
            except Exception as e:
                # bad substitution, try again
                continue
            if pd.isna(numer_answer):
                # bad substitution, try again
                continue
            # check if the answer is finite
            if not np.isfinite(numer_answer):
                # bad substitution, try again
                continue
            
            subs_vals_strs = {
                str(var): val for var, val in sub_vals.items()
            }
            question_subs.append((subs_vals_strs, numer_answer))
            
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