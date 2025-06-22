from sp_vars import *
import numpy as np
import pandas as pd
import sympy as sp
from check_answer_rowwise import replace_infinite_sums
import json
import sys
sys.set_int_max_str_digits(2**30)

N_SUBS = 5
N_TRIES = 200
QUESTIONS_FILE = 'ASyMOB_Validation_Dataset2.json'
OUTPUT_FILE = 'subs.csv'
VAR_SUBSTITUTIONS = {
    A: lambda: float(np.random.randint(1, 10)),
    B: lambda: float(np.random.randint(1, 10)),
    D: lambda: float(np.random.randint(1, 10)),
    F: lambda: float(np.random.randint(1, 10)),
    G: lambda: float(np.random.randint(1, 10)),
    H: lambda: float(np.random.randint(1, 10)),
    J: lambda: float(np.random.randint(1, 10)),
    K: lambda: float(np.random.randint(1, 10)),
    n: lambda: float(np.random.randint(1, 10)),
    x: lambda: float(np.abs(np.random.randn()*10)),
    sp.var('e'): lambda: sp.exp(1), # e is not a variable, but a constant
    # sp.var('pi'): lambda: sp.pi, # pi is not a variable, but a constant

    C: lambda: 0, # C is the integration constant, so we can set it to 0
}
SKIP_LIST = [
]

def floatify(expr):
    if expr.is_Number and expr.is_Integer:
        return sp.Float(expr)
    elif expr.args:
        return expr.func(*[floatify(arg) for arg in expr.args])
    else:
        return expr


if __name__ == '__main__':
    with open(QUESTIONS_FILE, 'r') as f:
        questions = json.load(f)

    all_questions_subs = {}
    for question in questions:
        q_id = question['Index']
        # if question['Variation'] in [f"Numeric-One-{i}" for i in range(7,11)]:
        #     continue
        # if question['Variation'] in [f"Numeric-All-{i}" for i in range(7,11)]:
        #     continue
        if int(q_id) in SKIP_LIST:
            # skip questions that are known to cause issues
            continue
        print(q_id, question['Answer in Sympy'])
        true_answer = sp.parse_expr(question['Answer in Sympy'], evaluate=False)
        true_answer = floatify(true_answer)
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
                # substitute pi and e, without including them in the sub_vals
                # numer_answer = numer_answer.subs({
                #     sp.var('pi'): sp.pi,
                #     sp.var('e'): sp.exp(1)
                # })
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
    csv_entries = []
    for i, subs_set in all_questions_subs.items():
        for subs, v in subs_set:
            csv_entries.append([i, json.dumps(subs), v])
    df = pd.DataFrame(csv_entries, columns=['question_id', 'subs', 'value'])
    df.to_csv(OUTPUT_FILE, index=False)