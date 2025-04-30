import pandas as pd
import numpy as np
import sympy as sp
import json
import math_parsers
from sp_vars import *
import multiprocessing as mp
from pathlib import Path
from timeout_utils import apply_with_timeout


# RESULTS_FILE = 'results_mp - 122 questions.xlsx'
RESULTS_FILE = 'results_mp_180_q.xlsx'
OUTPUT_FILE = 'checked_results.xlsx'
DISCARDED_FILE = 'discarded.xlsx'
OUTPUT_FOLDER = Path('checked_results_chunks')
NUMER_SUBS_FILE = 'numerical_subs.json'
CHUNK_TIMEOUT = 1 * 60  # 10 minutes
ROW_TIMEOUT = 10  # 60 seconds

# TODO - generate automatically.
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
    x: lambda: np.random.randn()*10,
    # sp.var('e'): lambda: sp.exp(1), # e is not a variable, but a constant
    # sp.var('pi'): lambda: sp.pi, # pi is not a variable, but a constant

    C: lambda: 0, # C is the integration constant, so we can set it to 0
}
UPPER_LIMIT_FINITE = 10
CHUNK_SIZE = 100
DEBUG = False


def replace_infinite_sums(expr, new_upper=UPPER_LIMIT_FINITE):
    """
    Recursively replaces all infinite sums (Sum objects with upper limit of oo)
    with versions using `new_upper` as the upper limit.
    """
    replacements = {}

    for s in expr.atoms(sp.Sum):
        new_limits = []
        changed = False
        for lim in s.limits:
            var, lower, upper = lim
            if upper == sp.oo:
                new_limits.append((var, lower, new_upper))
                changed = True
            else:
                new_limits.append(lim)
        if changed:
            replacements[s] = sp.Sum(s.function, *new_limits)

    return expr.xreplace(replacements)


def compare_numeric(true_answer, model_answer, subs_vals, allowed_diff=1e-5, 
                    strict=True, debug=DEBUG):
    model_answer = model_answer.removeO()
    if model_answer.has(sp.Sum):
        model_answer = replace_infinite_sums(model_answer)

    if true_answer.has(sp.Integral) != model_answer.has(sp.Integral):
        # If the model answer has an integral, but the true answer does not,
        # we can assume that the model is wrong.
        return False
    
    if debug:
        print('true_answer: ', true_answer)
        print('model_answer: ', model_answer)
        print('subs_vals:', subs_vals)

    diffs = []
    for subs, true_answer_numer in subs_vals:
        if C in model_answer.free_symbols:
            subs[C] = 0
        model_answer_numer = model_answer.subs(subs).doit()
        
        try:
            model_answer_numer = float(model_answer_numer.evalf())
        except Exception as e:
            # Couldn't cast to float -> invalid answer
            return False
        
        if true_answer_numer == 0 and model_answer_numer == 0:
            diffs.append(0)
            continue
        
        try:
            diffs.append(abs(
                (true_answer_numer - model_answer_numer) / 
                (true_answer_numer + model_answer_numer)
            ))
        except ZeroDivisionError:
            # If the sum is 0, but the terms themselves are not, we can
            # assume that the model is wrong by a sign
            return False

        if debug:
            print('subs: ', subs)
            print('diff: ', diffs[-1])
        
    try:
        if any([diff == sp.nan for diff in diffs]):
                return False
        if strict:
            # In strict mode, we check if the model and true answers are
            # numerically equal within the allowed difference.
            return all([diff < allowed_diff for diff in diffs])
        else:
            # In non-strict mode, we check if the model and true answers are
            # equal up to a constant factor.
            # It's meant to address integral answers, where the model might have
            # a constant factor in front of the answer.
            return pd.NA
        return (max(diffs) - min(diffs)) < allowed_diff
    except Exception as e:
        print(e)
        print(expr_a, expr_b)
        return pd.NA


def clean_df(df, save_discarded=False):
    df.true_answer = df.true_answer.map(math_parsers.parse_sympy_str)
    model_answer = df.final_answer_latex.map(
        math_parsers.latex_to_sympy)
    df['model_answer'] = model_answer.apply(lambda x: x[0])
    df['model_answer_type'] = model_answer.apply(lambda x: x[1])

    # Filter out invalid answers
    n_entries = len(df)
    print(f'Total number of entries: {n_entries}')

    succ_latex = df[~df.final_answer_latex.isna()]
    n_succ_latex = len(succ_latex)
    print(f'Successfully extracted latex: {n_succ_latex}.'
          f'Drop of {n_entries-n_succ_latex}')

    succ_sp = succ_latex[~succ_latex.model_answer.isna()]
    n_succ_sp = len(succ_sp)
    print(f'Successfully extracted sympy: {n_succ_sp}.'
          f'Drop of {n_succ_latex-n_succ_sp}')
    
    # Funny edge case - sometimes the model spits a wrong, numeric answer
    # like 1=0, which is translated to `False` in sympy. This later breaks the
    # comparison with the true answer. Discard these cases.
    clean_df = succ_sp[
        ~((succ_sp.model_answer != 0) & (succ_sp.model_answer == False))]
    
    if save_discarded:
        invalid_latex = df[df.final_answer_latex.isna()]
        invalid_sp = succ_latex[succ_latex.model_answer.isna()]
        invalid_sp_bool = df[
           (df.model_answer != 0) & (df.model_answer == False)]
        
        invalid_latex['discard_reason'] = 'Invalid latex'
        invalid_sp['discard_reason'] = 'Invalid sympy'
        invalid_sp_bool['discard_reason'] = 'Invalid sympy boolean'

        discarded = pd.concat([invalid_latex, invalid_sp, invalid_sp_bool])
        discarded.to_excel(DISCARDED_FILE, index=False)
        print(f'Discarded {len(discarded)} entries.')
    
    return clean_df


def check_symbolic_comparison(df, print_debug=False, timeout=None):
    if timeout is None:
        simplifyer = sp.simplify
    else:
        simplifyer = lambda expr: apply_with_timeout(
            sp.simplify, timeout, expr=expr)
    
    symb_equal = []
    for i, row in df.iterrows():
        if print_debug:
            print('Starting', i)
        true_answer = row.true_answer
        model_answer = row.model_answer
        if not true_answer.has(sp.Integral) and model_answer.has(sp.Integral):
            # If the model answer has an integral, but the true answer does not,
            # we can assume that the model is wrong.
            symb_equal.append(False)
            continue

        raw_diff = (true_answer.removeO() - model_answer.removeO())
        raw_diff = raw_diff.subs(
            {sp.var('pi'): sp.pi, sp.var('e'): sp.exp(1)}
        )
        try:
            diff = sp.powsimp(
                simplifyer(raw_diff), 
                force=True)
        except TimeoutError:
            print('Timeout error on row', i)
            symb_equal.append('Timeout')
            continue
        except Exception:
            symb_equal.append(pd.NA)
            continue
        
        symb_equal.append(
            (diff == 0) or (diff == C) or (diff == -C)
        )
        if print_debug:
            print('Finished', i)
    return pd.Series(symb_equal, index=df.index)


def check_answer_numeric(df, print_debug=False):
    with open(NUMER_SUBS_FILE, 'r') as f:
        numer_subs = json.load(f)
    numer_correct = []
    errors = []
    for i, row in df.iterrows():
        try:
            print(i)
            numer_correct.append(
                compare_numeric(
                    row.true_answer, 
                    row.model_answer, 
                    numer_subs[str(row.question_id)]
                ))
        except Exception as e:
            print(f'Error on row {i}:', e)
            print(row)
            errors.append(row)
            numer_correct.append(pd.NA)
    return pd.Series(numer_correct, index=df.index)


def check_answers(df, output_file):
    print(output_file)
    df = clean_df(df, save_discarded=True)
    try:
        # trying to run the entire chunk with a large timeout. If it fails, 
        # it usually means that inside the chunk there are a small number 
        # of row that take too much time. 
        # In that case, we run every row with a smaller timeout. It is 
        # implemented through the `timeout` parameter in 
        # the `check_symbolic_comparison` function.
        # This is a workaround for the fact that sympy doesn't have a timeout
        # for the simplify function. 
        try:
            # Chunk-wise timeout
            df['symbolic_comparison'] = apply_with_timeout(
                func=check_symbolic_comparison,
                timeout=CHUNK_TIMEOUT,
                df = df
            )
        except TimeoutError:
            # row-wise timeout
            print('Chunk-wise timeout error. Running row-wise.')
            df['symbolic_comparison'] = check_symbolic_comparison(
                df, 
                timeout=ROW_TIMEOUT
            )
        df['numeric_comparison'] = check_answer_numeric(df)

        # Convert sympys to strings for pickling on process exit
        df['model_answer'] = df.model_answer.astype('str')
        df.to_excel(OUTPUT_FOLDER / output_file)
    except Exception as e:
        print(f'Error processing file {output_file} :', e)
        df = pd.DataFrame(columns=['question_id', 'model', 'true_answer', 
                                   'model_answer', 'final_answer_latex'])
        df['error'] = str(e)
        df.to_excel(output_file, index=False)
    return df


def main():
    df = pd.read_excel(RESULTS_FILE, sheet_name='results')
    # df = clean_df(df, save_discarded=True)

    args = []
    for i in range(0, len(df), CHUNK_SIZE):
        args.append((df.iloc[i:i+CHUNK_SIZE].copy(), 
                     f'checked_{i}-{i+CHUNK_SIZE-1}.xlsx'))
        # args.append((RESULTS_FILE, i, f'checked_{i}-{i+200-1}.xlsx'))

    with mp.Pool(processes=10) as pool:
        # Map the function to the pool
        results = pool.starmap(check_answers, args)

    # Combine the results into a single DataFrame
    combined_df = pd.concat(results, ignore_index=True)
    combined_df.to_excel(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    main()