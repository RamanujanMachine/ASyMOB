import pandas as pd
import numpy as np
import sympy as sp
import json
import math_parsers
from sp_vars import *
from pebble import ProcessPool
from pathlib import Path
from timeout_utils import apply_with_timeout
from collect_llm_answers import load_questions, extract_latex_answer

# RESULTS_FILE = 'results_mp - 122 questions.json'
RESULTS_FILE = 'joined_results.xlsx'
OUTPUT_FILE = 'checked_results.xlsx'
DISCARDED_FILE = 'discarded.json'
OUTPUT_FOLDER = Path('checked_results_chunks')
DISCARDED_FOLDER = Path('discarded_results')
NUMER_SUBS_FILE = '11_5K_questions_subs.json'
QUESTIONS_FILE = '11_5K_questions.json'


POOL_SIZE = 10
CHUNK_TIMEOUT = 5 * 60  # 10 minutes
ROW_TIMEOUT = 30  # 30 seconds
UPPER_LIMIT_FINITE = 10
CHUNK_SIZE = 50
DEBUG = False


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
    if model_answer == sp.nan:
        # If the model answer is NaN, we can assume that the model is wrong.
        # See math_parsers.latex_to_sympy_llm for more details.
        return False
    
    model_answer = model_answer.expand().removeO()
    if model_answer.has(sp.Sum):
        model_answer = replace_infinite_sums(model_answer)

    if true_answer.has(sp.Integral) != model_answer.has(sp.Integral):
        # If the model answer has an integral, but the true answer does not,
        # we can assume that the model is wrong.
        return False

    # If the model answer has different free symbols than the true answer,
    # we can assume that the model is wrong.
    model_answer_symbols = set(model_answer.free_symbols)
    true_answer_symbols = set(true_answer.free_symbols)
    # We only allow for C, the integration constant, to be different.
    model_answer_symbols.add(C)
    true_answer_symbols.add(C)
    if model_answer_symbols != true_answer_symbols:
        return False

    if debug:
        print('true_answer: ', true_answer)
        print('model_answer: ', model_answer)
        print('subs_vals:', subs_vals)

    diffs = []
    for subs, true_answer_numer in subs_vals:
        # This should be a number, but it might have I (complex number) in it.
        true_answer_numer = sp.parse_expr(true_answer_numer)
        if C in model_answer.free_symbols:
            subs[C] = 0
        model_answer_numer = model_answer.subs(subs).evalf()
        
        # model_answer_numer = float(model_answer_numer.evalf())
        
        if true_answer_numer == 0 and model_answer_numer == 0:
            diffs.append(0)
            continue
        
        diff = (true_answer_numer - model_answer_numer)
        if strict:
            # In strict mode, we check if the model and true answers are
            # numerically equal within the allowed difference.
            try:
                diffs.append(abs(
                    diff / 
                    (true_answer_numer + model_answer_numer)
                ))
            except ZeroDivisionError:
                # If the sum is 0, but the terms themselves are not, we can
                # assume that the model is wrong by a sign
                return False
        else: 
            # In non-strict mode, we check if the model and true answers are
            # equal up to a constant factor.
            # It's meant to address integral answers, where the model might have
            # a constant factor in front of the answer.
            diffs.append(diff)
        
        if debug:
            print('subs: ', subs)
            print('diff: ', diffs[-1])
        

    if any([diff == sp.nan for diff in diffs]):
            return False

    if strict:
        return all([diff < allowed_diff for diff in diffs])
    else:
        mean_diff = np.mean(diffs)
        return all([
            abs((diff - mean_diff) / mean_diff) < allowed_diff for diff in diffs
        ])


def clean_df(df, save_discarded=False, discarded_file=DISCARDED_FILE):
    def _extract_latex_answer_wrapper(full_answer):
        try:
            return extract_latex_answer(full_answer)
        except Exception as e:
            print(f"Error extracting latex answer: {e}")
            return None
    df.true_answer = df.true_answer.map(math_parsers.parse_sympy_str)
    df.final_answer_latex = df.full_answer.apply(_extract_latex_answer_wrapper)
    model_answer = df.final_answer_latex.map(
        math_parsers.latex_to_sympy)
    df['model_answer'] = model_answer.apply(lambda x: x[0])
    df['model_answer_type'] = model_answer.apply(lambda x: x[1])

    # Filter out invalid answers
    n_entries = len(df)
    # print(f'Total number of entries: {n_entries}')

    succ_latex = df[~df.final_answer_latex.isna()]
    n_succ_latex = len(succ_latex)
    # print(f'Successfully extracted latex: {n_succ_latex}.'
    #       f'Drop of {n_entries-n_succ_latex}')

    succ_sp = succ_latex[~succ_latex.model_answer.isna()]
    n_succ_sp = len(succ_sp)
    # print(f'Successfully extracted sympy: {n_succ_sp}.'
    #       f'Drop of {n_succ_latex-n_succ_sp}')
    
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
        if len(discarded) > 0:
            discarded.to_json(discarded_file, index=False)
        # print(f'Discarded {len(discarded)} entries.')
    
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
        if model_answer == sp.nan:
            # If the model answer is NaN, we can assume that the model is wrong.
            # See math_parsers.latex_to_sympy_llm for more details.
            symb_equal.append(False)
            continue
        if not true_answer.has(sp.Integral) and model_answer.has(sp.Integral):
            # If the model answer has an integral, but the true answer does not,
            # we can assume that the model is wrong.
            symb_equal.append(False)
            continue

        raw_diff = (
            true_answer.expand().removeO() - 
            model_answer.expand().removeO()
        )
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


def check_answer_numeric(df, print_debug=False, strict=True):
    with open(NUMER_SUBS_FILE, 'r') as f:
        numer_subs = json.load(f)
    numer_correct = []
    errors = []
    for i, row in df.iterrows():
        try:
            if print_debug:
                print(i)
            numer_correct.append(
                compare_numeric(
                    row.true_answer, 
                    row.model_answer, 
                    numer_subs[str(row.question_id)],
                    strict=strict
                ))
        except Exception as e:
            print(f'Error on row {i}:', e)
            print(row)
            errors.append(row)
            numer_correct.append(pd.NA)
    return pd.Series(numer_correct, index=df.index)


def check_answers(df, output_file):
    df = clean_df(
        df, 
        save_discarded=True, 
        discarded_file=DISCARDED_FOLDER / output_file)
    
    try:
        df['symbolic_comparison'] = check_symbolic_comparison(df)
        df['numeric_comparison'] = check_answer_numeric(df)

        # Convert sympys to strings for pickling on process exit
        df['model_answer'] = df.model_answer.astype('str')
        df.to_json(OUTPUT_FOLDER / output_file)
    except Exception as e:
        print(f'Error processing file {output_file} :', e)
        df = pd.DataFrame(columns=['question_id', 'model', 'true_answer', 
                                   'model_answer', 'final_answer_latex'])
        df['error'] = str(e)
        df.to_json(output_file, index=False)
    return df


def check_answers_chunks(df, pool):
    # Sending chunks, with chunk-wise timeouts
    futures = []
    results = []
    for i in range(0, len(df), CHUNK_SIZE):
        target_df = df.iloc[i:i+CHUNK_SIZE].copy()
        args = (
            target_df, 
            f'checked_{i}-{i+CHUNK_SIZE-1}.json')
        future = pool.schedule(
            check_answers, 
            args=args, 
            timeout=CHUNK_TIMEOUT
        )
        futures.append((future, i))
        
    timed_out_chunks = []
    for future, i in futures:
        try:
            results.append(future.result())
            print(f"Chunk {i}:{i+CHUNK_SIZE} completed")
        except TimeoutError as e:
            print(f"Chunk {i}:{i+CHUNK_SIZE} timed out. "
                    "Adding to retry list")
            timed_out_chunks.append(i)

    return results, timed_out_chunks


def check_answers_rows(df, idxes_to_check, pool):
    # retry timed out chunk row-wise
    futures = []
    results = []
    for chunk_start in idxes_to_check:
        for i in range(chunk_start, chunk_start+CHUNK_SIZE):
            target_df = df.iloc[i:i+1].copy()
            args = (
                target_df, 
                f'checked_{i}.json')
            future = pool.schedule(
                check_answers, 
                args=args, 
                timeout=ROW_TIMEOUT
            )
            futures.append((future, i))

    for future, i in futures:
        try:
            results.append(future.result())
            print(f"Row {i} completed")
        except TimeoutError as e:
            print(f"Row {i} timed out.")
            errored_line = df[i:i+1].copy()
            errored_line['error'] = 'timeout'
            results.append(errored_line)
    return results


def main():
    df = pd.read_excel(RESULTS_FILE) #, sheet_name='results')
    df = df[~df.full_answer.isna()]
    df = df.sample(frac=1).reset_index(drop=True)

    # The conversion to string of the true answer often breaks the sympy's 
    # constants. In particular, it converts e to the constant E, which is not
    # what we want. Reloading the original expressions here.
    questions = load_questions(parse_sympy=False)
    questions_df = pd.DataFrame.from_records(
        questions,
        columns=['q_id', 'question', 'true_answer'],
        index='q_id')
    df.true_answer = questions_df.loc[df.question_id].true_answer.values
    
    results = []
    with ProcessPool(max_workers=POOL_SIZE) as pool:
        # chunk_results, timed_out_chunks = check_answers_chunks(df, pool)
        # row_results = check_answers_rows(df, timed_out_chunks, pool)
        futures = []
        for i in range(0, len(df)):
            target_df = df.iloc[i:i+1].copy()
            args = (
                target_df, 
                f'checked_{i}.json')
            future = pool.schedule(
                check_answers, 
                args=args, 
                timeout=ROW_TIMEOUT
            )
            futures.append((future, i))
        print(f"Scheduled {len(futures)} rows")

        completed = 0
        timeouts = 0
        for future, i in futures:
            try:
                results.append(future.result())
                # print(f"Row {i} completed")
                completed += 1
            except TimeoutError as e:
                print(f"Row {i} timed out.")
                timeouts += 1
                errored_line = df[i:i+1].copy()
                errored_line['error'] = 'timeout'
                errored_line.to_json(
                    DISCARDED_FOLDER / f'errored_{i}.json', index=False)

            if (completed + timeouts) % 50 == 0:
                print(f"Completed {completed} rows, {timeouts} timeouts")
    
    # Combine the results into a single DataFrame
    # combined_df = pd.concat(chunk_results + row_results, ignore_index=True)
    # combined_df.to_json(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    main()