import pandas as pd
import numpy as np
import sympy as sp
import json
import math_parsers
from sp_vars import *
from pebble import ProcessPool
from pathlib import Path
from collect_llm_answers import load_questions, extract_latex_answer
import traceback

# RESULTS_FILE = 'results_mp - 122 questions.json'
RESULTS_FILE = 'full_answers_not_checked_joined.xlsx'
OUTPUT_FILE = 'checked_numer_only_some_holes_removed_integrals.xlsx'
DISCARDED_FILE = 'discarded.json'
OUTPUT_FOLDER = Path('checked_results_chunks')
DISCARDED_FOLDER = Path('discarded_results')
NUMER_SUBS_FILE = 'numer_subs.json'
QUESTIONS_FILE = 'questions.json'


POOL_SIZE = 10
CHUNK_TIMEOUT = 5 * 60  # 10 minutes
ROW_TIMEOUT = 10  # 30 seconds
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


def _get_distinct_identifier(question_data):
    return (
        f"{question_data['question_id']}_"
        f"{question_data['model']}_"
        f"{question_data['code_execution']}"
    )


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
            diffs.append(abs(diff))
        
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
            abs(diff - mean_diff) < allowed_diff for diff in diffs
        ])


def compare_symbolic(true_answer, model_answer):
    """
    Compare two sympy expressions true_answer and model_answer.
    
    Returns True if they are equal, False otherwise.
    """
    # If the model answer is NaN, we can assume that the model is wrong.
    # See math_parsers.latex_to_sympy_llm for more details.
    if model_answer == sp.nan:
        return False

    # If the model answer has an integral, but the true answer does not,
    # we can assume that the model is wrong.
    if not true_answer.has(sp.Integral) and model_answer.has(sp.Integral):
        return False

    # Start comparing
    raw_diff = (
        true_answer.expand().removeO() - 
        model_answer.expand().removeO()
    )
    raw_diff = raw_diff.subs(
        {sp.var('pi'): sp.pi, sp.var('e'): sp.exp(1)}
    )
    
    diff = sp.powsimp(sp.simplify(raw_diff), force=True)
    return (diff == 0) or (diff == C) or (diff == -C)


def check_answers(question_data, numeric_subs, output_file):
    """
    Question data - a json containing the dataframe's row. 
    in includes thw following entries:
        question_id, model, code_execution, full_answer, tokens_used,
        final_answer_latex, question_text, true_answer, error
    Some of the fields are parsed as strings, and this function reconverts
    them to sympy objects.
    """
    try:
        question_data['true_answer'] = math_parsers.parse_sympy_str(
            question_data['true_answer'])
        
        question_data['final_answer_latex'] = extract_latex_answer(
            question_data['full_answer'])
        
        model_ans = math_parsers.latex_to_sympy(
            question_data['final_answer_latex']
        )
        question_data['model_answer'] = model_ans[0]
        question_data['model_answer_type'] = model_ans[1]
        
        # question_data['symbolic_comparison'] = compare_symbolic(
        #    question_data['true_answer'], 
        #    question_data['model_answer']
        # )

        if numeric_subs is None:
            question_data['numeric_comparison'] = None
            question_data['numeric_subs_error'] = 'missing data'
        else:
            if '\\int' in question_data['question_text']:
                strict = False
            else:
                strict = True
            question_data['numeric_comparison'] = compare_numeric(
                question_data['true_answer'], 
                question_data['model_answer'], 
                numeric_subs,
                strict=strict
            )

    except Exception as e:
        ex = traceback.format_exc()
        print(f'Error parsing {_get_distinct_identifier(question_data)}')
        print(f'Error: {ex}')
        question_data['error'] = str(ex)

    # Sympy objects are often not JSON serializable, convert them to strings.
    if 'model_answer' in question_data:
        question_data['model_answer'] = str(question_data['model_answer'])
    if 'true_answer' in question_data:
        question_data['true_answer'] = str(question_data['true_answer'])
    
    # Write to output file
    df = pd.DataFrame.from_records([question_data])
    df.to_excel(
        OUTPUT_FOLDER / output_file, 
        index=False
    )

    return question_data    



def iter_tasks(tasks_df, already_done_df=None):
    """
    Returns a dataframe with the tasks needed to be done.
    """
    if already_done_df is None:
        already_done_df = pd.DataFrame(columns=tasks_df.columns)

    already_done_identifiers = [_get_distinct_identifier(row.to_dict())
        for _, row in already_done_df.iterrows()]
    
    for _, row in tasks_df.iterrows():
        identifier = _get_distinct_identifier(row.to_dict())
        if identifier not in already_done_identifiers:
            yield row.to_dict()


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
    
    with open(NUMER_SUBS_FILE, 'r') as f:
        numer_subs = json.load(f)
    already_done_df = pd.read_excel(OUTPUT_FILE)
    bad_questions = list(numer_subs.keys())
    
    results = []
    with ProcessPool(max_workers=POOL_SIZE) as pool:
        # chunk_results, timed_out_chunks = check_answers_chunks(df, pool)
        # row_results = check_answers_rows(df, timed_out_chunks, pool)
        futures = []
        for i, question_data in enumerate(iter_tasks(df, already_done_df)):
            if str(question_data['question_id']) not in bad_questions:
                # print(f"Question {question_data['question_id']} has no subs")
                args = (
                    question_data, 
                    None,
                    f'checked_{i}.xlsx'
                )
            else:
                args = (
                    question_data, 
                    numer_subs[str(question_data['question_id'])],
                    f'checked_{i}.xlsx'
                )
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
                errored_line.to_excel(
                    OUTPUT_FOLDER / f'timeout_{i}.xlsx')

            if (completed + timeouts) % 50 == 0:
                print(f"Completed {completed} rows, {timeouts} timeouts")
    
    # Combine the results into a single DataFrame
    results = pd.DataFrame.from_records(results)
    combined_df = pd.concat([already_done_df, results], axis=1)
    combined_df.to_excel(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    main()