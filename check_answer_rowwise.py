import pandas as pd
import numpy as np
import sympy as sp
import json
from concurrent.futures import as_completed
from math_parsers import cached_parsing, latex_to_sympy
from sp_vars import *
from pebble import ProcessPool
from pathlib import Path
from collect_llm_answers import extract_latex_answer
import traceback
import numpy as np
from db_utils import get_connection
from functools import cache
import multiprocessing
import sys

# RESULTS_FILE = 'results_mp - 122 questions.json'
# RESULTS_FILE = 'working copy of things\\unchecked.xlsx'
RESULTS_FILE = r'results_4.1_only_assistants\joined.xlsx'
OUTPUT_FILE = r'results_4.1_only_assistants\4.1_assistant_checked.xlsx'
DISCARDED_FILE = 'discarded.json'
OUTPUT_FOLDER = Path('checked_results_chunks')
DISCARDED_FOLDER = Path('discarded_results')
NUMER_SUBS_FILE = 'numer_subs_x_positive.json'
QUESTIONS_FILE = 'questions.json'
SKIP_SYMB_CHECK_SOURCES = []

#[    'U-Math\nintegral_calc\n4c1292e1-d4b3-4acf-afaf-eaac62f2662d',
#     'UGMathBench\nCalculus_-_single_variable_0624',
#    'MathOdyssey\nProblem 328 from Calculus and Analysis - College Math',
#    'GHOSTS\nSymbolic IntegrationQ7'
#]

POOL_SIZE = 5
ROW_TIMEOUT = 30 
UPPER_LIMIT_FINITE = 10
CHUNK_SIZE = 50
DEBUG = False
CHECK_TIME = pd.Timestamp.now()
print(f"Check time: {CHECK_TIME}")


def load_tasks(sql_filter=None, parse_sympy=False, include_full_answer=True, recheck_errors=False):
    skip_retries_filter = '''
    (
    -- no answer and no error = not checked yet
    (numeric_correct is null and numeric_comparison_error is null)
               or
    (symbolic_correct is null and symbolic_comparison_error is null)
    )
    '''
    query = f"""
    select
        response_id,
        challenge_id,
        challenge,
        final_answer_latex,
        {'full_answer' if include_full_answer else 'null as full_answer'},
        answer_sympy as true_answer,
        numeric_correct,
        symbolic_correct,
        numeric_comparison_error,
        symbolic_comparison_error
    from asymob.model_responses resp
        left join asymob.challenges chal
            using (challenge_id)
        left join asymob.symbolic_verification sym_ver
            using (response_id)
        left join asymob.numeric_verification numer_ver
            using (response_id)
    -- only query unchecked items
    where (numeric_correct is null or symbolic_correct is null)
    and full_answer is not null
    {'and ' + sql_filter if sql_filter else ''}
    {'and ' + skip_retries_filter if not recheck_errors else ''}
    """ 
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
    df = df.sample(frac=1).reset_index(drop=True)

    if parse_sympy:
        df['true_answer'] = df['true_answer'].apply(cached_parsing)
    return df


def load_subs():
    query = 'select * from asymob.numerical_substitutions'
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
    df['subs_vals'] = df['subs_json'].apply(lambda x: json.loads(x))
    df.drop('subs_json', axis=1, inplace=True)
    df.set_index('challenge_id', inplace=True)
    return df


def update_db(question_data, update_symbolic=True, update_numeric=True):
    """
    Update the database with the results of the comparison.
    """
    if not update_numeric and not update_symbolic:
        print('noting new to push')
        return

    numeric_insert = r"""
    INSERT INTO asymob.numeric_verification (
        response_id,
        numeric_correct,
        strict_mode,
        numeric_comparison_error,
        check_time,
        latex_parsing_method,
        model_answer_sympy
    ) VALUES (
        %(response_id)s,
        %(numeric_correct)s,
        %(strict_mode)s,
        %(numeric_comparison_error)s,
        %(check_time)s,
        %(latex_parsing_method)s,
        %(model_answer_sympy)s
    )
    ON CONFLICT (response_id) DO UPDATE SET
        numeric_correct = EXCLUDED.numeric_correct,
        strict_mode = EXCLUDED.strict_mode,
        numeric_comparison_error = EXCLUDED.numeric_comparison_error,
        check_time = EXCLUDED.check_time,
        latex_parsing_method = EXCLUDED.latex_parsing_method,
        model_answer_sympy = EXCLUDED.model_answer_sympy;    
    """

    symbolic_insert = r"""
        INSERT INTO asymob.symbolic_verification (
            response_id,
            symbolic_correct,
            symbolic_comparison_error,
            check_time,
            latex_parsing_method,
            model_answer_sympy
        ) VALUES (
            %(response_id)s,
            %(symbolic_correct)s,
            %(symbolic_comparison_error)s,
            %(check_time)s,
            %(latex_parsing_method)s,
            %(model_answer_sympy)s
        )
        ON CONFLICT (response_id) DO UPDATE SET
            symbolic_correct = EXCLUDED.symbolic_correct,
            symbolic_comparison_error = EXCLUDED.symbolic_comparison_error,
            check_time = EXCLUDED.check_time,
            latex_parsing_method = EXCLUDED.latex_parsing_method,
            model_answer_sympy = EXCLUDED.model_answer_sympy;
    """

    question_data['check_time'] = CHECK_TIME
    question_data['model_answer_sympy'] = str(question_data.get('model_answer_sympy'))
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Insert the data into the database
            if update_symbolic:
                cursor.execute(symbolic_insert, question_data)
            if update_numeric:
                cursor.execute(numeric_insert, question_data)
            conn.commit()


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


def _meta_compare(model_answer, true_answer):
    """
    Compare the true answer and the model answer, by general metrics:
    - If the model answer is NaN, we can assume that the model gave an answer
      with unknown variables, and we can assume that the model is wrong.
      (see math_parsers.latex_to_sympy_llm for more details)
    - If the model answer has an integral, but the true answer does not,
      we can assume that the model is wrong, or that the model did not answer
      all the way through.
    - If the model answer has different free symbols than the true answer,
      we can assume that the model is wrong.
    """

    if model_answer == sp.nan:
        # See math_parsers.latex_to_sympy_llm for more details.
        return False
    
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

    return True


def compare_numeric(true_answer, model_answer, subs_vals, allowed_diff=1e-5, 
                    strict=True, debug=DEBUG):
    if debug:
        print('true_answer: ', true_answer)
        print('model_answer: ', model_answer)
        print('subs_vals:', subs_vals)

    diffs = []
    for true_answer_numer, subs in subs_vals:
        # This should be a number, but it might have I (complex number) in it.
        # So, we parse it as a sympy expression, which will convert it well.
        true_answer_numer = sp.parse_expr(true_answer_numer)
        if sp.var('e') in true_answer_numer.free_symbols:
            true_answer_numer = true_answer_numer.subs(
                {sp.var('e'): sp.exp(1)}
            )
            
        if C in model_answer.free_symbols:
            subs[C] = 0
        model_answer_numer = model_answer.subs(subs).evalf().doit()
                
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


def _compare_symbolic_wrapper(question_data):
    try:
        question_data['symbolic_correct'] = compare_symbolic(
            question_data['true_answer'], 
            question_data['model_answer_sympy']
        )
        question_data['symbolic_comparison_error'] = None
    except Exception as e:
        ex = traceback.format_exc()
        question_data['symbolic_correct'] = None
        question_data['symbolic_comparison_error'] = str(ex)
    
    return question_data


def _compare_numeric_wrapper(question_data, numeric_subs):
    try:
        if numeric_subs is None:
            # This should lead to None in the output file.
            question_data['numeric_correct'] = None
            question_data['numeric_comparison_error'] = \
                'missing substitution for numeric comparison'
            question_data['strict_mode'] = None
            return question_data

        question_data['strict_mode'] = True
        result = compare_numeric(
            question_data['true_answer'], 
            question_data['model_answer_sympy'], 
            numeric_subs,
            strict=True
        )
        # If the answer is not correct in a strict manner, and there is an
        # integral in the question, we can try to compare it in a non-strict 
        # way.
        if not result and (
            '\\int' in question_data['challenge'] or 
            'integral' in question_data['challenge'] or 
            'Integral' in question_data['challenge']
            ) and (
            '\\int_' not in question_data['challenge']
            ):
            question_data['strict_mode'] = False
            result = compare_numeric(
                question_data['true_answer'], 
                question_data['model_answer_sympy'], 
                numeric_subs,
                strict=False
            )
        question_data['numeric_correct'] = result
        question_data['numeric_comparison_error'] = None
    
    except Exception as e:
        ex = traceback.format_exc()
        question_data['numeric_correct'] = None
        question_data['numeric_comparison_error'] = str(ex)
    
    return question_data


def check_answer(question_data, numeric_subs, recheck_errors=False):
    """
    Question data - a json containing the dataframe's row. 
    """
    check_symbolic = (
        question_data['symbolic_correct'] is None and 
        (question_data['symbolic_comparison_error'] is None or recheck_errors)
    )
    check_numeric = (
        question_data['numeric_correct'] is None and 
        (question_data['numeric_comparison_error'] is None or recheck_errors)
    )
    if not check_symbolic and not check_symbolic:
        # Nothing to do here
        return 
    try:
        # the true answer is already in "clean" form, so we don't need to 
        # work hard for it.
        question_data['true_answer'] = cached_parsing(
            question_data['true_answer'])
        
        question_data['final_answer_latex'] = extract_latex_answer(
            question_data['full_answer'])
        
        model_answer, answer_type = latex_to_sympy(
            question_data['final_answer_latex']
        )
        model_answer = model_answer.expand().removeO()
        if model_answer.has(sp.Sum):
            model_answer = replace_infinite_sums(model_answer)


        question_data['model_answer_sympy'] = model_answer
        question_data['latex_parsing_method'] = answer_type
        
        should_check = _meta_compare(
            question_data['model_answer_sympy'], 
            question_data['true_answer']
        )         
            
        if not should_check:
            question_data['symbolic_correct'] = False
            question_data['numeric_correct'] = False
        else:
            if check_symbolic:
                question_data = _compare_symbolic_wrapper(question_data)
            
            if check_numeric:
                question_data = _compare_numeric_wrapper(
                    question_data, numeric_subs)

    except Exception as e:
        ex = traceback.format_exc()
        print(f'Error parsing {(question_data)}')
        print(f'Error: {ex}')
        question_data['symbolic_comparison_error'] = 'Joined error:\n' + str(ex)
        question_data['numeric_comparison_error'] = 'Joined error:\n' + str(ex)
    for key in [
        'numeric_correct', 'strict_mode', 
        'latex_parsing_method', 'model_answer_sympy', 
        'symbolic_correct']:
        if key not in question_data:
            question_data[key] = None
    
    update_db(question_data, check_symbolic, check_numeric)
    return question_data


def iter_tasks(tasks_df, already_done_df=None):
    """
    Returns a dataframe with the tasks needed to be done.
    """
    with open(QUESTIONS_FILE, 'r') as f:
        questions = json.load(f)
    questions_df = pd.DataFrame.from_records(questions)
    questions_df['question_id'] = questions_df['Index'].astype(np.int64)

    tasks_df = tasks_df.merge(
        questions_df[['question_id', 'Source', 'Variation']],
        left_on='question_id',
        right_on='question_id',
        how='left'
    )

    if already_done_df is None:
        already_done_df = pd.DataFrame(columns=tasks_df.columns)

    already_done_identifiers = [_get_distinct_identifier(row.to_dict())
        for _, row in already_done_df.iterrows()]
    
    for _, row in tasks_df.iterrows():
        identifier = _get_distinct_identifier(row.to_dict())

        if identifier in already_done_identifiers:
            continue
        
        row_dict = row.to_dict()

        if row['Variation'] != 'Original':
            continue
        yield row_dict


def main_single_core():
    tasks_df = load_tasks(parse_sympy=False, sql_filter="challenge_id < 17092")
    all_subs = load_subs()
    tasks_df.sort_values(by='challenge_id', inplace=True)

    chunk_start = sys.argv[1]
    chunk_end = chunk_start + CHUNK_SIZE
    tasks_df = tasks_df.iloc[chunk_start:chunk_end]

    results = []
    completed = 0
    timeouts = 0

    with ProcessPool(max_workers=1, max_tasks=1) as pool:
        for i, question_data in tasks_df.iterrows():
            question_data = question_data.to_dict()
            q_id = question_data['challenge_id']
            print(f"Checking {i}: {q_id} on response {question_data['response_id']}")

            if q_id not in all_subs.index:
                subs = None
            else:
                subs = all_subs.loc[q_id].values.tolist()

            future = pool.schedule(check_answer, args=(question_data, subs), timeout=ROW_TIMEOUT)

            try:
                result = future.result()
                results.append(result)
                completed += 1
            except TimeoutError:
                print(f"Row {i} (challenge_id={q_id}) timed out.")
                timeouts += 1
                errored_line = question_data.copy()
                for key in [
                    'numeric_correct', 'strict_mode',
                    'latex_parsing_method', 'model_answer_sympy',
                    'symbolic_correct']:
                    errored_line[key] = None
                errored_line['numeric_comparison_error'] = 'timeout'
                errored_line['symbolic_comparison_error'] = 'timeout'
                update_db(errored_line)
            except Exception as e:
                print(f"Row {i} (challenge_id={q_id}) failed with exception: {e}")
                errored_line = question_data.copy()
                for key in [
                    'numeric_correct', 'strict_mode',
                    'latex_parsing_method', 'model_answer_sympy',
                    'symbolic_correct']:
                    errored_line[key] = None
                errored_line['numeric_comparison_error'] = str(e)
                errored_line['symbolic_comparison_error'] = str(e)
                update_db(errored_line)

            if (completed + timeouts) % 50 == 0:
                print(f"Completed {completed} rows, {timeouts} timeouts")

def main():
    tasks_df = load_tasks(parse_sympy=False, sql_filter="challenge_id < 17092")
    all_subs = load_subs()
    tasks_df.sort_values(by='challenge_id', inplace=True)

    results = []
    with ProcessPool(max_workers=POOL_SIZE) as pool:
        # chunk_results, timed_out_chunks = check_answers_chunks(df, pool)
        # row_results = check_answers_rows(df, timed_out_chunks, pool)
        future_to_index = {}

        for i, question_data in tasks_df.iterrows():
            question_data = question_data.to_dict()
            q_id = question_data['challenge_id']
            
            if q_id not in all_subs.index:
                subs = None
            else:
                subs = all_subs.loc[q_id].values.tolist()
            args = (question_data, subs)
            future = pool.schedule(
                check_answer, 
                args=args, 
                timeout=ROW_TIMEOUT
            )
        future_to_index[future] = i

        print(f"Scheduled {len(tasks_df)} rows")

        completed = 0
        timeouts = 0
        for future in as_completed(future_to_index):
            print('something is done!')
            try:
                results.append(future.result())
                # print(f"Row {i} completed")
                completed += 1
            except TimeoutError as e:
                print(f"Row {i} timed out.")
                timeouts += 1
                errored_line = tasks_df.iloc[i].copy().to_dict()
                for key in [
                    'numeric_correct', 'strict_mode', 
                    'latex_parsing_method', 'model_answer_sympy', 
                    'symbolic_correct']:
                    errored_line[key] = None
                errored_line['numeric_comparison_error'] = 'timeout'
                errored_line['symbolic_comparison_error'] = 'timeout'
                update_db(errored_line)

            if (completed + timeouts) % 50 == 0:
                print(f"Completed {completed} rows, {timeouts} timeouts")
    
    # # Combine the results into a single DataFrame
    # results = pd.DataFrame.from_records(results)
    # combined_df = pd.concat([already_done_df, results], axis=1)
    # combined_df.to_excel(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    # main()
    main_single_core()