"""
A validation worker that continuously checks for new responses from the LLM
and validates them.
See `check_answers_rowwise.py` for the main validation logic.
"""
from check_answer_rowwise import check_answer, load_tasks, load_subs, update_db
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import time

N_PROCESSES = 4
SLEEP_TIME = 5  # seconds
TASK_TIMEOUT = 30 # seconds


def update_tasks(tasks_df):
    if tasks_df is None or tasks_df.empty:
        return load_tasks(
            parse_sympy=False, 
            sql_filter="challenge_id < 17092"
        )

    return tasks_df


def process_done_future(future, task):
    try:
        result = future.result(timeout=1)  # Adjust timeout as needed
        return result
    
    except TimeoutError:
        print("Task timed out.")
        task['numeric_comparison_error'] = 'timeout'
        task['symbolic_comparison_error'] = 'timeout'
    
    except Exception as e:
        print(f"Task failed with {e}")
        task['numeric_comparison_error'] = str(e)
        task['symbolic_comparison_error'] = str(e)

    for key in [
        'numeric_correct', 'strict_mode',
        'latex_parsing_method', 'model_answer_sympy',
        'symbolic_correct']:
        task[key] = None
    update_db(task)

def print_running_times(start_times):
    now = time.time()
    print(' | '.join(
        [f'{int(now-t):03}' if t is not None else 'XXX'
            for t in start_times]))

    

if __name__ == "__main__":
    all_subs = load_subs()
    tasks_df = load_tasks(
        parse_sympy=False, 
        # sql_filter="challenge_id < 17092"
    )

    running_futures = [None] * N_PROCESSES
    running_tasks = [None] * N_PROCESSES
    start_times = [None] * N_PROCESSES
    running_times = [None] * N_PROCESSES
    

    with ProcessPool(max_workers=N_PROCESSES) as pool:
        while True:
            for slot_id, (future, task) in enumerate(zip(running_futures, running_tasks)):
                if future is not None and future.done():
                    result = process_done_future(future, task)
                    # clear slot
                    running_futures[slot_id] = None
                    start_times[slot_id] = None

                # submit new tasks if slots are available
                if running_futures[slot_id] is None and len(tasks_df) > 0:
                    print(f"Submitting task for slot {slot_id}")
                    task = tasks_df.iloc[0]
                    tasks_df = tasks_df.iloc[1:]
                    question_data = task.to_dict()
                    q_id = question_data['challenge_id']
                    print(f"Checking {q_id} on response {question_data['response_id']}")

                    if q_id not in all_subs.index:
                        subs = None
                    else:
                        subs = all_subs.loc[q_id].values.tolist()
                    future = pool.schedule(
                        check_answer, 
                        args=(question_data, subs), 
                        timeout=TASK_TIMEOUT
                    )
                    start_times[slot_id] = time.time()
                    running_futures[slot_id] = future
                    running_tasks[slot_id] = question_data

            print_running_times(start_times)

            if all(f is None for f in running_futures) and len(tasks_df) == 0:
                print("All tasks completed.")
                tasks_df = update_tasks(tasks_df)
                tasks_df = load_tasks(
                    parse_sympy=False, 
                    # sql_filter="challenge_id < 17092"
                )
                print(f"Loaded {len(tasks_df)} tasks.")

            time.sleep(1)  # Sleep to avoid busy waiting