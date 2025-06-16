from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import numpy as np
import time


N_PROCESSES = 4
TASK_TIMEOUT = 5  # seconds

def function(n):
    time.sleep(n % 10)  # Simulate work that takes time
    return n


def process_done_future(future):
    try:
        result = future.result(timeout=1)  # Adjust timeout as needed
        print(f"Finished: {result}")
        return result
    except TimeoutError:
        print("Task timed out.")
        return "timeout"
    except ProcessExpired:
        print("Process expired.")
        return "expired"
    except Exception as e:
        print(f"Task failed with {e}")
        return "failed"


def print_running_times(start_times):
    now = time.time()
    print(' | '.join(
        [f'{int(now-t):03}' if t is not None else 'XXX'
            for t in start_times]))

if __name__ == "__main__":
    # Example usage of ProcessPool with a timeout
    queue = [i for i in range(10)]

    running_futures = [None] * N_PROCESSES
    start_times = [None] * N_PROCESSES

    # Using ProcessPool to run the function in parallel
    with ProcessPool(max_workers=N_PROCESSES) as pool:
        while True:
            for slot_id, future in enumerate(running_futures):
                if future is not None and future.done():
                    result = process_done_future(future)
                    # clear slot
                    running_futures[slot_id] = None
                    start_times[slot_id] = None

                # submit new tasks if slots are available
                if running_futures[slot_id] is None and queue:
                    print(f"Submitting task for slot {slot_id}")
                    n = queue.pop(0)
                    future = pool.schedule(
                        function, 
                        args=(n,), 
                        timeout=TASK_TIMEOUT
                    )
                    start_times[slot_id] = time.time()
                    running_futures[slot_id] = future

            print_running_times(start_times)

            if all(f is None for f in running_futures) and not queue:
                break

            time.sleep(1)  # Sleep to avoid busy waiting
