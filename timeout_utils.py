import sympy as sp
import time
import multiprocessing
from math_parsers import parse_sympy_str

def _worker(func, args, kwargs, result_queue):
    try:
        result = func(*args, **kwargs)
        result_queue.put((True, result))
    except Exception as e:
        result_queue.put((False, e))

def apply_with_timeout(func, timeout, *args, **kwargs):
    """
    Applies `func(*args, **kwargs)` with a timeout (in seconds).
    Raises TimeoutError if the function call exceeds the time limit.
    Returns the result of the function if completed in time.
    """
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(func, args, kwargs, result_queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Function call timed out after {timeout} seconds")

    success, result = result_queue.get()
    if success:
        return result
    else:
        raise result

# Example usage
if __name__ == "__main__":
    x, y = sp.symbols('x y')
    expr = parse_sympy_str(
        '-A*cot(8*F*x)*csc(8*F*x)**3/(32*F) -' \
        ' 3*A*cot(8*F*x)*csc(8*F*x)/(64*F) + ' \
        '3*A*log(Abs(tan(4*F*x)))/(64*F*log(E)) + C '
        '- 69*log(sin(400*x))/1600 + 69*log(cos(400*x))/1600 + ' \
        '23*csc(400*x)**4/12800 + 69*csc(400*x)**2/6400 -' \
        ' 23*sec(400*x)**4/12800 - 69*sec(400*x)**2/6400')

    print("Starting")
    start = time.time()
    try:
        result = apply_with_timeout(sp.simplify, timeout=10, expr=expr)
        print("Result:", result)
    except TimeoutError as e:
        print("Timeout:", e)
    print("Time taken:", time.time() - start)