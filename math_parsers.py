import re
import inspect
from sp_vars import *
import sympy as sp
import pandas as pd
from functools import cache
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from gemini_interface import GeminiInterface
import sys
sys.set_int_max_str_digits(10_000_000)


SYMPY_CONVERTER = GeminiInterface("gemini-2.0-flash")
KNOWN_FUNCTIONS = {
    'atan': sp.atan,
    'atan2': sp.atan2,
    'asin': sp.asin,
    'arctan': sp.atan,
    'O': sp.O,
    'gamma': sp.gamma,
    'Gamma': sp.gamma,
    'exp': sp.exp,
}

class LatexFuncCallError(Exception):
    pass

def remove_equality(sp_expr):
    try:
        if isinstance(sp_expr, sp.Equality):
            return remove_equality(sp_expr.rhs)
        return sp_expr
    except Exception as e:
        print(sp_expr)
        raise e


def clean_sp_object(sp_expr, swap_funcs=True):
    # Sometimes, the answer is a number, and the parsing might result in a 
    # pythonic-number, which is not compatible with sympy the rest of the 
    # pipeline. We need to convert it to a sympy object.
    if isinstance(sp_expr, float):
        return sp.Float(sp_expr)
    elif isinstance(sp_expr, int):
        return sp.Integer(sp_expr)
    
    sp_expr = fix_expr(remove_equality(sp_expr), swap_funcs)
    sp_expr =  sp_expr.subs({
        sp.var('e'): sp.exp(1),
        sp.var('pi'): sp.pi,
        sp.var('i'): sp.I,
    })
    return sp_expr


@cache
def parse_sympy_str(expr_str):
    try:
        if pd.isna(expr_str):
            return None
        expr_str = str(expr_str)
        return clean_sp_object(sp.parse_expr(
            expr_str, 
            local_dict=var_mapping, 
            evaluate=False
        ))
    except Exception as e:
        print(expr_str)
        raise e


def latex_to_sympy_deter(latex_str):
    """
    Convert a LaTeX string to a sympy expression, using the sympy parser.
    """
    if pd.isna(latex_str):
        return pd.NA
    try:
        # Latex cleanup
        latex_str = re.sub(
            r'\\(?:left|big|Big|Biggl|Bigl)' # Non-consuming left indicator
            r'[\[\(\|\{]', 
            lambda match: match.group(0)[-1], 
            latex_str)
        latex_str = re.sub(
            r'\\(?:right|big|Big|Biggr|Bigr)'
            r'[\]\)\|\}]', 
            lambda match: match.group(0)[-1], 
            latex_str)
        
        # Fix the latex string to be compatible with sympy
        latex_str = re.sub(
            r'(\\operatorname{.*?})',
            lambda match: '\\' + match.group(0)[:-1].split('{')[1],
            latex_str)
        
        # Fix the latex string to be compatible with sympy
        latex_str = re.sub(
            r'(?<!\\)atan',
            r'\\atan',
            latex_str)
        
        # use constants for e and pi
        return clean_sp_object(parse_latex(latex_str), swap_funcs=False)
    except LatexFuncCallError as e:
        raise e 
    except Exception as e:
        # print('Error parsing latex string:')
        # print(latex_str)
        # print(e)
        return pd.NA


def latex_to_sympy_llm(latex_str, debug=False):
    """
    Convert a LaTeX string to a sympy expression, using the LLM parser.
    """
    function_def = SYMPY_CONVERTER.send_message(
        message="Following is a mathematical expression written in latex. "
                "Please create a python function that recreates this "
                "expression using sympy, exactly as it is written in the "
                "expression given. The function's signature will be "
                "solution(...) and it will accept as parameters all of the "
                "variables and constant in the expression you are given. It "
                "will return the expression as a sympy object. The "
                "implementation will be identical to the expression given. "
                "You will not use any form of simplification or modification "
                "for the mathematical expression. Only print the function, "
                "without additional text. Access sympy functions through the " 
                "sp module (e.g. sp.hyper). There are no custom functions, "
                "only elementary functions. Do not use sp.Rational, use the " 
                f"division operator (e.g. /). \n\n $${latex_str}$$" 
        )
    if debug:
        print(function_def)
    function_def = function_def.strip("`\n")
    if function_def.startswith('python\n'):
        function_def = function_def[6:]

    exec(function_def, globals())
    
    # Dynamically extract the function's arguments and pass what is needed
    sig = inspect.signature(solution)
    accepted_params = sig.parameters
    
    for param in accepted_params:
        # Check if the model requested a variable that is not in the standard 
        # var_mapping. This which usually means that the model invented a new 
        # variable, which means it was wrong. We mark those cases as `sp.nan`.
        # This workaround lets us use the rest of the pipeline without
        # modification.
        if param not in used_vars:
            print(f"Unknown variable {param} in function definition")
            return sp.nan
    filtered_args = {k: v for k, v in var_mapping.items() if k in accepted_params}

    # Function solution defined dynamically
    return clean_sp_object(solution(
        **filtered_args
    ))

def latex_to_sympy(latex_str, debug=False):
    """
    Convert a LaTeX string to a sympy expression, using the LLM parser.
    """
    if pd.isna(latex_str):
        return pd.NA, 'na'
    try:
        latex_str = str(latex_str)
        sp_expr = latex_to_sympy_deter(latex_str)
        if (not pd.isna(sp_expr) and 
            sp_expr.free_symbols.issubset(set(var_mapping.values()))): 
            return sp_expr, 'deterministic'
        else:
            return latex_to_sympy_llm(latex_str, debug=debug), 'llm'
    except LatexFuncCallError as e:
        return latex_to_sympy_llm(latex_str, debug=debug), 'llm (val func call)'
    except Exception as e:
        # print('Error parsing latex string:')
        # print(latex_str)
        # print(e)
        return pd.NA, 'error'


def format_final_answer_to_sympy(textual_answer):
    sp_expr = SYMPY_CONVERTER.send_message(
        message="Following is an answer to a mathematical question. "
                "Please write the final answer as a string formatted such "
                "that it could later be evaluated using sympy. "
                "Only print this expression, without additional " 
                "text. Do not wrap it in parentheses, only a symbolic "
                "expression in unformatted plain text. Use underscores for "
                "all subscripted variables \n\n" + textual_answer
    )
    sp_expr = sp_expr.strip("`'\n\"")
    if sp_expr.startswith('sympy\n'):
        sp_expr = sp_expr[6:]
    if sp_expr.startswith('python\n'):
        sp_expr = sp_expr[7:]
    if sp_expr.startswith('plaintext\n'):
        sp_expr = sp_expr[10:]
    return sp_expr


def fix_expr(expr, swap_funcs=True):
    if not expr.args:  # Leaf node
        if str(expr) == 'E':
            # E is always e
            return sp.E
        return expr
    
    # False identification of var as string
    if str(expr.func) in used_vars:
        if swap_funcs:
            if len(expr.args) != 1:
                raise Exception('WTF')
            return sp.core.mul.Mul(
                var_mapping[str(expr.func)],
                fix_expr(expr.args[0], swap_funcs)
            )
        else:
            raise LatexFuncCallError(
                'Sympy identified a variable as a function, '
                'leaving it for LLM to fix'
            )
    if str(expr.func) in KNOWN_FUNCTIONS:
        return KNOWN_FUNCTIONS[str(expr.func)](
            *[fix_expr(arg, swap_funcs) for arg in expr.args]
        )
    if isinstance(expr.func, sp.core.function.UndefinedFunction):
        raise LatexFuncCallError(
            'Sympy identified a function as an undefined function, '
            'leaving it for LLM to fix')

    return expr.func(*[fix_expr(arg, swap_funcs) for arg in expr.args])
