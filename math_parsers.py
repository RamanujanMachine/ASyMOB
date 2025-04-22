from sp_vars import *
import sympy as sp
import pandas as pd
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from openai_interface import OpenAIInterface
from gemini_interface import GeminiInterface

SYMPY_CONVERTER = GeminiInterface("gemini-2.0-flash")


def remove_equality(sp_expr):
    try:
        if isinstance(sp_expr, sp.Equality):
            return sp_expr.rhs
        return sp_expr
    except Exception as e:
        print(sp_expr)
        raise e


def parse_sympy_str(expr_str):
    try:
        if pd.isna(expr_str):
            return None
        return remove_equality(sp.parse_expr(
            expr_str, 
            local_dict=var_mapping, 
            transformations=(
                standard_transformations + (implicit_multiplication_application,)
            )
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
        return remove_equality(fix_expr(parse_latex(latex_str)))
    except Exception as e:
        print('Error parsing latex string:')
        print(latex_str)
        print(e)
        return pd.NA


def latex_to_sympy_llm(latex_str):
    """
    Convert a LaTeX string to a sympy expression, using the LLM parser.
    """
    function_def = SYMPY_CONVERTER.send_message(
        "Following is a mathematical expression written in latex. Please " 
        "create a python function that recreates this expression using " 
        "sympy, exactly as it is written in the expression given. The " 
        "function's signature will be `solution(x, n, k, A, B, C, D, E, F)` "
        "and it will return the expression as a sympy object. The " 
        "implementation will be identical to the expression given. You will " 
        "not use any form of simplification or modification for the " 
        "mathematical expression. If the answer is not dependent on one "
        "or more of the parameters accepted it may not use them. Only print "
        "the function, without additional text. Assume all necessary "
        "parameters  and symbols already exists. Access sympy functions "
        "through the sp module (e.g. sp.hyper). Before returning the function"
        "run your code to make sure that your function is valid\n\n" + latex_str
        )
    
    print(function_def)
    function_def = function_def.strip("`\n")
    if function_def.startswith('python\n'):
        function_def = function_def[6:]

    exec(function_def, globals())
    
    # Function solution defined dynamically
    return solution(
        x, n, k, A, B, C, D, E, F
    )


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


def fix_expr(expr,):
    if not expr.args:  # Leaf node
        if str(expr) == 'E':
            # If not, it will often return the constant e
            return E
        return expr
    
    # False identification of var as string
    if str(expr.func) in used_vars:
        if len(expr.args) != 1:
            raise Exception('WTF')
        return sp.core.mul.Mul(
            var_mapping[str(expr.func)],
            fix_expr(expr.args[0])
            )
    
    if str(expr.func) == 'O':
        return sp.O(
            *[fix_expr(arg) for arg in expr.args]
        )

    return expr.func(*[fix_expr(arg) for arg in expr.args])
