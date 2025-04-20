from sp_vars import *
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from openai_interface import OpenAIInterface

SYMPY_CONVERTER = OpenAIInterface("gpt-4o")


def parse_sympy_str(expr_str):
    return sp.parse_expr(
        expr_str, 
        local_dict=var_mapping, 
        transformations=(
            standard_transformations + (implicit_multiplication_application,)
        )
    )


def latex_to_sympy_deter(latex_str):
    """
    Convert a LaTeX string to a sympy expression, using the sympy parser.
    """
    return parse_latex(latex_str)


def latex_to_sympy_llm(latex_str):
    """
    Convert a LaTeX string to a sympy expression, using the LLM parser.
    """
    function_def = SYMPY_CONVERTER.send_message(
        "Following is a mathematical expression written in latex. "
        "Please create a python function that implements the final "
        "answer, using sympy. The function signature will be "
        "`solution(x, n, A, B, C, D, E, F)` and it will return the "
        "final answer as a sympy object. The implementation will be identical " 
        "to the expression given. You will not use any form of simplification " 
        "to return the answer in sympy. If the answer is not dependent on one "
        "or more of the parameters accepted it may not use them. Only print " 
        "the function, without imports in plain text. Assume all necessary " 
        "parameters  and symbols already exists. Access sympy functions "
        "through the sp module (e.g. sp.hyper)\n\n" + latex_str
        )
    
    print(function_def)
    function_def = function_def.strip("`\n")
    if function_def.startswith('python\n'):
        function_def = function_def[6:]

    exec(function_def, globals())
    
    # Function solution defined dynamically
    return solution(
        x, n, A, B, C, D, E, F
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

