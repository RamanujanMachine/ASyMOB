from llm_interface import GenericLLMInterface
from gemini_interface import GeminiInterface
from openai_interface import OpenAIInterface
from models import MODELS
import json
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import pandas as pd
from sp_vars import *


QUESTIONS_PATH = 'questions.json'
# SYMPY_CONVERTER_MODEL = 'openai/gpt-4o'# 'gemini/gemini-2.5-pro-exp-03-25'
# SYMPY_CONVERTER = GeminiInterface("gemini-2.5-pro-exp-03-25") # ge(SYMPY_CONVERTER_MODEL)
SYMPY_CONVERTER = OpenAIInterface("o3-mini")

C = sp.symbols('C')


def parse_sympy_expr_no_simplify(expr_str):
    return sp.parse_expr(
        expr_str, 
        local_dict=var_mapping, 
        transformations=(
            standard_transformations + (implicit_multiplication_application,)
        )
    )


def load_questions(path=QUESTIONS_PATH):
    with open(path, 'r') as f:
        questions = json.load(f)
    # Convert answers to sympy objects
    for i in range(len(questions)):
        questions[i][2] = parse_sympy_expr_no_simplify(questions[i][2])
    return questions
    

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


def generate_sympy_obj(textual_answer):
    """
    Generate a sympy object from the final answer string.

    Returns a sympy object.
    """
    function_def = SYMPY_CONVERTER.send_message(
        message="Following is an answer to a mathematical question. "
                "Please create a python function that implements the final "
                "answer, using sympy. The function signature will be "
                "`solution(x, n, A, B, C, D, E, F)` and it will return the "
                "final answer as a sympy object. If the answer is not "
                "dependent on one or more of the parameters accepted it may "
                "not use them. The function will not "
                "simplify the result. Only print the function, without "
                "imports in plain text. Assume all necessary parameters "
                "and symbols already exists. Access sympy functions "
                "through the sp module (e.g. sp.hyper)\n\n" + textual_answer,
        code_execution=None
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
    
def ask_model(model, question_text, code_execution):
    
    """
    Ask the model a question and get the answer as a sympy expression.

    Returns a sympy object extracted from the textual answer, the sympy 
    expression as a string and the textual answer itself.
    """
    textual_answer = model.send_message(
        message=question_text,
        code_execution=code_execution,
    )
    # print(f"Textual answer: {textual_answer}")
    sp_expr = format_final_answer_to_sympy(textual_answer)
    try:
        sp_obj = generate_sympy_obj(textual_answer)
    except Exception as e:
        print(f"Error parsing sympy expression: {sp_expr}:\n{e}")
        return -1, sp_expr, textual_answer
    return sp_obj, sp_expr, textual_answer


def symbolic_comparison(A, B):
    """
    Compare two sympy expressions A and B.

    Returns True if they are equal, False otherwise.
    """
    try:
        return sp.simplify(A - B) == 0 or sp.simplify(A - B) == C or \
                sp.simplify(B - A) == C
    except Exception as e:
        print(f"Error comparing expressions: {A}, {B}")
        return None

def assess_model(model, question_text, true_answer, code_execution):
    """
    Assess the model's performance on a given question.

    Returns a dictionary with the model's answer, the true answer, and 
    whether they are equal.
    """
    try:
        sp_obj, sp_expr, textual_answer = ask_model(model, question_text, code_execution)
        print(f'Answer: {sp_expr}, {sp_obj}')
        symbolic_equal = symbolic_comparison(sp_obj, true_answer)
        return {
            'question_text': question_text,
            'true_answer': true_answer,
            'sp_obj': sp_obj,
            'sp_expr': sp_expr,
            'textual_answer': textual_answer,
            'symbolic_comparison': symbolic_equal,
        }
    except Exception as e:
        print(f"Error processing question '{question_text}' "
              f"with model '{model}': {e}")
        return {
            'model': model,
            'question_text': question_text,
            'true_answer': true_answer,
            'error': str(e),
        }

def main():
    questions = load_questions()
    results = []
    for q_id, question_text, true_answer in questions[:4]:
        for model_name, model in MODELS.items():
            print(f"Model: {model_name}, Question: {question_text}")

            # Use python
            results.append(
                {
                    'question_id': q_id,
                    'model': model_name,
                    'use_python': True,
                    **assess_model(
                        model, 
                        question_text, 
                        true_answer,
                        code_execution=True)
                })
            
            # No python
            results.append(
                {
                    'question_id': q_id,
                    'model': model_name,
                    'use_python': False,
                    **assess_model(
                        model, 
                        question_text, 
                        true_answer,
                        code_execution=False)
                })
                
    df = pd.DataFrame.from_records(results)
    df.to_excel(
        'results.xlsx', 
        index=False, 
        sheet_name='results'
    )


if __name__ == "__main__":
    main()