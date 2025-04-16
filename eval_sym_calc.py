from llm_interface import send_and_receive_message
from models import MODELS
import json
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import pandas as pd


QUESTIONS_PATH = 'questions.json'
SYMPY_CONVERTER_MODEL = 'openai/gpt-4o'# 'gemini/gemini-2.5-pro-exp-03-25'
NO_CODE_PREFIX = (
    "Assume you don't have access to a computer: do not use "
    "code, solve this manually - using your internal reasoning.\n"
)
USE_CODE_PREFIX = (
    "Please use Python to solve the following question. Don't show it, "
    "just run it internally.\n"
)

def parse_sympy_expr_no_simplify(expr_str):
    # Use transformations that avoid implicit simplification
    transformations = (
        standard_transformations + 
        (implicit_multiplication_application,)
    )
    
    # Parse without simplifying, but still recognizing variables and constants
    expr = parse_expr(expr_str, transformations=transformations, 
                      evaluate=False)
    
    return expr


def load_questions(path=QUESTIONS_PATH):
    with open(path, 'r') as f:
        questions = json.load(f)
    # Convert answers to sympy objects
    for i in range(len(questions)):
        questions[i][2] = parse_sympy_expr_no_simplify(questions[i][2])
    return questions
    

def format_final_answer_to_sympy(textual_answer, converter_model):
    sp_expr = send_and_receive_message(
        model=converter_model,
        message="Following is an answer to a mathematical question. "
                "Please write the final answer as a string formatted such "
                "that it could later be evaluated using sympy. "
                "Only print this expression, without additional " 
                "text. Do not wrap it in parentheses, only a symbolic "
                "expression in unformatted plain text.\n\n" + textual_answer
    )
    sp_expr = sp_expr.strip("`'\n\"")
    if sp_expr.startswith('sympy\n'):
        sp_expr = sp_expr[6:]
    if sp_expr.startswith('python\n'):
        sp_expr = sp_expr[7:]
    if sp_expr.startswith('plaintext\n'):
        sp_expr = sp_expr[10:]
    return sp_expr


def ask_model(model, question_text, sympy_converter_model=SYMPY_CONVERTER_MODEL):
    
    """
    Ask the model a question and get the answer as a sympy expression.

    Returns a sympy object extracted from the textual answer, the sympy 
    expression as a string and the textual answer itself.
    """
    textual_answer = send_and_receive_message(
        model=model,
        message=question_text
    )
    # print(f"Textual answer: {textual_answer}")
    sp_expr = format_final_answer_to_sympy(textual_answer, sympy_converter_model)
    try:
        sp_obj = parse_sympy_expr_no_simplify(sp_expr)
    except Exception as e:
        print(f"Error parsing sympy expression: {sp_expr}")
        return -1, sp_expr, textual_answer
    return sp_obj, sp_expr, textual_answer


def symbolic_comparison(A, B):
    """
    Compare two sympy expressions A and B.

    Returns True if they are equal, False otherwise.
    """
    try:
        return sp.simplify(A - B) == 0
    except Exception as e:
        print(f"Error comparing expressions: {A}, {B}")
        return None

def assess_model(model, question_text, true_answer):
    """
    Assess the model's performance on a given question.

    Returns a dictionary with the model's answer, the true answer, and 
    whether they are equal.
    """
    try:
        sp_obj, sp_expr, textual_answer = ask_model(model, question_text)
        print(f'Answer: {sp_expr}')
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
    for q_id, question_text, true_answer in questions:
        for model in MODELS[:1]:
            print(f"Model: {model}, Question: {question_text}")

            # Use python
            results.append(
                {
                    'question_id': q_id,
                    'model': model,
                    'use_python': True,
                    **assess_model(
                        model, 
                        USE_CODE_PREFIX + question_text, 
                        true_answer)
                })
            
            # No python
            results.append(
                {
                    'question_id': q_id,
                    'model': model,
                    'use_python': False,
                    **assess_model(
                        model, 
                        NO_CODE_PREFIX + question_text, 
                        true_answer)
                })
                
    df = pd.DataFrame.from_records(results)
    df.to_excel(
        'results.xlsx', 
        index=False, 
        sheet_name='results'
    )


if __name__ == "__main__":
    main()