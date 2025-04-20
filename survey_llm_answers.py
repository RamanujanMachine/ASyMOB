from llm_interface import GenericLLMInterface
from gemini_interface import GeminiInterface
from openai_interface import OpenAIInterface
import math_parsers
from models import MODELS
import json
import sympy as sp
import pandas as pd
from sp_vars import *
import re

QUESTIONS_PATH = 'questions.json'
# SYMPY_CONVERTER_MODEL = 'openai/gpt-4o'# 'gemini/gemini-2.5-pro-exp-03-25'
# SYMPY_CONVERTER = GeminiInterface("gemini-2.5-pro-exp-03-25") # ge(SYMPY_CONVERTER_MODEL)
SYMPY_CONVERTER = OpenAIInterface("o3-mini")
MATH_INSTRUCTIONS = (
    'Finish your answer by writing "The final answer is:" and then the '
    'answer in latex in a new line. use $$ to wrap over the latex text. '
    'Do not write anything after the latex answer.\n'
)
C = sp.symbols('C')


def load_questions(path=QUESTIONS_PATH):
    with open(path, 'r') as f:
        questions = json.load(f)
    # Convert answers to sympy objects
    parsed_questions = []
    for question in questions:
        q_id = question['Index']
        question_text = question['Challenge']
        sympy_str_answer = question['Answer in Sympy']

        true_answer = math_parsers.parse_sympy_str(sympy_str_answer)
        parsed_questions.append((q_id, question_text, true_answer))
    return parsed_questions
    

def extract_latex_answer(textual_answer):
    """
    Extract the latex answer from the textual answer.
    """
    # Find the last occurrence of "The final answer is:"
    match = re.search(
        r'[Tt]he final answer is:\s*\$\$(.*)\$\$', 
        textual_answer, 
        re.DOTALL)
    if match:
        latex_answer = match.group(1).strip()
        return latex_answer
    else:
        raise ValueError("No latex answer found in the textual answer.")


def ask_model(model, question_text, code_execution):
    """
    Ask the model a question and get the answer as a sympy expression.

    Returns a sympy object extracted from the textual answer, the sympy 
    expression as a string and the textual answer itself.
    """
    textual_answer = model.send_message(
        message=MATH_INSTRUCTIONS + question_text,
        code_execution=code_execution,
    )
    try:
        # extract the final answer from the textual answer
        latex_answer = extract_latex_answer(textual_answer)
        
        # convert the latex answer to a sympy expression
        sp_deter = math_parsers.latex_to_sympy_deter(latex_answer)
        sp_llm = math_parsers.latex_to_sympy_llm(latex_answer)
    except Exception as e:
        print(f"Error processing the answer: {e}")
        print(f"Textual answer: {textual_answer}")
        sp_deter = None
        sp_llm = None
        latex_answer = None
    
    return {
        'full_answer': textual_answer,
        'final_answer_latex': latex_answer,
        'sp_deter': sp_deter,
        'sp_llm': sp_llm,
    }


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


def main():
    questions = load_questions()
    results = []
    for q_id, question_text, true_answer in questions:
        for model_name, model in MODELS.items():
            print(f"Model: {model_name}, Question: {question_text}")

            for code_execution in [True, False]:
                print(f"Code execution: {code_execution}")
                results.append(
                    {
                        'question_id': q_id,
                        'model': model_name,
                        'question_text': question_text,
                        'code_execution': code_execution,
                        'true_answer': true_answer,
                        **ask_model(
                            model, 
                            question_text, 
                            code_execution=code_execution)
                    })

    df = pd.DataFrame.from_records(results)
    df.to_excel(
        'results.xlsx', 
        index=False, 
        sheet_name='results'
    )


if __name__ == "__main__":
    main()