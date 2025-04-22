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
SYMPY_CONVERTER = OpenAIInterface("gpt-4o")
MATH_INSTRUCTIONS = (
    'Finish your answer by writing "The final answer is:" and then the '
    'answer in latex in a new line. Write the answer as a single expression. '
    'Do not split your answer to different terms. Use $$ to wrap over the '
    'latex text. Do not write anything after the latex answer.\n'
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
    # Then use different parentheses to options to wrap the answer.
    # The reges will not consume the parentheses, but will consume the text 
    # inside.
    matches = re.findall(
        r'[Tt]he final answer is:?\s*'
        r'(?:(?:\\\()|(?:\\\[)|(?:\$\$))'
        r'(.*?)'
        r'(?:(?:\\\))|(?:\\\])|(?:\$\$))',
        textual_answer, 
        re.DOTALL)
    if not matches:
        # escalate - just look for the last boxed{.*}
        matches = re.findall(
            r'\\boxed\{(.*?)\}' + '(?:\n|$)',
            textual_answer, 
            re.DOTALL)
    
    if not matches:
        raise ValueError("No latex answer found in the textual answer.")
    
    latex_answer = matches[-1].strip()

    # clean up the latex answer
    latex_answer = latex_answer.replace(r'\displaystyle', '')
    latex_answer = latex_answer.replace(r'\dots', '')
    if latex_answer.startswith(r'\boxed{'):
        latex_answer = latex_answer[8:-1].strip()
    
    return latex_answer


def ask_model(model, question_text, code_execution):
    """
    Ask the model a question and get the answer as a sympy expression.

    Returns a sympy object extracted from the textual answer, the sympy 
    expression as a string and the textual answer itself.
    """
    try:
        textual_answer = model.send_message(
            message=MATH_INSTRUCTIONS + question_text,
            code_execution=code_execution,
        )
    except Exception as e:
        print(f"Error sending message to model: {e}")
        print('message:', MATH_INSTRUCTIONS + question_text)
        return {}
    
    result = {
        'full_answer': textual_answer,
    }
    try:
        # extract the final answer from the textual answer
        latex_answer = extract_latex_answer(textual_answer)
        result['final_answer_latex'] = latex_answer
        
        # convert the latex answer to a sympy expression
        result['sp_deter'] = math_parsers.latex_to_sympy_deter(latex_answer)
        # result['sp_llm'] = math_parsers.latex_to_sympy_llm(latex_answer)
        result['sp_llm'] = None
    except Exception as e:
        print(f"Error processing the answer: {e}")
        print(f"Textual answer: {textual_answer}")
        result['sp_deter'] = None if 'sp_deter' not in result else result['sp_deter']
        result['sp_llm'] = None if 'sp_llm' not in result else result['sp_llm']
    
    return result

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