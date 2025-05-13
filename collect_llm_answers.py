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
from db_utils import load_questions


RETRY_ATTEMPT = True

MATH_INSTRUCTIONS = (
    'Finish your answer by writing "The final answer is:" and then the '
    'answer in latex in a new line. Write the answer as a single expression. '
    'Do not split your answer to different terms. Use $$ to wrap over the '
    'latex text. Do not write anything after the latex answer.\n'
)
NO_CODE_PREFIX = (
    "Assume you don't have access to a computer: do not use "
    "code, solve this manually - using your internal reasoning.\n"
)
USE_CODE_PREFIX = (
    "Please use Python to solve the following question. Don't show it, "
    "just run it internally.\n"
)



def _incentivize_code_execution(message, use_code=True):
    """
    Modify the message to incentivize code execution.
    
    Args:
        message (str): The original message.
    
    Returns:
        str: The modified message.
    """
    if use_code is None:
        return message
    if use_code:
        return USE_CODE_PREFIX + message
    else:
        return NO_CODE_PREFIX + message


def extract_latex_answer(textual_answer):
    """
    Extract the latex answer from the textual answer.
    """
    # Find the last occurrence of "The final answer is:"
    # Then use different parentheses to options to wrap the answer.
    # The reges will not consume the parentheses, but will consume the text 
    # inside.
    matches = re.findall(
        r'\**[Tt]he final answer is:?\**\s*'
        r'(?:(?:\\\()|(?:\\\[)|(?:\$+))'
        r'(.*?)'
        r'(?:(?:\\\))|(?:\\\])|(?:\$+))',
        textual_answer, 
        re.DOTALL)
    
    if not matches:
        # escalate - just look for the last boxed{.*}
        matches = re.findall(
            r'\\boxed\{(.*?)\}' + '(?:\n|$)',
            textual_answer, 
            re.DOTALL)
        
    if not matches:
        # escalate harder - just look for the last $$(.*?)$$
        matches = re.findall(
            r'\$\$(.*?)\$\$',
            textual_answer, 
            re.DOTALL)
        
    if not matches:
        raise ValueError("No latex answer found in the textual answer.")
    
    latex_answer = matches[-1].strip()

    # clean up the latex answer
    latex_answer = latex_answer.replace(r'\displaystyle', '')
    latex_answer = latex_answer.replace(r'\dots', '')
    if latex_answer.startswith(r'\boxed{'):
        latex_answer = latex_answer[7:-1].strip()
    
    return latex_answer


def ask_model(model, question_text, code_execution):
    """
    Ask the model a question and get the answer as a sympy expression.

    Returns a sympy object extracted from the textual answer, the sympy 
    expression as a string and the textual answer itself.
    """
    prompt = _incentivize_code_execution(
        MATH_INSTRUCTIONS + question_text, 
        use_code=code_execution
    )
    try:
        textual_answer, tokens = model.send_message(
            message=prompt,
            code_execution=code_execution,
            return_tokens=True
        )
    except Exception as e:
        print(f"Error sending message to model: {e}")
        print('message:', MATH_INSTRUCTIONS + question_text)
        return {'prompt': prompt, 'error': str(e)}
    
    result = {
        'prompt': prompt,
        'full_answer': textual_answer,
        'tokens_used': tokens
    }
    try:
        # extract the final answer from the textual answer
        latex_answer = extract_latex_answer(textual_answer)
        result['final_answer_latex'] = latex_answer
                
    except Exception as e:
        print(f"Error processing the answer: {e}")
        print(f"Textual answer: {textual_answer}")
        result['final_answer_latex'] = None
        result['error'] = str(e)
        # result['sp_deter'] = None if 'sp_deter' not in result else result['sp_deter']
        # result['sp_llm'] = None if 'sp_llm' not in result else result['sp_llm']
    
    return result


def enumerate_tasks_configurations(
        retry_attempt=RETRY_ATTEMPT, prev_res_file='results_mp.xlsx'):
    questions_df = load_questions()
    tasks = []

    # The loops order is important. 
    # We might hit rate limits if we ask the same model too many things
    # in parallel. Setting the fast changing item to be the model, leads
    # to multiple different models running together, which reduces the 
    # per-model load.
    questions_df.sort_values(['challenge_id', 'model'], inplace=True)
    for q_id, question_text, true_answer in questions:
        for model_name, model in MODELS.items():
            if not model.support_code():
                tasks.append({
                    'question_id': q_id,
                    'model': model_name,
                    'question_text': question_text,
                    'true_answer': true_answer,
                    'code_execution':None
                })
            else:
                tasks.append({
                    'question_id': q_id,
                    'model': model_name,
                    'question_text': question_text,
                    'true_answer': true_answer,
                    'code_execution': True
                })
                tasks.append({
                    'question_id': q_id,
                    'model': model_name,
                    'question_text': question_text,
                    'true_answer': true_answer,
                    'code_execution': False
                })

    if not retry_attempt:
        return tasks
    
    # removing already succeeded tasks
    print(f'filtering {len(tasks)} tasks...')
    prev_results = pd.read_excel(prev_res_file)
    already_succeeded_code_runners = prev_results[
        ~prev_results.full_answer.isna() & ~prev_results.code_execution.isna()
        ][['question_id', 'model', 'code_execution']].values.tolist()
    already_succeeded_not_code_runners = prev_results[
        ~prev_results.full_answer.isna() & prev_results.code_execution.isna()
        ][['question_id', 'model']].values.tolist()

    filtered_tasks = []
    for task in tasks:
        q_id = int(task['question_id'])
        model_name = task['model']
        code_execution = task['code_execution']

        # Code runners
        if ((not pd.isna(code_execution)) and 
            [q_id, model_name, float(code_execution)] in already_succeeded_code_runners):
            continue 

        # Not code runners
        if (pd.isna(code_execution) and 
            [q_id, model_name] in already_succeeded_not_code_runners):
            continue 

        filtered_tasks.append(task)
    print(f'Resulted in {len(filtered_tasks)} tasks!')
    return filtered_tasks


def main():
    results = []
    for task in enumerate_tasks_configurations():
        results.append({
            **task,
            **ask_model(
                task['model'], 
                task['question_text'], 
                task['code_execution'])
        })

    df = pd.DataFrame.from_records(results)
    df.to_excel(
        'results.xlsx', 
        index=False, 
        sheet_name='results'
    )
    df.to_pickle(
        'results.pkl', 
        index=False, 
    )


if __name__ == "__main__":
    main()