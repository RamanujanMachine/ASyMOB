from models import MODELS_GENERATORS
import pandas as pd
from sp_vars import *
import re
from db_utils import load_tasks, insert_row, get_connection
import traceback
import multiprocessing as mp
from pathlib import Path
import json


RETRY_ATTEMPT = True

# # prompt 2:
# MATH_INSTRUCTIONS = (
#     'Answer the question. End with: "The final answer is:" followed by a '
#     'single LaTeX expression on a new line, wrapped in $$.\n'
# )

# NO_CODE_PREFIX = (
#     "Assume you don’t have access to a computer. Do not use code.\n"
# )

# USE_CODE_PREFIX = (
#     "You may use your internal Python tool to solve this.\n"
# )
# print(MATH_INSTRUCTIONS)
# print(NO_CODE_PREFIX)
# print(USE_CODE_PREFIX)

# ChatGPT suggested prompt (prompt 1)
MATH_INSTRUCTIONS = (
    'Answer the following question. End with: "The final answer is:" followed '
    'by a single LaTeX expression on a new line, wrapped in $$.\n'
    'Do not split the answer into multiple terms or add text after the LaTeX.\n'
)
NO_CODE_PREFIX = (
    "Assume you don't have access to a computer, and do not use code to solve "
    "the question.\n"
)
USE_CODE_PREFIX = ( ''
    "Use Python tools to assist you solving this problem.\n"
)

# Original 
# MATH_INSTRUCTIONS = (
#     'Finish your answer by writing "The final answer is:" and then the '
#     'answer in latex in a new line. Write the answer as a single expression. '
#     'Do not split your answer to different terms. Use $$ to wrap over the '
#     'latex text. Do not write anything after the latex answer.\n'
# )
# NO_CODE_PREFIX = (
#     "Assume you don't have access to a computer: do not use "
#     "code, solve this manually - using your internal reasoning.\n"
# )
# USE_CODE_PREFIX = (
#     "Please use Python to solve the following question. Don't show it, "
#     "just run it internally.\n"
# )
# All models that support responses API
MODELS_LIST = [
    # ('DeepSeek-Prover-V2-671B', None),
    ('DeepSeek-R1', None),
    ('DeepSeek-V3', None),
    ('gemini/gemini-2.0-flash', False),
    ('gemini/gemini-2.0-flash', True),
    ('gemini/gemini-2.5-flash-preview-04-17', False),
    ('gemini/gemini-2.5-flash-preview-04-17', True),
    # ('gemini/gemma-3-27b-it', None),
    # ('gpt-4.1', False),
    # ('gpt-4.1', True),
    # ('gpt-4o', False),
    # ('gpt-4o', True),
    # ('gpt-4o-mini', False),
    # ('gpt-4o-mini', True),
    ('o4-mini', False),
    ('o4-mini', True),
    ('o3', False),
    ('o3', True),
    # ('meta-llama/Llama-4-Scout-17B-16E-Instruct', None),
    # ('nvidia/Llama-3_3-Nemotron-Super-49B-v1', None),
    # ('Qwen/Qwen2.5-72B-Instruct', None),
]
# CHUNKS_DIR = Path('LLM_survey_chunks_QWEN3')


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
    
    if '' in matches:
        matches.remove('')

    if not matches:
        # escalate - just look for the last boxed{.*}
        matches = re.findall(
            r'\\boxed\{(.*?)\}' + '(?:\n|$|")',
            textual_answer, 
            re.DOTALL)
        
    if not matches:
        # escalate harder - just look for the last $$(.*?)$$
        matches = re.findall(
            r'\$\$(.*?)\$\$',
            textual_answer, 
            re.DOTALL)

    if not matches:
        matches = re.findall(
            r"output='(.*?)'",
            textual_answer, 
            re.DOTALL)

    if not matches:
        matches = re.findall(
            r'output="(.*?)"',
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


def ask_model(model_name, question_text, code_execution):
    """
    Ask the model a question and get the answer as a sympy expression.

    Returns a sympy object extracted from the textual answer, the sympy 
    expression as a string and the textual answer itself.
    """
    prompt = MATH_INSTRUCTIONS + _incentivize_code_execution(
        question_text, 
        use_code=code_execution
    )
    if model_name not in MODELS_GENERATORS:
        return {
            'prompt': prompt,
            'error': f"Model {model_name} not found."
        }
    model = MODELS_GENERATORS[model_name]()
    try:
        textual_answer, tokens, code_attempted = model.send_message(
            message=prompt,
            code_execution=code_execution,
            return_extra=True
        )
    except Exception as e:
        print(f"Error sending message to {model_name}: {e}")
        print('message:', MATH_INSTRUCTIONS + question_text)
        return {'prompt': prompt, 'error': str(e)}
    
    result = {
        'prompt': prompt,
        'full_answer': textual_answer,
        'tokens_used': tokens,
        'code_attempted': code_attempted
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
    
    return result


def upload_result_to_db(conn, task, result, acquisition_time):
    added_row = {
        'challenge_id': task.get('challenge_id'),
        'model': task.get('model'),
        'code_execution': task.get('code_execution'),
        'prompt': result.get('prompt'),
        'full_answer': result.get('full_answer'),
        'tokens_used': result.get('tokens_used'),
        'final_answer_latex': result.get('final_answer_latex'),
        'error': result.get('error', None),
        'acquisition_time': acquisition_time,
        'acquisition_method': 'Responses/completion API',
        'code_attempted': result.get('code_attempted', None)
    }
    # Uncomment the following lines to insert the row into a file
    # output_filename = (
    #     f'sample_data/result_{added_row["model"]}_'
    #     f'{added_row["challenge_id"]}_{added_row["code_execution"]}'
    #     '.json'
    # )
    # with open(output_filename, 'w') as f:
    #     json.dump(added_row, f)

    # Comment out the following line to insert the row into a file
    insert_row(
        conn=conn,  # Assuming you have a connection object
        table_name='model_responses',
        data_dict=added_row
    )


def llm_survey_wrapper(task, acquisition_time):
    """
    Wrapper function to call the LLM and process the result.
    """
    q_id = task['challenge_id']
    model = task['model']
    code_execution=task['code_execution']
    indicator = Path(f'collection_state/{q_id}_{model.replace("/", '_')}_{code_execution}.txt')
    indicator.touch()
    try:
        print(f"Processing question {q_id} ({code_execution})"
              f"with model {model}...")
        result = ask_model(
            model, 
            task['challenge'], 
            code_execution)
        
    except Exception as e:
        print(f"Error processing question {q_id} "
              f"with model {model}: {e}")
        result = {'error': traceback.format_exc()}

    print('got answer, pushing to db')
    with get_connection() as conn:
        upload_result_to_db(
            conn, 
            task, 
            result, 
            acquisition_time
        )
    indicator.unlink()


def main():
    acquisition_time = '2025-05-21 19:00:00.000000'
    tasks_df = load_tasks(
        models=MODELS_LIST, 
        # sql_filter="challenge_id <= 17092 and error is null",
        sql_filter="variation in ('Original', 'Numeric-All-0', 'Equivalence-One-Easy')",
        retry_errors=True
    )
    # Comment the command above and uncomment the following line to use tasks
    # from a sample file 
    # tasks_df = pd.read_pickle('sample_data\\sample_collect_answers.pkl')
    tasks_df = tasks_df.sample(frac=1).reset_index(drop=True)

    args = [
        (task.to_dict(), acquisition_time) 
        for _, task in tasks_df.iterrows()
    ]
    print(f"Number of tasks: {len(args)}")
    with mp.Pool(processes=30) as pool:
        # Map the function to the pool
        # results = pool.starmap(collect_single_question, args)
        pool.starmap(llm_survey_wrapper, args)


if __name__ == "__main__":
    main()