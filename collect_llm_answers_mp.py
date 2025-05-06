from models import MODELS, get_model
import multiprocessing as mp
import pandas as pd
from collect_llm_answers import ask_model, enumerate_tasks_configurations
import time
from pathlib import Path

CHUNKS_DIR = Path('LLM_survey_chunks')


def llm_survey_wrapper(q_id, question_text, true_answer, model_name, code_execution):
    """
    Wrapper function to call the LLM and process the result.
    """
    try:
        print(f"Processing question {q_id} with model {model_name}...")
        model = get_model(model_name) # MODELS[model_name]
        result = ask_model(model, question_text, code_execution=code_execution)
        result['question_id'] = q_id
        result['model'] = model_name
        result['true_answer'] = true_answer
        result['question_text'] = question_text
        result['code_execution'] = code_execution

    except Exception as e:
        print(f"Error processing question {q_id} with model {model_name}: {e}")
        result = {
            'question_id': q_id,
            'model': model_name,
            'true_answer': true_answer,
            'question_text': question_text,
            'code_execution': code_execution,
            'error': str(e)
        }

    df = pd.DataFrame.from_records([result])
    df.to_excel(
        CHUNKS_DIR / f'{q_id}_{model_name.replace('/', '-')}_{code_execution}.xlsx', 
        index=False, 
        sheet_name='results'
    )

    return result


def collect_single_question(q_id, question_text, true_answer):
    results = []
    for model_name, model in MODELS.items():
        if not model.support_code():
            results.append(llm_survey_wrapper(
                q_id, 
                question_text, 
                true_answer, 
                model_name, 
                None))
        else:
            for code_execution in [True, False]:
                results.append(llm_survey_wrapper(
                    q_id, 
                    question_text, 
                    true_answer, 
                    model_name, 
                    code_execution))
    
    df = pd.DataFrame.from_records(results)
    df.to_excel(
        CHUNKS_DIR / f'results_chunk_q_{q_id}.xlsx', 
        index=False, 
        sheet_name='results'
    )

    return results
    

def main():
    args = []
    for task in enumerate_tasks_configurations():
        q_id = task['question_id']
        question_text = task['question_text']
        true_answer = task['true_answer']
        model_name = task['model']
        code_execution = task['code_execution']

        args.append((
            q_id, 
            question_text, 
            true_answer, 
            model_name, 
            code_execution))

    print(f"Number of tasks: {len(args)}")

    start = time.time()
    # Create a pool of workers
    with mp.Pool(processes=10) as pool:
        # Map the function to the pool
        # results = pool.starmap(collect_single_question, args)
        results = pool.starmap(llm_survey_wrapper, args)

    # Process the results
    df = pd.DataFrame.from_records(results)
    df.to_excel(
        'results_mp.xlsx', 
        index=False
    )
    print(f'that took {time.time() - start}')
    df.to_pickle(
        'results.pkl', 
    )


if __name__ == '__main__':
    main()
