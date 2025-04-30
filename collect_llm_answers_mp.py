from models import MODELS, get_model
import multiprocessing as mp
import pandas as pd
from collect_llm_answers import load_questions, ask_model
import time

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

        return result
    except Exception as e:
        print(f"Error processing question {q_id} with model {model_name}: {e}")
        return {}


def collect_single_question(q_id, question_text, true_answer):
    for model_name, model in MODELS.items():
        if not model.support_code():
            llm_survey_wrapper(
                q_id, 
                question_text, 
                true_answer, 
                model_name, 
                None)
        else:
            for code_execution in [True, False]:
                llm_survey_wrapper(
                    q_id, 
                    question_text, 
                    true_answer, 
                    model_name, 
                    code_execution)

def main():
    """
    Main function to run the multiprocessing.
    """
    questions = load_questions()
    args = []
    for q_id, question_text, true_answer in questions:
        args.append((q_id, question_text, true_answer))

    print(f"Number of tasks: {len(args)}")

    start = time.time()
    # Create a pool of workers
    with mp.Pool(processes=10) as pool:
        # Map the function to the pool
        results = pool.starmap(collect_single_question, args)

    # Process the results
    df = pd.DataFrame.from_records(results)
    df.to_excel(
        'results_mp.xlsx', 
        index=False, 
        sheet_name='results'
    )
    print(f'that took {time.time() - start}')
    df.to_pickle(
        'results.pkl', 
    )


if __name__ == '__main__':
    main()
