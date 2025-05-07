from collect_llm_answers import enumerate_tasks_configurations, MATH_INSTRUCTIONS, extract_latex_answer, load_questions
from openai_interface import OpenAIInterface
import json
import pandas as pd
from pathlib import Path
from itertools import islice
import time 

OPENAI_FOLDER = Path('openai_batches')

TRACK_FILE = OPENAI_FOLDER / 'tracking.json'
COMPLETED_BATCHES = OPENAI_FOLDER / 'completed_batch.json'
FAILED_BATCHES = OPENAI_FOLDER / 'failed_batch.json'

BATCH_REQUEST_FOLDER = OPENAI_FOLDER / 'batch_requests'
REQUEST_FILE_FORMAT = 'openai_batch_requests_{model}_{i}.jsonl'

RESULTS_FOLDER = OPENAI_FOLDER / 'batch_results'
FAILED_REQUESTS_FOLDER = OPENAI_FOLDER / 'failed_requests'
COMPLETED_REQUESTS_FOLDER = OPENAI_FOLDER / 'completed_requests'

OPENAI_IFACE = OpenAIInterface(model='') # Used only to create the client
SUPPORTING_MODELS = ['gpt-4o', 'gpt-4.1', 'gpt-4o-mini']
BATCH_SIZE = 60
LOCAL_QUEUE_SIZE = 50
TIME_BETWEEN_UPDATES = 10

def _get_request_file_from_batch(batch):
    return BATCH_REQUEST_FOLDER / batch['metadata']['description'].split('\\')[-1]
def _write_to_json(data, json_path, append=True):
    if append:
        with open(json_path, 'rb') as f:
            old_data = json.load(f)
    else:
        old_data = []

    with open(json_path, 'w') as f:
        json.dump(old_data + data, f, indent=2)


def create_request(task, max_tokens=15_000):
    model_name = task['model'].split('/')[-1]
    code_execution = task['code_execution']
    custom_id = (
        f"{task['question_id']}_"
        f"{model_name}_"
        f"{code_execution}"
    )

    question_text = OPENAI_IFACE._incentivize_code_execution(
        MATH_INSTRUCTIONS + task['question_text'],
        use_code=code_execution
    )

    request = {
        "custom_id": custom_id, 
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": {
            "model": model_name, 
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_text}
            ],
            "max_tokens": max_tokens
        }
    }
    return json.dumps(request)


def create_batch_files(tasks):
    requests_lines = {model: [] for model in SUPPORTING_MODELS}
    for task in tasks:
        model = task['model'].split('/')[-1]
        if model not in SUPPORTING_MODELS:
            continue
        requests_lines[model].append(create_request(task))
    
    created_files = []
    for model in SUPPORTING_MODELS:
        for batch_id in range(len(requests_lines[model]) // BATCH_SIZE):
            start = batch_id * BATCH_SIZE
            end = (batch_id + 1) * BATCH_SIZE
            batch = requests_lines[model][start:end]

            batch_file = (
                BATCH_REQUEST_FOLDER / 
                REQUEST_FILE_FORMAT.format(i=batch_id, model=model)
            )
            with open(batch_file, 'w') as f:
                f.write('\n'.join(batch))
            created_files.append(batch_file)
    return created_files
    

def send_batches(batch_files, track_file=TRACK_FILE):
    batches_sent = []

    for batch_local_file in batch_files:
        batch_remote_file = OPENAI_IFACE.client.files.create(
            file=open(batch_local_file, "rb"),
            purpose="batch"
        )
        batch_remote_id = batch_remote_file.id
        batch_obj = OPENAI_IFACE.client.batches.create(
            input_file_id=batch_remote_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": batch_local_file.name
            }
        )
        batches_sent.append(batch_obj)

    _write_to_json([b.to_dict() for b in batches_sent], track_file, append=True)

    return batches_sent


def download_batch_content(remote_batch, target_folder=RESULTS_FOLDER):
    batch_response = OPENAI_IFACE.client.files.content(
        remote_batch.output_file_id)
    results = []
    for line in batch_response.iter_lines():
        raw_response = json.loads(line)
        question_descriptor = raw_response['custom_id']
        q_id, model_name, code_execution = question_descriptor.split('_')
        
        full_answer = raw_response['response']['body']['choices'][0]['message']['content']
        response = {
            'q_id': int(q_id),
            'model_name': model_name,
            'code_execution': code_execution,
            'full_answer': full_answer,
            'tokens_used': raw_response['response']['body']['usage']['total_tokens'],
        }
        try:
            response['final_answer_latex'] = extract_latex_answer(full_answer)
        except Exception as e:
            response['error'] = str(e)

        results.append(response)

    return pd.DataFrame.from_records(results)


def load_local_state(track_file=TRACK_FILE):
    with open(track_file, 'r') as f:
        tracked_batches = json.load(f)
    return tracked_batches


def update_state_from_remote(questions_df, track_file=TRACK_FILE):
    def _move_request(batch, target_folder):
        request_file = _get_request_file_from_batch(batch)
        target_location = target_folder / request_file.name
        if not request_file.is_file() and target_location.is_file():
            print('Weird, file already moved', request_file, target_location)
            return
        request_file.rename(target_location)

    last_state = load_local_state()
    updated_state = []
    rerun_list = []
    failed_list = []
    completed_list = []

    for batch in last_state:
        batch_descriptor = batch['metadata']['description'].split('\\')[-1]
        batch_descriptor = batch_descriptor[22:-6]
        remote_batch = OPENAI_IFACE.client.batches.retrieve(batch['id'])

        if remote_batch.status == 'completed':
            completed_list.append(remote_batch.to_dict())
            # Download and fix file
            result_df = download_batch_content(remote_batch)
            result_df = result_df.merge(
                right=questions_df,
                how='left', # Shouldn't matter, but this is safer
                on='q_id'
            )
            result_df.to_excel(RESULTS_FOLDER / (batch_descriptor + '.xlsx'))
            continue

        if remote_batch.status == 'failed':
            print(remote_batch.errors)
            if remote_batch.errors.data[0].message.startswith(
                'Enqueued token limit reached'):
                rerun_list.append(_get_request_file_from_batch(batch))
            else:
                failed_list.append(remote_batch.to_dict())
            continue

        # Still needs tracking (probably)
        updated_state.append(remote_batch.to_dict())

    print(f'Batch status: '
          f'{len(updated_state)} still running, '
          f'{len(completed_list)} completed, '
          f'{len(rerun_list)} needs rerunning '
          f'{len(failed_list)} failed')
    
    # move completed and failed requests
    for batch in completed_list:
        _move_request(batch, COMPLETED_REQUESTS_FOLDER)
    for batch in failed_list:
        _move_request(batch, FAILED_REQUESTS_FOLDER)

    # Update state files
    _write_to_json(updated_state, track_file, append=False)
    _write_to_json(failed_list, FAILED_BATCHES, append=True)
    _write_to_json(completed_list, COMPLETED_BATCHES, append=True)

    running_batches = [_get_request_file_from_batch(b) for b in updated_state]
    return running_batches, rerun_list


def collect_results(
        questions_df, tracked_batches, 
        completed_log=COMPLETED_BATCHES, target_folder=RESULTS_FOLDER, 
        failed_log=FAILED_BATCHES):
    
    
    updated_batches = []
    completed_batches = []
    failed_batches = []
    for batch in tracked_batches:
        # Remove the irrelevant stuff from the path
        batch_descriptor = batch['metadata']['description'][44:-6]
        remote_batch = OPENAI_IFACE.client.batches.retrieve(batch['id'])

        if remote_batch.status != 'completed':
            # Still needs tracking
            updated_batches.append(remote_batch.to_dict())
            continue

        if remote_batch.status == 'failed':
            failed_batches.append(remote_batch.to_dict())
            continue

        result_df = download_batch_content(remote_batch)
        result_df = result_df.merge(
            right=questions_df,
            how='left', # Shouldn't matter, but this is safer
            on='q_id'
        )
        result_df.to_excel(target_folder / (batch_descriptor + '.xlsx'))
        completed_batches.append(remote_batch.to_dict())

    _write_to_json(updated_batches, track_file, append=False)
    _write_to_json(completed_batches, completed_log)
    _write_to_json(failed_batches, failed_log)


def enumerate_new_batches(n_batches, ignore=[], model=None):
    new_batch_files = []
    for file in BATCH_REQUEST_FOLDER.iterdir():
        if model is not None and model not in file.name:
            continue
        if file not in ignore:
            new_batch_files.append(file)
        
        if len(new_batch_files) == n_batches:
            return new_batch_files
        
    # No more files
    return new_batch_files


def main():
    # on first iteration:
    # tasks = enumerate_tasks_configurations()
    # batch_files = create_batch_files(tasks)
    
    questions = load_questions(parse_sympy=False)
    questions_df = pd.DataFrame.from_records(
        questions,
        columns=['q_id', 'question', 'true_answer'],
        index='q_id')
    
    # Load the batches currently running, to initialize the local queue
    live_batches = load_local_state()
    print(f'There were {len(live_batches)} batches running')
    while True:
        time.sleep(TIME_BETWEEN_UPDATES)
        live_batches, rerun_batches = update_state_from_remote(questions_df)
        n_new_batches = LOCAL_QUEUE_SIZE - len(rerun_batches) - len(live_batches)
        if n_new_batches <= 0:
            continue
        new_batches = enumerate_new_batches(
            n_new_batches, 
            ignore=live_batches+rerun_batches)
        batches_to_send = rerun_batches + new_batches
        print(f'Sending {n_new_batches} new batches')
        send_batches(batches_to_send)


if __name__ == '__main__':
    main()