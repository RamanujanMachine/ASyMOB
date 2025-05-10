from openai_interface import OpenAIInterface
from collect_llm_answers import MATH_INSTRUCTIONS, extract_latex_answer, load_questions
import random
import traceback
from pathlib import Path
import pandas as pd
import multiprocessing as mp
import time
import re

MODELS = ['gpt-4o'] # ], 'gpt-4.1', 'gpt-4o-mini']
OUTPUT_FOLDER = Path('assistant_outputs')
BATCH_SIZE = 1000
N_WORKERS = 5


class TerminalException(Exception):
    pass

# We only need the iface to get the client with the api key.
def send_message(client, assistant, message):
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        # instructions=''
    )
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    response = messages.data[0].content[0].text.value
    if run.status != 'completed':
        raise Exception(
            f"Run failed: {run.status}- {run.last_error.message}")

    return response, run.usage.total_tokens


def send_message_in_thread(thread, client, assistant, message):
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        # instructions=''
    )
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    response = messages.data[0].content[0].text.value

    # Delete the thread messages
    # prev_messages = messages.data
    # if len(prev_messages) != 3:
    #     raise TerminalException(
    #         f"Expected 3 messages, got {len(prev_messages)}")
    # for message in messages.data[:2]:
    #     client.beta.threads.messages.delete(
    #         thread_id=thread.id, message_id=message.id)
    
    if run.status != 'completed':
        raise Exception(
            f"Run failed: {run.status}- {run.last_error.message}")

    return response, run.usage.total_tokens


def calculator_test(iface, client, assistant, n_tries=10):
    correct = 0
    
    for i in range(n_tries):
        a = random.randint(int(1e10), int(1e20))
        b = random.randint(int(1e10), int(1e20))

        response_text = send_message(
            client, 
            assistant, 
            iface._incentivize_code_execution(
                MATH_INSTRUCTIONS + f"What is {a} * {b}.",
                use_code=True
            )
        )
        resp_latex = extract_latex_answer(response_text)
        resp = int(resp_latex)

        if resp == a * b:
            print(f"Correct: {a} * {b} = {resp}")
            correct += 1
        else:
            print(f"Incorrect: {a} * {b} = {resp}")
    print(f"Correct: {correct} / {n_tries} = {correct / n_tries:.2%}")
    return correct / n_tries


def test_models():
    iface = OpenAIInterface(model='gpt-4o')
    client = iface.client
    assistants = {}
    for model in MODELS:
        assistants[model] = client.beta.assistants.create(
        model=model,
        tools=[{"type": "code_interpreter"}]
    )

    for model, assistant in assistants.items():
        print(f"Testing {model}...")
        try:
            score = calculator_test(iface, client, assistant)
            print(f"Score: {score:.2%}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            client.beta.assistants.delete(assistant.id)


def handle_batch(questions_data, model_name, question_timeout=60):
    print(f"Processing batch with model {model_name}...")
    iface = OpenAIInterface(model=model_name)
    # format message
    joined_question = (
        'Here is a list of questions. You will answer them one-by-one without '
        'mixing them. Write "Answer %d:" before answering every questions.\n\n'
    )
    results = []
    for q_id, question_text, true_answer in questions_data:
        question = iface._incentivize_code_execution(
            MATH_INSTRUCTIONS + question_text,
            use_code=True
        )
        joined_question += f"Question {q_id}: {question}\n\n"
    
    # Interact with the model
    client = iface.client
    assistant = client.beta.assistants.create(
        model=model_name,
        tools=[{"type": "code_interpreter"}]
    )
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=joined_question.strip()
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # One message for each question plus our message
    desired_messages_count = len(questions_data) + 1
    start = time.time()
    while True:
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        if len(messages.data) >= desired_messages_count:
            break
        if time.time() - start > question_timeout * len(questions_data):
            print("Timeout waiting for model response.")
            break
        time.sleep(1)
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    total_tokens = run.usage.total_tokens
    
    # Parse the results
    q_id_to_data = {int(data[0]): data for data in questions_data}
    results = []
    for message in messages.data:
        if message.role != 'assistant':
            continue
        message_text = message.content[0].text.value
        if not message_text.startswith('Answer'):
            print(f"Unexpected message format: {message_text}")
            continue

        # Extract the question number from the message
        q_id = int(message_text[7:message_text.index(':')])
        _, question_text, true_answer = q_id_to_data[q_id]

        result = {
            'model': model_name,
            'question_id': q_id,
            'question_text': question_text,
            'true_answer': true_answer,
            'code_execution': True,
        }

        try:
            response = message_text.split('\n', 1)[1].strip()
            result['full_answer'] = response
            result['tokens_used'] = total_tokens / len(questions_data)

            latex_answer = extract_latex_answer(response)
            result['final_answer_latex'] = latex_answer

        except Exception as e:
            print(f"Error sending message to model: {e}")
            result['error'] = traceback.format_exc()

        results.append(result)
        df = pd.DataFrame.from_records([result])
        output_file = OUTPUT_FOLDER / f"{q_id}_{model_name}_results.xlsx"
        df.to_excel(output_file, index=False)

    print(f"Finished processing batch with model {model_name}.")
    return results


def main(): 
    questions = load_questions(parse_sympy=False)
    tasks = []
    for model_name in MODELS:
        for i in range(0, len(questions), BATCH_SIZE):
            batch = questions[i:i + BATCH_SIZE]
            tasks.append((batch, model_name))

    with mp.Pool(N_WORKERS) as pool:
        results = pool.starmap(handle_batch, tasks)
    
    joined_results = []
    for batch_result in results:
        joined_results += batch_result

    joined_df = pd.DataFrame.from_records(joined_results)
    joined_df.to_excel(OUTPUT_FOLDER / "all_results.xlsx", index=False)


if __name__ == "__main__":
    main()