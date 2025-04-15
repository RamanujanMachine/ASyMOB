from litellm import completion
import json


def _get_api_key(provider):
    with open('api_keys.json', 'r') as f:
        keys = json.load(f)
    
    if provider not in keys:
        raise ValueError("API key not found in api_key.txt")
    
    return keys[provider]


def send_and_receive_message(model, message):
    provider = model.split('/')[0]
    api_key = _get_api_key(provider)
    messages = [
        {"role": "user", "content": message}
    ]
    response = completion(
        model=model, 
        api_key=api_key, 
        messages=messages)
    return response.choices[0].message.content
