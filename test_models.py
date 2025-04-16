from llm_interface import send_and_receive_message
from models import MODELS


for model in MODELS:
    response = send_and_receive_message(model, "Hello, how are you?")
    print(model, response)
