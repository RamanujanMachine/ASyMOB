from litellm import completion
import json

NO_CODE_PREFIX = (
    "Assume you don't have access to a computer: do not use "
    "code, solve this manually - using your internal reasoning.\n"
)
USE_CODE_PREFIX = (
    "Please use Python to solve the following question. Don't show it, "
    "just run it internally.\n"
)

class GenericLLMInterface:
    """
    A class to interact with different LLMs.
    
    Attributes:
        model (str): The model to use.
        api_key (str): The API key for the model.
    """

    def __init__(self, model):
        self.model = model
        self._load_api_key(model.split('/')[0])

    def _load_api_key(self, provider):
        with open('api_keys.json', 'r') as f:
            keys = json.load(f)
        
        if provider not in keys:
            raise ValueError("API key not found in api_key.txt")
        
        self.api_key = keys[provider]

    def _incentivize_code_execution(self, message, use_code=True):
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
    
    def send_message(self, message, code_execution=None):
        """
        Send a message to the LLM and receive a response.
        
        Args:
            message (str): The message to send.
        
        Returns:
            str: The response from the LLM.
        """
        message = self._incentivize_code_execution(message, use_code=code_execution)

        messages = [
            {"role": "user", "content": message}
        ]
        response = completion(
            model=self.model, 
            api_key=self.api_key, 
            messages=messages)
        return response.choices[0].message.content