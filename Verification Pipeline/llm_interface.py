from abc import ABC, abstractmethod
from litellm import completion
import json

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

    def send_message(self, message, code_execution=None, return_tokens=False):
        """
        Send a message to the LLM and receive a response.
        
        Args:
            message (str): The message to send.
        
        Returns:
            str: The response from the LLM.
        """
        messages = [
            {"role": "user", "content": message}
        ]
        response = completion(
            model=self.model, 
            api_key=self.api_key, 
            messages=messages)
        
        return response.choices[0].message.content

    @abstractmethod
    def support_code(self):
        pass