from abc import abstractmethod
import json


class LLMInterface:
    def __init__(self, api_key: str = None):
        pass

    @abstractmethod
    def write(self):
        """
        Write a message to the language model.
        """
        pass
    
    @abstractmethod
    def read(self):
        """
        Read a message from the language model.
        """
        pass

    @abstractmethod
    def chat(self):
        """
        Write a single message and read the response from the language model.
        """
        pass

    def _load_api_key(self, api_key: str = None, provider: str = None) -> str:
        if api_key is None:
            with open('api_keys.json', 'r') as f:
                keys = json.load(f)
            if provider not in keys:
                raise ValueError("API key not found in api_key.txt and "
                                 "was not provided.")
            api_key = keys['gemini']
        self.api_key = api_key