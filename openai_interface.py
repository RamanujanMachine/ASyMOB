from llm_interface import GenericLLMInterface
import openai
from openai import OpenAI

class OpenAIInterface(GenericLLMInterface):
    """
    A class to interact with OpenAI's official API using SDK >= 1.0.0,
    with optional code interpreter (tools) support.
    """
    def __init__(self, model):
        self.model = model
        self._load_api_key(provider='openai')
        self.client = OpenAI(api_key=self.api_key)

    def send_message(self, message, code_execution=False, flex=False, return_tokens=False):
        """
        Send a message to OpenAI's API, optionally enabling the code interpreter tool.
        
        Args:
            message (str): The user message.
            use_code (bool): Whether to enable the code interpreter tool.

        Returns:
            str: The assistant's reply.
        """
        if not flex:
            response = self.client.responses.create(
                model=self.model,
                input=message,
            )
        else:
            response = self.client.with_options(timeout=900.0).responses.create(
                model=self.model,
                input=message,
                service_tier="flex",
            )

        if return_tokens:
            return (
                response.output_text, 
                response.usage.total_tokens
            )

        return response.output_text
    
    def support_code(self):
        return True