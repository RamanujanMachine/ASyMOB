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

    def send_message(self, message, code_execution=False, return_extra=False):
        """
        Send a message to OpenAI's API, optionally enabling the code 
        interpreter tool.
        
        Args:
            message (str): The user message.
            use_code (bool): Whether to enable the code interpreter tool.

        Returns:
            str: The assistant's reply.
        """

        if not code_execution:
            response = self.client.responses.create(
                model=self.model,
                input=message
            )
        else:
            response = response = self.client.responses.create(
                model=self.model,
                input=message,
                tools=[{
                    'type': 'code_interpreter', 
                    'container': {'type': 'auto'}}]
            )

        if return_extra:
            code_used = any([
                out.type == 'code_interpreter_call' 
                for out in response.output])
            
            return (
                response.output_text, 
                response.usage.total_tokens,
                code_used
            )

        return response.output_text
    
    def support_code(self):
        return True