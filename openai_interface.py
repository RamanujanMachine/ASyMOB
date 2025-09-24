from llm_interface import GenericLLMInterface
import openai
from openai import OpenAI
import uuid

TIMEOUT = 60
REUSE_CONTAINER = True

class OpenAIInterface(GenericLLMInterface):
    """
    A class to interact with OpenAI's official API using SDK >= 1.0.0,
    with optional code interpreter (tools) support.
    """
    def __init__(self, model):
        self.model = model
        self._load_api_key(provider='openai')
        self.client = OpenAI(api_key=self.api_key)
        self.container = None

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
            if REUSE_CONTAINER:
                if self.container is None:
                    self.container = self.client.containers.create(
                        name=f"code-interpreter-{str(uuid.uuid4())}",
                    )
                    print(f"Created container {self.container.id}")
                else:
                    print(f"Reusing container {self.container.id}")
                container = self.container
            else:
                container = self.client.containers.create(
                    name=f"code-interpreter-{str(uuid.uuid4())}",
                )

            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=message,
                    tools=[{
                        'type': 'code_interpreter', 
                        # 'container': {'type': 'auto'}}]
                        'container': container.id}],
                    timeout=TIMEOUT
                )
            except openai.OpenAIError as e:
                # for every error, delete the container and raise
                print(f'error during code execution: {e}')
                print(f'cleaning up container {container.id}')
                if REUSE_CONTAINER:
                    self.container = None
                raise e

        code_interpreter_ids = set()
        for p in response.output:
            if p.type == 'code_interpreter_call':
                code_interpreter_ids.add(p.container_id)
        if len(code_interpreter_ids) > 1:
            import json
            error_data = {
                'message': message,
                'model': self.model,
                'code_execution': code_execution,
                'response': response.to_dict(),
            }
            with open('openai_multiple_code_interpreter.json', 'w') as f:
                json.dump(error_data, f, indent=2)

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