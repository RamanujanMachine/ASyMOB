from groq import Groq
from llm_interface import GenericLLMInterface, USE_CODE_PREFIX, NO_CODE_PREFIX

class GroqInterface(GenericLLMInterface):
    """
    A class to interact specifically with models over Groq.

    Inherits from GenericLLMInterface but overrides send_message to use
    the groq API.
    """
    def __init__(self, model):
        self.model = model
        self._load_api_key(provider='groq')
        self.client = Groq(api_key=self.api_key)

    def send_message(self, message, code_execution=False, 
                     max_completion_tokens=100_000):
        message = self._incentivize_code_execution(
            message, use_code=code_execution)
        print(f"Sending message to Groq model: {self.model}")
        print(f"Message: {message}")

        # Some models have a max token limit.
        if self.model == 'meta-llama/llama-4-scout-17b-16e-instruct':
            if max_completion_tokens > 8192:
                max_completion_tokens = 8192
        elif self.model == 'meta-llama/llama-4-maverick-17b-128e-instruct': 
            if max_completion_tokens > 8192:
                max_completion_tokens = 8192

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                    "role": "user",
                    "content": message}],
            temperature=0.6,
            max_completion_tokens=max_completion_tokens,
            top_p=0.95,
            stream=True,
            stop=None,
        )

        response = ''
        for chunk in completion:
            if chunk.choices[0].delta.content is None:
                continue
            response += chunk.choices[0].delta.content

        return response