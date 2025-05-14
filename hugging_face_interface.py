from llm_interface import GenericLLMInterface
from huggingface_hub import InferenceClient

class HuggingFaceInterface(GenericLLMInterface):
    """
    A class to interact with Hugging Face's Inference API with optional code interpreter (tools) support.
    """
    def __init__(self, model, inference_provider="novita"):
        self.model = model
        self._load_api_key(provider='huggingface')
        self.client = InferenceClient(
            api_key=self.api_key, 
            provider=inference_provider,
            timeout=600 # up to ten minutes
            )
        self.chat_history = []

    def send_message(self, message, code_execution=False, return_tokens=False):
        """
        Send a message to Hugging Face's API, optionally enabling the code interpreter tool.

        Args:
            message (str): The user message.
            code_execution (bool): Whether to incentivize code execution.

        Returns:
            str: The assistant's reply.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": message}],
            temperature=0.5,
            max_tokens=20_000,
            top_p=0.7,
            stream=False
        )

        full_reply = response.choices[0].message.content

        if return_tokens:
            return (
                full_reply,
                response.usage.total_tokens
            )
        return full_reply

    def support_code(self):
        return False
