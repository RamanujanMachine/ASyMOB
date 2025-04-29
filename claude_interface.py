from llm_interface import GenericLLMInterface, USE_CODE_PREFIX, NO_CODE_PREFIX
import anthropic

class ClaudeInterface(GenericLLMInterface):
    """
    A class to interact with Anthropic's Claude API,
    with optional code execution support through prompting.
    """
    def __init__(self, model="claude-3-7-sonnet-20250219"):
        self.model = model
        self._load_api_key(provider='anthropic')
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
    def send_message(self, message, code_execution=False):
        """
        Send a message to Anthropic's Claude API, optionally encouraging code execution.
        
        Args:
            message (str): The user message.
            code_execution (bool): Whether to encourage Claude to write executable code.

        Returns:
            str: The assistant's reply.
        """
        message = self._incentivize_code_execution(message, use_code=code_execution)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        return response.content[0].text
    
    def support_code(self):
        return True