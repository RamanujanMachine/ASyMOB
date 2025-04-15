import google.generativeai as genai
from llm_interface import LLMInterface


class GeminiInterface(LLMInterface):
    def __init__(self, api_key: str = None, 
                 model: str = 'models/gemini-2.0-flash-exp'):
        super().__init__()
        self._load_api_key(api_key, 'gemini')
        genai.configure(api_key=self.api_key)
        
        # Load the Gemini Pro model
        self.model = genai.GenerativeModel(model)

        # Start a chat session
        self.chat = self.model.start_chat()

    def write(self, prompt: str) -> str:
        """
        Write a message to the language model.
        """
        pass

    def read(self) -> str:
        """
        Read a message from the language model.
        """
        # This method is not applicable for Gemini API as it doesn't maintain a conversation state.
        pass

    def send_and_receive(self, prompt: str) -> str:
        """
        Write a single message and read the response from the language model.
        """
        # Send the prompt to the model and get the response
        response = self.chat.send_message(prompt)
        
        # Return the response text
        return response.text