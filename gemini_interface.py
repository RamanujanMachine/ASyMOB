from google import genai
from google.genai import types
import os
from llm_interface import GenericLLMInterface, USE_CODE_PREFIX, NO_CODE_PREFIX

CODE_RUNNING_MODELS = [
    'gemini-2.0-flash',
    # 2.0 flash light does not support code execution
    'gemini-2.5-flash-preview-04-17',
    'gemini-2.5-pro-preview-03-25',
    # gemma-3-27b-it does not support code execution
]
class GeminiInterface(GenericLLMInterface):
    """
    A class to interact specifically with Google Gemini models using the native API.

    Inherits from GenericLLMInterface but overrides send_message to use
    the google-generativeai SDK.
    """
    def __init__(self, model):
        self.model = model
        self._load_api_key(provider='gemini')
        self.client = genai.Client(api_key=self.api_key)

    def send_message(self, message, code_execution=False, return_tokens=False):
        tools = []
        if code_execution and self.model in CODE_RUNNING_MODELS:
            tools.append(
                types.Tool(code_execution=types.ToolCodeExecution)
            )

        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            response_mime_type="text/plain",
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=message),
                ],
            ),
        ]

        response = ''

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.candidates is None or chunk.candidates[0].content is None or chunk.candidates[0].content.parts is None:
                continue
            if chunk.candidates[0].content.parts[0].text:
                # print(chunk.candidates[0].content.parts[0].text, end="")
                response += chunk.candidates[0].content.parts[0].text
            if chunk.candidates[0].content.parts[0].executable_code:
                # print(chunk.candidates[0].content.parts[0].executable_code)
                response += str(chunk.candidates[0].content.parts[0].executable_code)
                if not code_execution:
                    # Should't be here, but just in case
                    print("Code execution is not enabled.")
                    raise ValueError("Code execution is not enabled.")
            if chunk.candidates[0].content.parts[0].code_execution_result:
                response += str(chunk.candidates[0].content.parts[0].code_execution_result)
                if not code_execution:
                    # Should't be here, but just in case
                    print("Code execution is not enabled.")
                    raise ValueError("Code execution is not enabled.")
                # print(chunk.candidates[0].content.parts[0].code_execution_result)
        
        if return_tokens:
            # looks like the last chunk contains all of the token count
            return (
                response,
                chunk.usage_metadata.total_token_count 
            )
        return response

    def support_code(self):
        if 'gemma' in self.model:
            return False
        return True