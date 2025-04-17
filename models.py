from llm_interface import GenericLLMInterface
from gemini_interface import GeminiInterface
from openai_interface import OpenAIInterface

MODELS = {
    "openai/o3-mini": OpenAIInterface("o3-mini"),
    "openai/gpt-4o": OpenAIInterface("gpt-4o"),
    "openai/gpt-4o-mini": OpenAIInterface("gpt-4o-mini"),
    
    "gemini/gemini-2.0-flash": GeminiInterface("gemini-2.0-flash"),
    "gemini/gemini-2.0-flash-thinking-exp": GeminiInterface("gemini-2.0-flash-thinking-exp"),
    "gemini/gemini-2.5-pro-exp-03-25": GeminiInterface("gemini-2.5-pro-exp-03-25"),
}