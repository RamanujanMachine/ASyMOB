from llm_interface import GenericLLMInterface
from gemini_interface import GeminiInterface
from openai_interface import OpenAIInterface
from groq_interface import GroqInterface


MODELS = {
    # OpenAI models
    "openai/o3-mini": OpenAIInterface("o3-mini"),
    "openai/gpt-4o": OpenAIInterface("gpt-4o"),
    "openai/gpt-4o-mini": OpenAIInterface("gpt-4o-mini"),

    # Google Gemini models
    "gemini/gemini-2.0-flash": \
        GeminiInterface("gemini-2.0-flash"),
    # "gemini/gemini-2.0-flash-lite": \
    #     GeminiInterface("gemini-2.0-flash-lite"),
    # "gemini/gemini-2.5-flash-preview-04-17": \
    #     GeminiInterface("gemini-2.5-flash-preview-04-17"),
    "gemini/gemini-2.5-pro-preview-03-25": \
        GeminiInterface("gemini-2.5-pro-preview-03-25"),
    "gemini/gemma-3-27b-it": \
        GeminiInterface("gemma-3-27b-it"),

    # CLAUDE - MUST HAVE

    # Groq models
    # MOVE to qwen-math
    'qwen-qwq-32b' : GroqInterface("qwen-qwq-32b"),
    'meta-llama/llama-4-maverick-17b-128e-instruct': \
        GroqInterface("meta-llama/llama-4-maverick-17b-128e-instruct"),
    'meta-llama/llama-4-scout-17b-16e-instruct': \
        GroqInterface("meta-llama/llama-4-scout-17b-16e-instruct"),

    # What is the "distill" thing?
    'deepseek-r1-distill-llama-70b': \
        GroqInterface("deepseek-r1-distill-llama-70b"),
}