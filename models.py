from llm_interface import GenericLLMInterface
from gemini_interface import GeminiInterface
from openai_interface import OpenAIInterface
from groq_interface import GroqInterface
from claude_interface import ClaudeInterface
from hugging_face_interface import HuggingFaceInterface

MODELS = {
    # OpenAI models
    "openai/o3": OpenAIInterface("o3"),
    "openai/gpt-4o": OpenAIInterface("gpt-4o"), 
    "openai/gpt-4.1": OpenAIInterface("gpt-4.1"),

    # Google Gemini models
    "gemini/gemini-2.0-flash": \
        GeminiInterface("gemini-2.0-flash"),
    "gemini/gemini-2.5-pro-preview-03-25": \
        GeminiInterface("gemini-2.5-pro-preview-03-25"),
    "gemini/gemma-3-27b-it": \
        GeminiInterface("gemma-3-27b-it"),

    # Claude
    'claude': ClaudeInterface(),

    # HuggingFace models
    'Qwen/Qwen3-235B-A22B': HuggingFaceInterface('Qwen/Qwen3-235B-A22B'),

    # Groq models
    'meta-llama/llama-4-maverick-17b-128e-instruct': \
        GroqInterface("meta-llama/llama-4-maverick-17b-128e-instruct"),
    'meta-llama/llama-4-scout-17b-16e-instruct': \
        GroqInterface("meta-llama/llama-4-scout-17b-16e-instruct"),

    # What is the "distill" thing?
    'deepseek-r1-distill-llama-70b': \
        GroqInterface("deepseek-r1-distill-llama-70b"),
}