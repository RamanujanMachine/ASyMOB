from llm_interface import GenericLLMInterface
from gemini_interface import GeminiInterface
from openai_interface import OpenAIInterface
from groq_interface import GroqInterface
from claude_interface import ClaudeInterface
from hugging_face_interface import HuggingFaceInterface

MODELS_GENERATORS = {
    # OpenAI models
    "openai/o3": lambda: OpenAIInterface("o3"),
    "openai/gpt-4o": lambda: OpenAIInterface("gpt-4o"), 
    "openai/gpt-4.1": lambda: OpenAIInterface("gpt-4.1"),

    # Google Gemini models
    "gemini/gemini-2.0-flash": lambda: GeminiInterface("gemini-2.0-flash"),
    "gemini/gemini-2.5-pro-preview-03-25": lambda: GeminiInterface("gemini-2.5-pro-preview-03-25"),
    "gemini/gemma-3-27b-it": lambda: GeminiInterface("gemma-3-27b-it"),

    # Claude
    # 'claude': lambda: ClaudeInterface(),

    # HuggingFace models
    'Qwen/Qwen3-235B-A22B': lambda: HuggingFaceInterface('Qwen/Qwen3-235B-A22B'),
    'DeepSeek-Prover-V2-671B': lambda: HuggingFaceInterface("deepseek-ai/DeepSeek-Prover-V2-671B"),
    'DeepSeek-R1': lambda: HuggingFaceInterface('deepseek-ai/DeepSeek-R1'),
    'DeepSeek-V3': lambda: HuggingFaceInterface('deepseek-ai/DeepSeek-V3'),
    'meta-llama/Llama-4-Scout-17B-16E-Instruct': lambda: HuggingFaceInterface("meta-llama/Llama-4-Scout-17B-16E-Instruct"),

    # Groq models
    'meta-llama/llama-4-maverick-17b-128e-instruct': lambda: GroqInterface("meta-llama/llama-4-maverick-17b-128e-instruct"),

    # Missing models:
    # OpenMath2-LLaMA3.1-70B-Nemo
    # Qwen2.5-Math-7B

}

def get_model(model_name):
    return MODELS_GENERATORS[model_name]()


MODELS = {k : get_model(k) for k in MODELS_GENERATORS.keys()}