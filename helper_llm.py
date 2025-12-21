"""
LLMクライアント抽象化レイヤー

OpenAI API と Gemini 3 API の両方に対応する統一インターフェースを提供。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type, List, Dict
import os
import json
import logging

from pydantic import BaseModel
from dotenv import load_dotenv

# SDK imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
    from google.api_core import exceptions
except ImportError:
    genai = None

import tiktoken

load_dotenv()

logger = logging.getLogger(__name__)

# --- LLM モデル設定 --- #
LLM_MODELS = [
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "gemini-2.5-flash-preview",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

LLM_PRICING = {
    "gemini-2.5-flash": {"input": 0.0001, "output": 0.0004}, # Estimated
    "gemini-3-pro-preview": {"input": 0.00125, "output": 0.010},
    "gemini-2.5-flash-preview": {"input": 0.00015, "output": 0.0035},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
}

LLM_LIMITS = {
    "gemini-2.5-flash": {"max_tokens": 1000000, "max_output": 8192},
    "gemini-3-pro-preview": {"max_tokens": 1000000, "max_output": 64000},
    "gemini-2.5-flash-preview": {"max_tokens": 1000000, "max_output": 64000},
    "gemini-2.0-flash": {"max_tokens": 1000000, "max_output": 8192},
    "gemini-1.5-pro": {"max_tokens": 1000000, "max_output": 8192},
    "gemini-1.5-flash": {"max_tokens": 1000000, "max_output": 8192},
}

# --- Embedding モデル設定 --- #
EMBEDDING_MODELS = [
    "gemini-embedding-001",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

EMBEDDING_PRICING = {
    "gemini-embedding-001": 0.0001,
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
}

EMBEDDING_DIMS = {
    "gemini-embedding-001": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


class LLMClient(ABC):
    @abstractmethod
    def generate_content(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_structured(self, prompt: str, response_schema: Type[BaseModel], model: Optional[str] = None, **kwargs) -> BaseModel:
        pass

    @abstractmethod
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-4o-mini"):
        if not OpenAI:
            raise ImportError("openai package is not installed.")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = default_model

    def generate_content(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        model = model or self.default_model
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content

    def generate_structured(self, prompt: str, response_schema: Type[BaseModel], model: Optional[str] = None, **kwargs) -> BaseModel:
        model = model or self.default_model
        messages = [{"role": "user", "content": prompt}]
        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_schema,
            **kwargs
        )
        return response.choices[0].message.parsed

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        model = model or self.default_model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, default_model: str = "gemini-2.0-flash"):
        if not genai:
            raise ImportError("google-generativeai package is not installed.")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=self.api_key)
        self.default_model = default_model

    def generate_content(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        model_name = model or self.default_model
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, **kwargs)
        return response.text

    def generate_structured(self, prompt: str, response_schema: Type[BaseModel], model: Optional[str] = None, **kwargs) -> BaseModel:
        model_name = model or self.default_model
        model = genai.GenerativeModel(model_name)
        
        # Gemini JSON mode
        generation_config = {"response_mime_type": "application/json"}
        # スキーマをプロンプトに追加する簡易実装（SDKの進化に合わせて変更可能）
        schema_prompt = f"{prompt}\n\nOutput in JSON format following this schema: {response_schema.model_json_schema()}"
        
        response = model.generate_content(schema_prompt, generation_config=generation_config, **kwargs)
        try:
            return response_schema.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"JSON parse error: {e}")
            raise

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        model_name = model or self.default_model
        model = genai.GenerativeModel(model_name)
        return model.count_tokens(text).total_tokens

def create_llm_client(provider: str = "gemini", **kwargs) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(**kwargs)
    return GeminiClient(**kwargs)

# Helper functions
def get_available_llm_models() -> List[str]:
    return LLM_MODELS

def get_llm_model_pricing(model_name: str) -> Dict[str, float]:
    return LLM_PRICING.get(model_name, {"input": 0.0, "output": 0.0})

def get_llm_model_limits(model_name: str) -> Dict[str, int]:
    return LLM_LIMITS.get(model_name, {"max_tokens": 0, "max_output": 0})

def get_available_embedding_models() -> List[str]:
    return EMBEDDING_MODELS

def get_embedding_model_pricing(model_name: str) -> float:
    return EMBEDDING_PRICING.get(model_name, 0.0)

def get_embedding_model_dimensions(model_name: str) -> int:
    return EMBEDDING_DIMS.get(model_name, 0)