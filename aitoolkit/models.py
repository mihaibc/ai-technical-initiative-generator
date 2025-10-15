from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from huggingface_hub import InferenceClient


@dataclass
class LLMConfig:
    provider: str = os.environ.get("LLM_PROVIDER", "hf_inference")  # hf_inference|local|openai|together|groq
    model: str = os.environ.get("HF_INFERENCE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    max_new_tokens: int = 600
    temperature: float = 0.4
    top_p: float = 0.9
    stop: Optional[List[str]] = None


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._init_provider()

    def _init_provider(self):
        p = self.config.provider
        if p == "hf_inference":
            token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
            self.hf = InferenceClient(token=token)
        elif p == "openai":
            # Lazy import to avoid dependency unless needed
            from openai import OpenAI  # type: ignore

            self.openai = OpenAI()
        elif p == "together":
            # Uses Together's API
            self.together_api_key = os.environ.get("TOGETHER_API_KEY")
        elif p == "groq":
            self.groq_api_key = os.environ.get("GROQ_API_KEY")
        elif p == "local":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            model_id = self.config.model
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
        else:
            raise ValueError(f"Unsupported provider: {p}")

    def generate(self, prompt: str, **kwargs) -> str:
        p = self.config.provider
        if p == "hf_inference":
            model = self.config.model
            resp = self.hf.text_generation(
                prompt,
                model=model,
                max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                stop_sequences=kwargs.get("stop", self.config.stop),
            )
            return resp
        elif p == "local":
            out = self.pipe(
                prompt,
                max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=True,
            )
            return out[0]["generated_text"]
        elif p == "openai":
            # Assumes model is a chat model id
            client = self.openai
            completion = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", self.config.model),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
            return completion.choices[0].message.content or ""
        elif p == "together":
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.together_api_key}", "Content-Type": "application/json"}
            data = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }
            r = requests.post(url, headers=headers, json=data, timeout=60)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"]
        elif p == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"}
            data = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }
            r = requests.post(url, headers=headers, json=data, timeout=60)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unsupported provider: {p}")
