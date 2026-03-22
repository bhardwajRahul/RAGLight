from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
from typing_extensions import override
from ..config.settings import Settings
from .llm import LLM
import logging

from openai import OpenAI


class LMStudioModel(LLM):
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        api_base: Optional[str] = None,
        role: str = "user",
    ) -> None:
        self.api_base = api_base or Settings.DEFAULT_LMSTUDIO_CLIENT
        super().__init__(model_name, system_prompt, system_prompt_file, self.api_base)
        logging.info(f"Using LMStudio with {model_name} model 🤖")
        self.role: str = role

    @override
    def load(self) -> OpenAI:
        return OpenAI(base_url=self.api_base, api_key="lm-studio")

    @override
    def generate(self, input: Dict[str, Any]) -> str:
        history = input.get("history", [])
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {"role": self.role, "content": input.get("question", "")}
        if "images" in input:
            payload["images"] = input["images"]
        messages.append(payload)

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    @override
    def generate_streaming(self, input: Dict[str, Any]) -> Iterable[str]:
        history = input.get("history", [])
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {"role": self.role, "content": input.get("question", "")}
        if "images" in input:
            payload["images"] = input["images"]
        messages.append(payload)

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
