"""Pluggable LLM backends for the explanation step.

Pick one with environment variables (or Kaggle Secrets):

    LLM_PROVIDER   anthropic (default) | openrouter | ollama | openai
    LLM_MODEL      model name for that provider (sensible default per provider)
    LLM_BASE_URL   override the endpoint (e.g. LM Studio http://localhost:1234/v1, vLLM, llama.cpp server)
    LLM_API_KEY    key for the provider; falls back to ANTHROPIC_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY

Everything except Anthropic goes through the OpenAI-compatible chat API, which OpenRouter, Ollama,
LM Studio, vLLM and llama.cpp's server all expose.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    default_model: str
    base_url: str | None
    key_env: str | None      # provider-specific key variable
    needs_key: bool = True


PROVIDERS = {
    "anthropic": ProviderSpec("claude-haiku-5-5", None, "ANTHROPIC_API_KEY"),
    "openrouter": ProviderSpec("meta-llama/llama-3.3-70b-instruct", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ProviderSpec("llama3.1:8b", "http://localhost:11434/v1", None, needs_key=False),
    "openai": ProviderSpec("gpt-4o-mini", None, "OPENAI_API_KEY"),  # any OpenAI-compatible server via LLM_BASE_URL
}


def settings():
    name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {name!r}")
    spec = PROVIDERS[name]
    key = os.environ.get("LLM_API_KEY") or (os.environ.get(spec.key_env) if spec.key_env else None)
    model = os.environ.get("LLM_MODEL") or spec.default_model
    base_url = os.environ.get("LLM_BASE_URL") or spec.base_url
    return name, spec, model, base_url, key


def available() -> bool:
    name, spec, _, base_url, key = settings()
    if name == "openai" and base_url and not key:
        return True  # local OpenAI-compatible server (LM Studio, vLLM, llama.cpp) usually needs no key
    return bool(key) or not spec.needs_key


def describe() -> str:
    name, _, model, base_url, _ = settings()
    return f"{name}:{model}" + (f" @ {base_url}" if base_url and name != "openrouter" else "")


class ChatClient:
    """`chat(system, user) -> str` over whichever provider is configured."""

    def __init__(self):
        self.provider, spec, self.model, base_url, key = settings()
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=key)
        else:
            from openai import OpenAI

            self._client = OpenAI(base_url=base_url, api_key=key or "not-needed")

    def chat(self, system: str, user: str, max_tokens: int = 600) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                                messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in resp.content if b.type == "text")
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return resp.choices[0].message.content or ""
