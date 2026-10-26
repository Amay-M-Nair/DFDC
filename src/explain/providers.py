"""Pluggable LLM backends for the explanation step.

Pick one with environment variables (or Kaggle Secrets):

    LLM_PROVIDER   anthropic (default) | groq | openrouter | ollama | openai
    LLM_MODEL      model name for that provider (sensible default per provider)
    LLM_BASE_URL   override the endpoint (e.g. LM Studio http://localhost:1234/v1, vLLM, llama.cpp server)
    LLM_API_KEY    key used when the provider's own variable (GROQ_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY,
                   OPENAI_API_KEY) is not set
    LLM_MIN_INTERVAL      seconds between API requests, to stay inside free-tier rate limits (default 0)
    LLM_REASONING_EFFORT  reasoning effort for models that think before answering ("off" to not send it)
    LLM_MAX_TOKENS        output token limit (reasoning tokens included where the provider counts them)

Everything except Anthropic goes through the OpenAI-compatible chat API, which Groq, OpenRouter, Ollama,
LM Studio, vLLM and llama.cpp's server all expose. Groq has a free tier. The OpenAI-compatible providers use
temperature 0, so their results are (nearly) reproducible.
"""
import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    default_model: str
    base_url: str | None
    key_env: str | None      # provider-specific key variable
    needs_key: bool = True
    reasoning_effort: str | None = None
    max_tokens: int = 600


PROVIDERS = {
    "anthropic": ProviderSpec("claude-haiku-5-5", None, "ANTHROPIC_API_KEY"),
    "groq": ProviderSpec("openai/gpt-oss-120b", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                         reasoning_effort="low", max_tokens=1024),
    "openrouter": ProviderSpec("meta-llama/llama-3.3-70b-instruct", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ProviderSpec("llama3.1:8b", "http://localhost:11434/v1", None, needs_key=False),
    "openai": ProviderSpec("gpt-4o-mini", None, "OPENAI_API_KEY"),  # any OpenAI-compatible server via LLM_BASE_URL
}


def settings():
    name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {name!r}")
    spec = PROVIDERS[name]
    key = (os.environ.get(spec.key_env) if spec.key_env else None) or os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL") or spec.default_model
    base_url = os.environ.get("LLM_BASE_URL") or spec.base_url
    return name, spec, model, base_url, key


def available() -> bool:
    name, spec, _, base_url, key = settings()
    if name == "openai" and base_url and not key:
        return True  # local OpenAI-compatible server (LM Studio, vLLM, llama.cpp) usually needs no key
    return bool(key) or not spec.needs_key


def describe() -> str:
    name, spec, model, base_url, _ = settings()
    return f"{name}:{model}" + (f" @ {base_url}" if base_url and (base_url != spec.base_url or name == "openai") else "")


class ChatClient:
    """`chat(system, user) -> str` over whichever provider is configured."""

    def __init__(self):
        self.provider, self.spec, self.model, base_url, key = settings()
        effort = os.environ.get("LLM_REASONING_EFFORT", self.spec.reasoning_effort or "")
        self.reasoning_effort = None if effort in ("", "off") else effort
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", self.spec.max_tokens))
        self.min_interval = float(os.environ.get("LLM_MIN_INTERVAL", "0"))
        self._last_request = 0.0
        self.last_latency = 0.0  # seconds taken by the last successful API request (waits not included)
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=key)
        else:
            from openai import OpenAI

            self._client = OpenAI(base_url=base_url, api_key=key or "not-needed")

    def chat(self, system: str, user: str, max_tokens: int = None) -> str:
        return self.chat_batch([(system, user)], max_tokens)[0]

    def chat_batch(self, pairs, max_tokens: int = None, retries: int = 4):
        """Completions for several (system, user) pairs, one API call each, spaced by LLM_MIN_INTERVAL and
        retried with exponential backoff on rate limits / transient server errors."""
        max_tokens = max_tokens or self.max_tokens
        out = []
        for system, user in pairs:
            for attempt in range(retries + 1):
                time.sleep(max(0.0, self._last_request + self.min_interval - time.time()))
                self._last_request = start = time.time()
                try:
                    out.append(self._api(system, user, max_tokens))
                    self.last_latency = time.time() - start
                    break
                except Exception as e:  # error classes differ per SDK
                    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                    transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or "rate limit" in str(e).lower()
                    if attempt == retries or not transient:
                        raise
                    time.sleep(min(60, 3 * 2 ** attempt))
        return out

    def _api(self, system: str, user: str, max_tokens: int) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                                messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in resp.content if b.type == "text")
        kwargs = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
        return resp.choices[0].message.content or ""
