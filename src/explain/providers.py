"""Pluggable LLM backends for the explanation step.

Pick one with environment variables (or Kaggle Secrets):

    LLM_PROVIDER   anthropic (default) | openrouter | ollama | openai | hf
    LLM_MODEL      model name for that provider (sensible default per provider)
    LLM_BASE_URL   override the endpoint (e.g. LM Studio http://localhost:1234/v1, vLLM, llama.cpp server)
    LLM_API_KEY    key for the provider; falls back to ANTHROPIC_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY

Everything except Anthropic and hf goes through the OpenAI-compatible chat API, which OpenRouter, Ollama,
LM Studio, vLLM and llama.cpp's server all expose.

hf runs a Hugging Face chat model on the local GPU with `transformers` (e.g. Qwen/Qwen2.5-7B-Instruct):
    HF_TOKEN       only for gated models (e.g. meta-llama/*): accept the licence on the model page first
    LLM_LOAD_4BIT  "1" to load in 4-bit (needs bitsandbytes) for models too big for the GPU in fp16
    LLM_BATCH      prompts generated together per batch (default 8)
hf decodes greedily and OpenAI-compatible providers use temperature 0, so their results are reproducible.
"""
import gc
import os
import time
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
    "hf": ProviderSpec("Qwen/Qwen2.5-7B-Instruct", None, "HF_TOKEN", needs_key=False),
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


_HF_CACHE = {}  # model id -> (tokenizer, model); one model on the GPU at a time


def unload_hf():
    """Free the GPU before loading the next Hugging Face model."""
    _HF_CACHE.clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


class HFLocal:
    """A Hugging Face chat model on the local GPU (greedy decoding, batched, left padding)."""

    def __init__(self, model_id: str, load_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_id not in _HF_CACHE:
            unload_hf()
            token = os.environ.get("HF_TOKEN") or None
            tok = AutoTokenizer.from_pretrained(model_id, token=token)
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            kwargs = {"device_map": "auto", "token": token}
            if load_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            else:  # T4 has no bfloat16; transformers 5 renamed torch_dtype -> dtype
                import transformers

                dtype_arg = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
                kwargs[dtype_arg] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
            _HF_CACHE[model_id] = (tok, model)
        self.tok, self.model = _HF_CACHE[model_id]

    def _prompt(self, system: str, user: str) -> str:
        try:
            return self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                                tokenize=False, add_generation_prompt=True)
        except Exception:  # templates without a system role: put the instructions in the user turn
            return self.tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}],
                                                tokenize=False, add_generation_prompt=True)

    def generate(self, pairs, max_tokens: int):
        import torch

        enc = self.tok([self._prompt(s, u) for s, u in pairs], return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


class ChatClient:
    """`chat(system, user) -> str` over whichever provider is configured."""

    def __init__(self):
        self.provider, spec, self.model, base_url, key = settings()
        if self.provider == "hf":
            self._hf = HFLocal(self.model, load_4bit=os.environ.get("LLM_LOAD_4BIT") == "1")
        elif self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=key)
        else:
            from openai import OpenAI

            self._client = OpenAI(base_url=base_url, api_key=key or "not-needed")

    def chat(self, system: str, user: str, max_tokens: int = 600) -> str:
        return self.chat_batch([(system, user)], max_tokens)[0]

    def chat_batch(self, pairs, max_tokens: int = 600, retries: int = 4):
        """Completions for several (system, user) pairs. hf: generated in GPU batches; APIs: one call each,
        retried with exponential backoff on rate limits / transient server errors."""
        if self.provider == "hf":
            size = int(os.environ.get("LLM_BATCH", "8"))
            return [t for i in range(0, len(pairs), size) for t in self._hf.generate(pairs[i:i + size], max_tokens)]
        out = []
        for system, user in pairs:
            for attempt in range(retries + 1):
                try:
                    out.append(self._api(system, user, max_tokens))
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
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return resp.choices[0].message.content or ""
