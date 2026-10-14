"""Client for an OpenAI-compatible chat endpoint served on AMD GPUs.

Works with AMD Token Factory (the hackathon API) or your own vLLM server on an AMD
Instinct MI300X in AMD Developer Cloud. Configure with environment variables:

    LLM_BASE_URL   e.g. https://<token-factory-host>/v1  or  http://<mi300x-ip>:8000/v1
    LLM_API_KEY    API key for that endpoint
    LLM_MODEL      model name exposed by the endpoint
    LLM_HARDWARE   label shown in the UI, e.g. "AMD Instinct MI300X · ROCm · vLLM"
"""
from __future__ import annotations

import os
import random
import re
import time

import httpx


class LLM:
    def __init__(self):
        self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")
        self.hardware = os.environ.get("LLM_HARDWARE", "AMD Instinct GPU · ROCm")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "90"))
        self.no_think = os.environ.get("LLM_DISABLE_THINKING", "1") == "1"
        # Token Factory: "none" turns thinking off on DeepSeek/Qwen3.8-Flash/MiMo; "low" for models without "none"
        self.reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", "")
        self.stats = {"calls": 0, "errors": 0, "completion_tokens": 0, "seconds": 0.0}

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def info(self) -> dict:
        return {"enabled": self.enabled, "model": self.model or None,
                "hardware": self.hardware if self.enabled else None,
                "endpoint": re.sub(r"//([^/@]+@)", "//", self.base_url) or None, **self.stats}

    def chat(self, messages: list[dict], max_tokens: int = 1000, temperature: float = 0.2,
             json_mode: bool = False) -> tuple[str, dict]:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        elif self.no_think:   # self-hosted vLLM + Qwen3: skip hidden reasoning, answer directly
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.perf_counter()
        self.stats["calls"] += 1
        try:
            r = self._post(body, headers)
            if r.status_code in (400, 422):   # endpoint rejects an optional field: retry with the plain request
                for k in ("chat_template_kwargs", "response_format", "reasoning_effort"):
                    body.pop(k, None)
                r = self._post(body, headers)
            r.raise_for_status()
        except Exception:
            self.stats["errors"] += 1
            raise
        data = r.json()
        dt = time.perf_counter() - t0
        text = data["choices"][0]["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        usage = data.get("usage") or {}
        ctoks = usage.get("completion_tokens", 0)
        self.stats["completion_tokens"] += ctoks
        self.stats["seconds"] += dt
        if not text and data["choices"][0].get("finish_reason") == "length":
            raise RuntimeError("model used its whole token budget thinking; set LLM_REASONING_EFFORT=none or low")
        meta = {"model": data.get("model", self.model), "hardware": self.hardware, "latency_s": round(dt, 2),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": ctoks,
                "tokens_per_s": round(ctoks / dt, 1) if ctoks and dt else None}
        return text, meta

    def _post(self, body: dict, headers: dict) -> httpx.Response:
        """POST with retry on rate limits (429) and gateway hiccups (502/503), honouring Retry-After."""
        for attempt in range(4):
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
            if r.status_code not in (429, 502, 503) or attempt == 3:
                return r
            wait = float(r.headers.get("retry-after") or 2 ** attempt)
            time.sleep(min(wait, 60) + random.uniform(0, 1))
        return r
