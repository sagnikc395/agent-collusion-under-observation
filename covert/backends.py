"""Batched chat-completion backends.

Every backend takes a list of conversations and returns one reply per
conversation, so the experiment can run each role as a single batch.
"""

import json
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class Backend:
    name = "base"

    def generate(
        self, convs: list[list[dict]], max_tokens: int, temperature: float, seed: int
    ) -> list[str]:
        raise NotImplementedError


class VLLMBackend(Backend):
    """Offline vLLM engine; the fastest option on a Colab GPU."""

    name = "vllm"

    def __init__(self, model: str, max_model_len: int = 2048, gpu_memory_utilization: float = 0.85):
        from vllm import LLM

        # dtype="half": T4s have no bf16, and fp16 is fine for 1.5B-7B models.
        self.llm = LLM(
            model=model,
            dtype="half",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )

    def generate(self, convs, max_tokens, temperature, seed):
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
        outs = self.llm.chat(convs, params, use_tqdm=False)  # the run log reports progress
        return [o.outputs[0].text.strip() for o in outs]


class HFBackend(Backend):
    """Plain transformers fallback, for when vLLM will not install."""

    name = "hf"

    def __init__(self, model: str, batch_size: int = 32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, device_map="auto"
        )
        self.batch_size = batch_size

    def generate(self, convs, max_tokens, temperature, seed):
        from tqdm.auto import tqdm

        self.torch.manual_seed(seed)
        texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
        out = []
        for i in tqdm(range(0, len(texts), self.batch_size), leave=False):
            enc = self.tok(texts[i : i + self.batch_size], return_tensors="pt", padding=True).to(
                self.model.device
            )
            sample = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_tokens, pad_token_id=self.tok.eos_token_id, **sample
                )
            new = gen[:, enc["input_ids"].shape[1] :]
            out += [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]
        return out


class OpenAICompatBackend(Backend):
    """Any local OpenAI-compatible server (Ollama, `vllm serve`, LM Studio)."""

    name = "openai"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434/v1", workers: int = 8):
        self.model, self.url, self.workers = model, base_url.rstrip("/") + "/chat/completions", workers

    def _one(self, conv, max_tokens, temperature, seed):
        body = {
            "model": self.model,
            "messages": conv,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.load(r)["choices"][0]["message"]["content"].strip()
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def generate(self, convs, max_tokens, temperature, seed):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda c: self._one(c, max_tokens, temperature, seed), convs))


class ScriptedBackend(Backend):
    """Answers with a Python function; used by the tests, no model needed."""

    name = "scripted"

    def __init__(self, fn: Callable[[list[dict]], str]):
        self.fn = fn

    def generate(self, convs, max_tokens, temperature, seed):
        return [self.fn(c) for c in convs]


def make_backend(kind: str, model: str, **kw) -> Backend:
    if kind == "auto":
        try:
            import vllm  # noqa: F401

            kind = "vllm"
        except ImportError:
            kind = "hf"
    cls = {"vllm": VLLMBackend, "hf": HFBackend, "openai": OpenAICompatBackend}[kind]
    return cls(model, **kw)
