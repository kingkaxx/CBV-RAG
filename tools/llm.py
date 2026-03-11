from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics.usage import UsageTracker


class LLMEngine:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        dtype: Optional[str] = None,
        max_new_tokens: int = 128,
        usage_tracker: Optional[UsageTracker] = None,
    ) -> None:
        self.device = device
        torch_dtype = getattr(torch, dtype) if dtype and hasattr(torch, dtype) else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.usage_tracker = usage_tracker or UsageTracker()

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[str] = None,
        name: str = "llm.generate",
    ) -> Tuple[str, dict]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = outputs[0]
        completion_ids = generated_ids[prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        if stop and stop in text:
            text = text.split(stop)[0]
        usage_record = self.usage_tracker.track(
            name=name,
            prompt_tokens=prompt_len,
            completion_tokens=int(max(0, len(completion_ids))),
        )
        return text.strip(), usage_record
