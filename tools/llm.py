from __future__ import annotations

from typing import Optional, Tuple, Any

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
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.device = device
        torch_dtype = getattr(torch, dtype) if dtype and hasattr(torch, dtype) else None

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        if model is not None:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self.model.to(device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
        # Apply chat template if available (required for Qwen3 instruct models)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False  # Disable Qwen3 CoT thinking tokens
            )
        else:
            formatted = prompt
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[-1])

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
            # FIX: prevent repetition loops (Kevin Spacey bug)
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 4,
        }
        # Disable Qwen3 thinking mode — keeps output concise and avoids token budget issues
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            gen_kwargs["temperature"] = 0.0 if temperature == 0 else temperature
            gen_kwargs["do_sample"] = False
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0]
        completion_ids = generated_ids[prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        # Strip Qwen3 thinking blocks from decoded output
        import re
        text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)  # strip any remaining special tokens
        text = text.strip()

        # Stop string handling — support single string or list
        stop_list = [stop] if isinstance(stop, str) else (stop or [])
        # Qwen3 CoT: always stop at second-attempt markers
        stop_list += ["Okay, let me try", "Wait—actually", "Let me reconsider",
                      "Hmm,", "Actually,", "\n\nOkay"]
        for s in stop_list:
            if s and s in text:
                text = text.split(s)[0]

        # FIX: strip at first newline for answer-style prompts that end with "Answer:"
        # This prevents "Answer: Paris\nReasoning: ..." from being returned in full.
        # Only strip if the decoded text looks like a multi-line answer+reasoning block.
        if "\nReasoning:" in text:
            text = text.split("\nReasoning:")[0]
        elif "\nAnswer:" in text:
            # model repeated the Answer: prefix — take only first answer
            text = text.split("\nAnswer:")[0]

        usage_record = self.usage_tracker.track(
            name=name,
            prompt_tokens=prompt_len,
            completion_tokens=int(max(0, len(completion_ids))),
        )
        return text.strip(), usage_record