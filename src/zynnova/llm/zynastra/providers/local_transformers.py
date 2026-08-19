"""Transformers runtime for models stored in the external ZynNova workspace."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..config import ProviderConfig
from ..types import Message, ProviderResponse, ToolSpec


class LocalTransformersProvider:
    name = "local-transformers"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install zynnova[llm-local] for local models") from exc
        path = Path(config.model).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
        ).eval()

    async def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> ProviderResponse:
        rendered = [m.as_openai() for m in messages]
        if tools:
            rendered = [
                {"role": "system", "content": "Available tools (return tool calls only if your chat template supports them): " + json.dumps([t.as_openai() for t in tools])},
                *rendered,
            ]
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(rendered, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n".join(f"{m['role']}: {m.get('content','')}" for m in rendered) + "\nassistant:"
        inputs = tok(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        max_new = self.config.max_output_tokens or 1024
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=self.config.temperature is not None and self.config.temperature > 0,
                temperature=self.config.temperature or 1.0,
                pad_token_id=tok.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        return ProviderResponse(text=tok.decode(generated, skip_special_tokens=True), finish_reason="stop")

    async def aclose(self) -> None:
        return None


__all__ = ["LocalTransformersProvider"]
