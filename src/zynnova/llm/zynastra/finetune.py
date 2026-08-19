"""LoRA/QLoRA supervised fine-tuning with outputs isolated in the external workspace."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .workspace import Workspace


@dataclass(frozen=True, slots=True)
class SFTConfig:
    dataset: str | Path
    text_field: str = "text"
    output_name: str = "adapter"
    max_seq_length: int = 2048
    epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] | None = None
    qlora_4bit: bool = False


def _load_dataset(spec: str | Path):
    from datasets import load_dataset
    path = Path(spec).expanduser() if isinstance(spec, Path) or Path(str(spec)).suffix else None
    if path is not None and path.exists():
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}: return load_dataset("json", data_files=str(path), split="train")
        if suffix in {".csv", ".tsv"}: return load_dataset("csv", data_files=str(path), split="train")
        raise ValueError(f"unsupported local dataset: {path}")
    return load_dataset(str(spec), split="train")


def finetune_lora(base_model: str | Path, config: SFTConfig, workspace: Workspace) -> Path:
    """Run SFT. The base model and adapter remain outside the installed ZynNova package."""
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError("install zynnova[llm-local] for LoRA/QLoRA fine-tuning") from exc
    workspace.ensure()
    run = workspace.finetunes / f"{config.output_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    quant = None
    if config.qlora_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model), trust_remote_code=True, device_map="auto",
        quantization_config=quant, torch_dtype=None if quant else "auto",
    )
    peft_cfg = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules) if config.target_modules else None,
        task_type="CAUSAL_LM",
    )
    args = TrainingArguments(
        output_dir=str(run / "checkpoints"), num_train_epochs=config.epochs,
        learning_rate=config.learning_rate, per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=10, save_strategy="epoch", report_to=[], bf16=torch.cuda.is_available(),
    )
    dataset = _load_dataset(config.dataset)
    kwargs: dict[str, Any] = {"model": model, "args": args, "train_dataset": dataset, "peft_config": peft_cfg}
    # TRL changed the text-field configuration surface across releases. Prefer
    # processing_class + dataset_text_field when accepted, then fall back cleanly.
    try:
        trainer = SFTTrainer(processing_class=tokenizer, dataset_text_field=config.text_field, max_seq_length=config.max_seq_length, **kwargs)
    except TypeError:
        try: trainer = SFTTrainer(tokenizer=tokenizer, dataset_text_field=config.text_field, max_seq_length=config.max_seq_length, **kwargs)
        except TypeError: trainer = SFTTrainer(processing_class=tokenizer, **kwargs)
    trainer.train()
    adapter = run / "adapter"
    trainer.model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    (run / "run.json").write_text(json.dumps({"base_model":str(base_model),"config":asdict(config)}, default=str, indent=2), encoding="utf-8")
    return adapter


__all__ = ["SFTConfig", "finetune_lora"]
