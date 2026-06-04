"""Model loading and trainable-parameter setup for LLM fine-tuning."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from src.training.fine_tuning import FineTuningMethod, get_strategy


def load_model(args: argparse.Namespace):
    strategy = get_strategy(args.fine_tuning_method)
    model_kwargs: dict[str, Any] = {}

    dtype = resolve_torch_dtype(args.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    attention_implementation = resolve_attention_implementation(args)
    if attention_implementation is not None:
        model_kwargs["attn_implementation"] = attention_implementation

    if strategy.method == FineTuningMethod.QLORA:
        model_kwargs["quantization_config"] = qlora_quantization_config(dtype or torch.float16)

    return AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)


def configure_fine_tuning(model, args: argparse.Namespace):
    strategy = get_strategy(args.fine_tuning_method)
    if strategy.method == FineTuningMethod.FULL:
        print_trainable_parameters(model)
        return model
    if strategy.method == FineTuningMethod.PARTIAL:
        model = configure_partial_fine_tuning(model, strategy.default_hyperparameters)
        print_trainable_parameters(model)
        return model
    if strategy.method in {FineTuningMethod.LORA, FineTuningMethod.QLORA, FineTuningMethod.DORA}:
        return configure_lora_like_fine_tuning(model, args, strategy)
    if strategy.method == FineTuningMethod.PREFIX_TUNING:
        return configure_prefix_tuning(model, args, strategy)
    if strategy.method == FineTuningMethod.PROMPT_TUNING:
        return configure_prompt_tuning(model, args, strategy)
    if strategy.method == FineTuningMethod.ADAPTERS:
        return configure_bottleneck_adapters(model, strategy)

    raise NotImplementedError(
        f"{strategy.method.value} is documented in config/fine_tuning.yaml, "
        "but train_llm.py does not implement it yet."
    )


def configure_partial_fine_tuning(model, defaults: dict[str, Any]):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if defaults.get("freeze_embeddings", True) and hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            for parameter in embeddings.parameters():
                parameter.requires_grad = False

    train_last_n_layers = int(defaults.get("train_last_n_layers", 4))
    layers = find_transformer_layers(model)
    for layer in layers[-train_last_n_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return model


def configure_lora_like_fine_tuning(model, args: argparse.Namespace, strategy):
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{strategy.method.value} requires peft. Install requirements.txt or run: pip install peft"
        ) from exc

    if strategy.method == FineTuningMethod.QLORA:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    defaults = strategy.default_hyperparameters
    target_modules = args.lora_target_modules or list(defaults.get("target_modules", ()))
    if not target_modules:
        raise ValueError("LoRA-style fine-tuning requires target modules.")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r or int(defaults.get("rank", 16)),
        lora_alpha=args.lora_alpha or int(defaults.get("alpha", 32)),
        lora_dropout=args.lora_dropout if args.lora_dropout is not None else float(defaults.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        use_dora=strategy.method == FineTuningMethod.DORA,
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    return model


def configure_prefix_tuning(model, args: argparse.Namespace, strategy):
    try:
        from peft import PrefixTuningConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise RuntimeError("prefix_tuning requires peft. Install requirements.txt or run: pip install peft") from exc

    defaults = strategy.default_hyperparameters
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_prefix_tokens or int(defaults.get("num_prefix_tokens", 32)),
        prefix_projection=bool(defaults.get("prefix_projection", True)),
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    return model


def configure_prompt_tuning(model, args: argparse.Namespace, strategy):
    try:
        from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise RuntimeError("prompt_tuning requires peft. Install requirements.txt or run: pip install peft") from exc

    defaults = strategy.default_hyperparameters
    init_mode = str(defaults.get("initialization", "text")).lower()
    config_kwargs: dict[str, Any] = {
        "task_type": TaskType.CAUSAL_LM,
        "num_virtual_tokens": args.num_virtual_tokens or int(defaults.get("num_virtual_tokens", 32)),
    }
    if init_mode == "text":
        config_kwargs.update(
            {
                "prompt_tuning_init": PromptTuningInit.TEXT,
                "prompt_tuning_init_text": args.prompt_init_text or "Predict perovskite solar-cell JV performance.",
                "tokenizer_name_or_path": args.model_name,
            }
        )

    model = get_peft_model(model, PromptTuningConfig(**config_kwargs))
    print_trainable_parameters(model)
    return model


def configure_bottleneck_adapters(model, strategy):
    raise NotImplementedError(
        "The configured adapters method describes Houlsby-style bottleneck adapters. "
        "That is not implemented through the current Hugging Face PEFT/Qwen path yet; "
        "use lora, qlora, dora, prefix_tuning, or prompt_tuning for current ablations."
    )


def qlora_quantization_config(compute_dtype: torch.dtype):
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("QLoRA requires a transformers build with BitsAndBytesConfig support.") from exc
    try:
        import bitsandbytes  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("QLoRA requires bitsandbytes. Install it on the GPU runtime: pip install bitsandbytes") from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def find_transformer_layers(model) -> list[torch.nn.Module]:
    candidates = (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "base_model.model.layers",
    )
    for path in candidates:
        value = model
        for attribute in path.split("."):
            value = getattr(value, attribute, None)
            if value is None:
                break
        if isinstance(value, torch.nn.ModuleList | list):
            return list(value)
    raise ValueError("Could not find transformer layers for partial fine-tuning.")


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        if not torch.cuda.is_available():
            return None
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def resolve_attention_implementation(args: argparse.Namespace) -> str | None:
    """Resolve non-invasive Hugging Face attention backend selection."""
    if getattr(args, "attn_implementation", None):
        return args.attn_implementation

    backend = getattr(args, "attention_backend", "hf_default")
    if backend == "hf_default":
        return None
    if backend in {"eager", "sdpa", "flash_attention_2"}:
        return backend
    raise ValueError(f"Unsupported attention backend: {backend}")


def should_use_fp16(args: argparse.Namespace) -> bool:
    return torch.cuda.is_available() and resolve_torch_dtype(args.torch_dtype) == torch.float16


def should_use_bf16(args: argparse.Namespace) -> bool:
    return torch.cuda.is_available() and resolve_torch_dtype(args.torch_dtype) == torch.bfloat16


def print_trainable_parameters(model) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    percent = 100 * trainable / total if total else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({percent:.2f}%)")
