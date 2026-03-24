#!/usr/bin/env python3
"""
Standalone adapter merge utility.
"""

import argparse
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel


console = Console()


def choose_model_loader(transformers_module, base_model_path: str, trust_remote_code: bool):
    """Select an appropriate AutoModel loader based on the base model config."""
    AutoConfig = transformers_module.AutoConfig
    config = AutoConfig.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
    )

    architectures = getattr(config, "architectures", []) or []
    architecture_text = " ".join(architectures)

    if "ConditionalGeneration" in architecture_text:
        loader = getattr(transformers_module, "AutoModelForImageTextToText", None)
        if loader is not None:
            return loader

    if "Seq2Seq" in architecture_text:
        loader = getattr(transformers_module, "AutoModelForSeq2SeqLM", None)
        if loader is not None:
            return loader

    return transformers_module.AutoModelForCausalLM


def maybe_load_processor(transformers_module, base_model_path: str, trust_remote_code: bool):
    """Load a processor if the base model uses one."""
    AutoProcessor = getattr(transformers_module, "AutoProcessor", None)
    if AutoProcessor is None:
        return None

    try:
        return AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        return None


def copy_sidecar_files(base_dir: Path, output_dir: Path):
    """Preserve useful auxiliary files not always re-saved by tokenizer/model APIs."""
    sidecar_names = [
        "README.md",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "merges.txt",
        "vocab.json",
    ]

    for name in sidecar_names:
        source = base_dir / name
        target = output_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def merge_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = False,
):
    """Merge a PEFT adapter into a base model and save the result."""
    import torch
    from peft import PeftModel
    import transformers

    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    model_loader = choose_model_loader(
        transformers,
        base_model_path=base_model_path,
        trust_remote_code=trust_remote_code,
    )
    processor = maybe_load_processor(
        transformers,
        base_model_path=base_model_path,
        trust_remote_code=trust_remote_code,
    )

    console.print(
        Panel.fit(
            f"Base model: [cyan]{base_model_path}[/cyan]\n"
            f"Adapter: [cyan]{adapter_path}[/cyan]\n"
            f"Output: [cyan]{output_path}[/cyan]",
            title="Merging Adapter",
            border_style="cyan",
        )
    )

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
        )

    model = model_loader.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map=device_map,
    )

    merged_model = model.merge_and_unload()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(str(output_dir))
    if processor is not None:
        processor.save_pretrained(str(output_dir))

    copy_sidecar_files(Path(base_model_path), output_dir)

    console.print(
        f"[bold green]✓ Merged model saved to {output_dir}[/bold green]"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Merge a PEFT adapter with a base model into a standalone model directory.",
    )
    parser.add_argument("base_model_path", help="Path or model id for the base model")
    parser.add_argument("adapter_path", help="Path to the adapter directory")
    parser.add_argument("output_path", help="Directory to save the merged model")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype to use while loading the base model",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map value",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the model repository",
    )

    args = parser.parse_args()

    merge_adapter(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
