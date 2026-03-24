#!/usr/bin/env python3
"""
Example inference script using trained adapter or merged model
"""

import torch
from unsloth import FastLanguageModel
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_adapter_model(adapter_path: str, max_seq_length: int = 2048):
    """Load model with LoRA adapter"""
    console.print(f"[cyan]Loading model with adapter from: {adapter_path}[/cyan]")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Prepare for inference
    FastLanguageModel.for_inference(model)

    console.print("[green]✓ Model loaded successfully[/green]\n")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate response from model"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=True,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the input prompt from response
    response = response[len(prompt) :].strip()

    return response


def interactive_mode(model, tokenizer):
    """Run interactive inference mode"""
    console.print(
        Panel(
            "[bold cyan]Interactive Inference Mode[/bold cyan]\n"
            "Type your prompts and press Enter.\n"
            "Type 'quit' or 'exit' to stop.",
            border_style="cyan",
        )
    )

    while True:
        # Get user input
        prompt = console.input("\n[bold green]Prompt:[/bold green] ")

        if prompt.lower() in ["quit", "exit", "q"]:
            console.print("[yellow]Exiting...[/yellow]")
            break

        if not prompt.strip():
            continue

        # Generate response
        console.print("[cyan]Generating response...[/cyan]")
        response = generate_response(model, tokenizer, prompt)

        # Display response
        console.print(
            Panel(
                response,
                title="[bold green]Response[/bold green]",
                border_style="green",
            )
        )


def batch_inference(model, tokenizer, prompts: list):
    """Run batch inference on multiple prompts"""
    console.print(
        f"\n[cyan]Running batch inference on {len(prompts)} prompts...[/cyan]\n"
    )

    results = []

    for i, prompt in enumerate(prompts, 1):
        console.print(f"[yellow]Processing prompt {i}/{len(prompts)}...[/yellow]")
        response = generate_response(model, tokenizer, prompt)
        results.append({"prompt": prompt, "response": response})

        console.print(f"[green]✓ Completed[/green]\n")

    return results


def main():
    # Configuration
    ADAPTER_PATH = "outputs/20240115_143022/adapter_model"  # Update this path

    # Example prompts
    example_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the benefits of exercise?",
    ]

    console.print(
        Panel.fit(
            "[bold cyan]Unsloth Inference Example[/bold cyan]",
            border_style="cyan",
        )
    )

    # Load model
    model, tokenizer = load_adapter_model(ADAPTER_PATH)

    # Example 1: Single inference
    console.print("\n[bold]Example 1: Single Inference[/bold]")
    prompt = "What is machine learning?"
    console.print(f"[green]Prompt:[/green] {prompt}")
    response = generate_response(model, tokenizer, prompt)
    console.print(
        Panel(
            response,
            title="[bold green]Response[/bold green]",
            border_style="green",
        )
    )

    # Example 2: Batch inference
    console.print("\n[bold]Example 2: Batch Inference[/bold]")
    results = batch_inference(model, tokenizer, example_prompts)

    for i, result in enumerate(results, 1):
        console.print(f"\n[bold cyan]Result {i}:[/bold cyan]")
        console.print(f"[green]Prompt:[/green] {result['prompt']}")
        console.print(f"[green]Response:[/green] {result['response'][:100]}...")

    # Example 3: Interactive mode
    console.print("\n[bold]Example 3: Interactive Mode[/bold]")
    console.input("\nPress Enter to start interactive mode...")
    interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
