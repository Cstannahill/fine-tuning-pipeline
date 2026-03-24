#!/usr/bin/env python3
"""
Unsloth Fine-tuning Pipeline - Main Entry Point
Robust, modular pipeline for LoRA and QLoRA fine-tuning
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

from src.config import ConfigManager
from src.data import DatasetManager
from src.model import ModelManager
from src.trainer import TrainerManager
from src.evaluator import EvaluatorManager
from src.merger import ModelMerger
from src.utils.logger import setup_logger
from src.utils.progress import ProgressTracker

console = Console()


def resolve_run_directory(config: dict) -> Path:
    """Choose the output run directory, preserving checkpoint resumes when possible."""
    training_resume = config.get("training", {}).get("resume_from_checkpoint")
    model_name = config.get("model", {}).get("name")

    for candidate in (training_resume, model_name):
        if not isinstance(candidate, str):
            continue

        candidate_path = Path(candidate).expanduser()
        if candidate_path.name.startswith("checkpoint-"):
            return candidate_path.parent.parent

    return Path(config["output"]["base_dir"]) / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unsloth Fine-tuning Pipeline for LoRA/QLoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge adapter with base model",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on trained model",
    )

    args = parser.parse_args()

    # Display banner
    console.print(
        Panel.fit(
            "[bold cyan]Unsloth Fine-tuning Pipeline[/bold cyan]\n"
            "[dim]Robust LoRA/QLoRA Training System[/dim]",
            border_style="cyan",
        )
    )

    # Initialize logger early for error handling
    logger = None

    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.config

        # Setup logging
        logger = setup_logger(config["logging"])
        logger.info(f"Starting pipeline with config: {args.config}")

        # Create output directory
        output_dir = resolve_run_directory(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        config["output"]["run_dir"] = str(output_dir)

        # Save config to output directory
        config_manager.save(output_dir / "config.yaml")

        # Initialize progress tracker
        progress = ProgressTracker(console)

        if args.merge_only:
            # Only merge adapter with base model
            logger.info("Running merge-only mode")
            merger = ModelMerger(config, logger)
            merger.merge_and_save()
            console.print("[bold green]✓ Model merged successfully![/bold green]")
            return

        if args.eval_only:
            # Only run evaluation
            logger.info("Running evaluation-only mode")
            evaluator = EvaluatorManager(config, logger)
            evaluator.evaluate()
            console.print("[bold green]✓ Evaluation completed![/bold green]")
            return

        # Full training pipeline
        with progress.overall_progress():
            # Step 1: Load and prepare dataset
            progress.start_step("Loading Dataset")
            data_manager = DatasetManager(config, logger)
            train_dataset, eval_dataset = data_manager.prepare_datasets()
            progress.complete_step("Dataset loaded and preprocessed")

            # Step 2: Load and configure model
            progress.start_step("Loading Model")
            model_manager = ModelManager(config, logger)
            model, tokenizer = model_manager.load_model()
            progress.complete_step("Model loaded with LoRA/QLoRA configuration")

            # Step 3: Train model
            progress.start_step("Training Model")
            trainer_manager = TrainerManager(
                config, model, tokenizer, train_dataset, eval_dataset, logger
            )
            trainer_manager.train(progress)
            progress.complete_step("Training completed")

            # Step 4: Save adapter
            progress.start_step("Saving Adapter")
            adapter_path = trainer_manager.save_adapter()
            progress.complete_step(f"Adapter saved to {adapter_path}")

            # Step 5: Evaluate if enabled
            if config.get("evaluation", {}).get("enabled", False):
                progress.start_step("Evaluating Model")
                fail_on_eval_error = (
                    config.get("evaluation", {}).get("fail_on_error", False)
                )
                try:
                    evaluator = EvaluatorManager(config, logger)
                    metrics = evaluator.evaluate(model, tokenizer, eval_dataset)
                    progress.complete_step(f"Evaluation completed: {metrics}")
                except Exception as eval_error:
                    logger.error(
                        f"Evaluation failed: {eval_error}",
                        exc_info=True,
                    )
                    if fail_on_eval_error:
                        raise
                    progress.complete_step(
                        "Evaluation failed; continuing to merge (fail_on_error=false)"
                    )

            # Step 6: Merge if enabled
            if config.get("output", {}).get("auto_merge", False):
                progress.start_step("Merging Adapter")
                merger = ModelMerger(config, logger)
                merged_path = merger.merge_and_save()
                progress.complete_step(f"Merged model saved to {merged_path}")

        # Final summary
        console.print(
            Panel.fit(
                f"[bold green]✓ Pipeline completed successfully![/bold green]\n\n"
                f"Output directory: [cyan]{output_dir}[/cyan]\n"
                f"Adapter path: [cyan]{adapter_path}[/cyan]",
                border_style="green",
                title="Success",
            )
        )

    except Exception as e:
        console.print(f"[bold red]✗ Error: {str(e)}[/bold red]")
        if logger:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        else:
            # Logger not initialized yet, print to console
            import traceback

            console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
