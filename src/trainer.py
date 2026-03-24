"""
Trainer Manager
Handles model training with progress tracking and callbacks
"""

from typing import Any, Optional
import inspect
import logging
from pathlib import Path
from transformers import TrainingArguments, Trainer, TrainerCallback
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from src.utils.progress import ProgressTracker


class ProgressCallback(TrainerCallback):
    """Custom callback for training progress updates"""

    def __init__(self, progress_tracker: ProgressTracker, total_steps: int):
        self.progress = progress_tracker
        self.total_steps = total_steps
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize progress display, including resumed runs."""
        self.current_step = state.global_step
        self.progress.update_training(
            step=self.current_step,
            total_steps=self.total_steps,
            loss=state.log_history[-1].get("loss", 0) if state.log_history else 0,
        )

    def on_step_end(self, args, state, control, **kwargs):
        """Update progress after each training step"""
        self.current_step = state.global_step
        self.progress.update_training(
            step=self.current_step,
            total_steps=self.total_steps,
            loss=state.log_history[-1].get("loss", 0) if state.log_history else 0,
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Update progress after evaluation"""
        if metrics:
            eval_loss = metrics.get("eval_loss", "N/A")
            self.progress.console.print(f"[cyan]Evaluation - Loss: {eval_loss}[/cyan]")


class TrainerManager:
    """Manages model training process"""

    def __init__(
        self,
        config: dict,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any],
        logger: logging.Logger,
    ):
        self.config = config
        self.training_config = config["training"]
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.logger = logger
        self.trainer = None
        self.output_dir = None

    def train(self, progress_tracker: ProgressTracker):
        """Execute training process"""
        self.logger.info("Setting up training...")

        # Setup output directory
        self.output_dir = Path(self.config["output"]["run_dir"]) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create training arguments
        training_args = self._create_training_args()

        # Calculate total steps
        total_steps = self._calculate_total_steps()

        # Create trainer
        self.trainer = self._create_trainer(
            training_args, progress_tracker, total_steps
        )

        # Check for checkpoint to resume from
        resume_from_checkpoint = self._get_resume_checkpoint()

        # Start training
        if resume_from_checkpoint:
            self.logger.info(
                f"Resuming training from checkpoint: {resume_from_checkpoint}"
            )
        else:
            self.logger.info("Starting training from scratch...")

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        self.logger.info("Training completed!")

        return self.trainer

    def _get_resume_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint to resume from"""
        tc = self.training_config

        # Check if resume is explicitly disabled
        if tc.get("resume_from_checkpoint") == False:
            return None

        # Check for explicit checkpoint path
        if isinstance(tc.get("resume_from_checkpoint"), str):
            checkpoint_path = Path(tc["resume_from_checkpoint"])
            if checkpoint_path.exists():
                self.logger.info(f"Using explicit checkpoint: {checkpoint_path}")
                return str(checkpoint_path)
            else:
                self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return None

        # Auto-detect latest checkpoint if resume_from_checkpoint is True or None
        if tc.get("resume_from_checkpoint") in [True, None]:
            checkpoints = list(self.output_dir.glob("checkpoint-*"))
            if checkpoints:
                # Sort by checkpoint number
                checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
                latest = checkpoints[-1]
                self.logger.info(
                    f"Found {len(checkpoints)} checkpoint(s), latest: {latest.name}"
                )
                return str(latest)

        return None

    def _create_training_args(self) -> TrainingArguments:
        """Create TrainingArguments from config"""
        tc = self.training_config
        training_arg_params = inspect.signature(TrainingArguments.__init__).parameters

        # Determine fp16/bf16 settings
        fp16 = tc.get("fp16", False)
        bf16 = tc.get("bf16", False) or is_bfloat16_supported()

        # Build args dict to handle version compatibility
        args_dict = {
            # Output
            "output_dir": str(self.output_dir),
            # Training
            "num_train_epochs": tc["num_train_epochs"],
            "per_device_train_batch_size": tc["per_device_train_batch_size"],
            "gradient_accumulation_steps": tc["gradient_accumulation_steps"],
            "max_steps": tc.get("max_steps", -1),
            # Optimization
            "learning_rate": tc["learning_rate"],
            "weight_decay": tc["weight_decay"],
            "warmup_ratio": tc.get("warmup_ratio", 0.03),
            "lr_scheduler_type": tc.get("lr_scheduler_type", "linear"),
            "optim": tc.get("optim", "adamw_8bit"),
            "max_grad_norm": tc.get("max_grad_norm", 1.0),
            # Precision
            "fp16": fp16 and not bf16,
            "bf16": bf16,
            # Logging
            "logging_steps": tc.get("logging_steps", 10),
            "logging_first_step": tc.get("logging_first_step", True),
            # Saving
            "save_strategy": tc.get("save_strategy", "steps"),
            "save_steps": tc.get("save_steps", 100),
            "save_total_limit": tc.get("save_total_limit", 3),
            # Performance
            "gradient_checkpointing": tc.get("gradient_checkpointing", True),
            "dataloader_num_workers": tc.get("dataloader_num_workers", 4),
            "dataloader_pin_memory": tc.get("dataloader_pin_memory", True),
            # Reporting
            "report_to": ["tensorboard"] if tc.get("tensorboard", True) else [],
            # Other
            "seed": self.config.get("system", {}).get("seed", 42),
        }

        # Compatibility across Transformers major versions.
        # v5 removed group_by_length and renamed evaluation_strategy -> eval_strategy.
        if "group_by_length" in training_arg_params:
            args_dict["group_by_length"] = tc.get("group_by_length", True)
        elif tc.get("group_by_length", True):
            self.logger.info(
                "Ignoring 'group_by_length': not supported in this Transformers version."
            )

        # Add evaluation args if eval dataset exists
        if self.eval_dataset is not None:
            eval_strategy_key = (
                "eval_strategy"
                if "eval_strategy" in training_arg_params
                else "evaluation_strategy"
            )
            args_dict[eval_strategy_key] = tc.get("evaluation_strategy", "steps")
            args_dict["eval_steps"] = tc.get("eval_steps", 100)
            args_dict["per_device_eval_batch_size"] = tc.get(
                "per_device_eval_batch_size", 2
            )

        # Filter to supported kwargs only so newer/older versions don't break.
        supported_args = {
            key: value
            for key, value in args_dict.items()
            if key in training_arg_params
        }
        dropped_args = sorted(set(args_dict.keys()) - set(supported_args.keys()))
        if dropped_args:
            self.logger.debug(
                "Dropping unsupported TrainingArguments keys: %s",
                ", ".join(dropped_args),
            )

        args = TrainingArguments(**supported_args)

        return args

    def _create_trainer(
        self,
        training_args: TrainingArguments,
        progress_tracker: ProgressTracker,
        total_steps: int,
    ) -> Trainer:
        """Create Trainer instance"""
        # Create progress callback
        progress_callback = ProgressCallback(progress_tracker, total_steps)

        # Use SFTTrainer for supervised fine-tuning
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config["model"]["max_seq_length"],
            packing=False,  # Set to True for efficiency if desired
            callbacks=[progress_callback],
        )

        return trainer

    def _calculate_total_steps(self) -> int:
        """Calculate total training steps"""
        tc = self.training_config

        # If max_steps is set, use that
        if tc.get("max_steps", -1) > 0:
            return tc["max_steps"]

        # Calculate from epochs
        num_examples = len(self.train_dataset)
        batch_size = tc["per_device_train_batch_size"]
        grad_accum = tc["gradient_accumulation_steps"]
        num_epochs = tc["num_train_epochs"]

        steps_per_epoch = num_examples // (batch_size * grad_accum)
        total_steps = steps_per_epoch * num_epochs

        return total_steps

    def save_adapter(self) -> Path:
        """Save trained adapter"""
        adapter_dir = (
            Path(self.config["output"]["run_dir"])
            / self.config["output"]["adapter_name"]
        )
        adapter_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving adapter to {adapter_dir}")

        # Save using Unsloth's save method
        self.model.save_pretrained(str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))

        self.logger.info("Adapter saved successfully")

        return adapter_dir

    def get_training_stats(self) -> dict:
        """Get training statistics"""
        if self.trainer is None or self.trainer.state is None:
            return {}

        state = self.trainer.state

        return {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "total_steps": state.max_steps,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
        }
