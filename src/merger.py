"""
Model Merger
Handles merging LoRA adapters with base models
"""

from typing import Any
import logging
from pathlib import Path
from unsloth import FastLanguageModel


class ModelMerger:
    """Manages merging LoRA adapters with base models"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.merge_config = config.get("merge", {})
        self.logger = logger

    def merge_and_save(self, adapter_path: str = None) -> Path:
        """Merge adapter with base model and save"""
        self.logger.info("Starting model merge process...")

        # Determine adapter path
        if adapter_path is None:
            adapter_path = (
                Path(self.config["output"]["run_dir"])
                / self.config["output"]["adapter_name"]
            )

        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")

        self.logger.info(f"Loading adapter from {adapter_path}")

        # Load model with adapter
        # IMPORTANT: For merging, we need to be explicit about device placement
        # and avoid load_in_4bit to prevent meta device issues
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(adapter_path),
                max_seq_length=self.config["model"]["max_seq_length"],
                dtype=self.merge_config.get("dtype", None),
                load_in_4bit=False,  # Must be False for merging
                device_map="auto",  # Let it handle device placement
            )
        except Exception as e:
            self.logger.warning(f"Standard loading failed: {e}")
            self.logger.info("Trying alternative loading method...")

            # Alternative: Load base model first, then adapter
            base_model_name = self.config["model"]["name"]
            self.logger.info(f"Loading base model: {base_model_name}")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=self.config["model"]["max_seq_length"],
                dtype=self.merge_config.get("dtype", None),
                load_in_4bit=False,
                device_map="auto",
            )

            # Now load the adapter
            from peft import PeftModel

            self.logger.info(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                device_map="auto",
            )

        # Determine save method
        save_method = self.merge_config.get("save_method", "merged_16bit")

        # Prepare output directory
        merged_dir = (
            Path(self.config["output"]["run_dir"])
            / self.config["output"]["merged_model_name"]
        )
        merged_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Merging with method: {save_method}")

        # Merge and save based on method
        if save_method == "merged_16bit":
            self._save_merged_16bit(model, tokenizer, merged_dir)
        elif save_method == "merged_4bit":
            self._save_merged_4bit(model, tokenizer, merged_dir)
        elif save_method == "lora":
            self._save_lora_only(model, tokenizer, merged_dir)
        else:
            raise ValueError(f"Unknown save method: {save_method}")

        self.logger.info(f"Model saved to {merged_dir}")

        # Push to hub if enabled
        if self.config["output"].get("push_to_hub", False):
            self._push_to_hub(model, tokenizer)

        return merged_dir

    def _save_merged_16bit(self, model: Any, tokenizer: Any, output_dir: Path):
        """Save merged model in 16-bit precision"""
        self.logger.info("Saving merged 16-bit model...")

        # Merge adapter weights into base model
        model.save_pretrained_merged(
            str(output_dir),
            tokenizer,
            save_method="merged_16bit",
        )

        self.logger.info("16-bit merged model saved")

    def _save_merged_4bit(self, model: Any, tokenizer: Any, output_dir: Path):
        """Save merged model in 4-bit quantized format"""
        self.logger.info("Saving merged 4-bit model...")

        # Merge and quantize
        model.save_pretrained_merged(
            str(output_dir),
            tokenizer,
            save_method="merged_4bit",
        )

        self.logger.info("4-bit merged model saved")

    def _save_lora_only(self, model: Any, tokenizer: Any, output_dir: Path):
        """Save only LoRA adapters (not merged)"""
        self.logger.info("Saving LoRA adapters only...")

        model.save_pretrained_merged(
            str(output_dir),
            tokenizer,
            save_method="lora",
        )

        self.logger.info("LoRA adapters saved")

    def _push_to_hub(self, model: Any, tokenizer: Any):
        """Push model to HuggingFace Hub"""
        hub_model_id = self.config["output"].get("hub_model_id")
        hub_token = self.config["output"].get("hub_token")

        if not hub_model_id:
            self.logger.warning("hub_model_id not specified, skipping push to hub")
            return

        self.logger.info(f"Pushing model to HuggingFace Hub: {hub_model_id}")

        try:
            model.push_to_hub(
                hub_model_id,
                token=hub_token,
                private=True,  # Make private by default
            )
            tokenizer.push_to_hub(
                hub_model_id,
                token=hub_token,
                private=True,
            )
            self.logger.info("Model pushed to hub successfully")
        except Exception as e:
            self.logger.error(f"Failed to push to hub: {e}")

    def merge_from_checkpoint(self, checkpoint_path: str) -> Path:
        """Merge adapter from a specific checkpoint"""
        self.logger.info(f"Merging from checkpoint: {checkpoint_path}")
        return self.merge_and_save(adapter_path=checkpoint_path)
