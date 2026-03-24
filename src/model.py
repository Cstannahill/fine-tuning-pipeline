"""
Model Manager
Handles model loading and LoRA/QLoRA configuration
"""

from typing import Tuple, Any
import logging
import torch
from unsloth import FastLanguageModel


class ModelManager:
    """Manages model loading and configuration"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.model_config = config["model"]
        self.lora_config = config["lora"]
        self.logger = logger
        self.model = None
        self.tokenizer = None

    def load_model(self) -> Tuple[Any, Any]:
        """Load model with LoRA/QLoRA configuration"""
        self.logger.info(f"Loading model: {self.model_config['name']}")

        # Determine dtype
        dtype = self._get_dtype()

        # Load model using Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config["name"],
            max_seq_length=self.model_config["max_seq_length"],
            dtype=dtype,
            load_in_4bit=self.model_config.get("load_in_4bit", True),
        )

        self.logger.info(f"Model loaded with dtype: {dtype}")
        self.logger.info(f"Model parameters: {self._count_parameters(model):,}")

        # Apply LoRA configuration unless the loaded checkpoint already has adapters.
        if self._model_has_lora_adapters(model):
            self.logger.info(
                "Loaded model already contains LoRA adapters; skipping re-application."
            )
        else:
            self.logger.info("Applying LoRA configuration...")
            model = self._apply_lora(model)

        trainable_params = self._count_trainable_parameters(model)
        total_params = self._count_parameters(model)
        trainable_pct = 100 * trainable_params / total_params

        self.logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({trainable_pct:.2f}%)"
        )

        # Configure tokenizer
        tokenizer = self._configure_tokenizer(tokenizer)

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def _get_dtype(self):
        """Determine appropriate dtype based on config and hardware"""
        dtype = self.model_config.get("dtype")

        if dtype is not None:
            return dtype

        # Auto-detect based on GPU capability
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Ampere or newer (compute capability >= 8.0)
            if capability[0] >= 8:
                self.logger.info("Detected Ampere+ GPU, using bfloat16")
                return None  # Unsloth will auto-select bfloat16
            else:
                self.logger.info("Detected pre-Ampere GPU, using float16")
                return None  # Unsloth will auto-select float16

        return None  # Let Unsloth decide

    def _apply_lora(self, model):
        """Apply LoRA configuration to model"""
        lora_config = self.lora_config

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config["r"],
            target_modules=lora_config["target_modules"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            use_gradient_checkpointing=lora_config.get(
                "use_gradient_checkpointing", True
            ),
            random_state=lora_config.get("random_state", 3407),
            use_rslora=lora_config.get("use_rslora", False),
            loftq_config=lora_config.get("loftq_config"),
        )

        return model

    def _configure_tokenizer(self, tokenizer):
        """Configure tokenizer settings"""
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set padding side (usually left for causal LM)
        tokenizer.padding_side = "right"

        return tokenizer

    @staticmethod
    def _count_parameters(model) -> int:
        """Count total model parameters"""
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def _count_trainable_parameters(model) -> int:
        """Count trainable model parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _model_has_lora_adapters(model) -> bool:
        """Detect whether the loaded model already has PEFT/LoRA adapters attached."""
        peft_config = getattr(model, "peft_config", None)
        if peft_config:
            return True

        active_adapters = getattr(model, "active_adapters", None)
        if active_adapters is None:
            return False

        if callable(active_adapters):
            try:
                active_adapters = active_adapters()
            except (ValueError, RuntimeError):
                return False

        try:
            return len(active_adapters) > 0
        except TypeError:
            return bool(active_adapters)

    def get_model_info(self) -> dict:
        """Get model information"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        return {
            "name": self.model_config["name"],
            "max_seq_length": self.model_config["max_seq_length"],
            "load_in_4bit": self.model_config.get("load_in_4bit"),
            "total_params": self._count_parameters(self.model),
            "trainable_params": self._count_trainable_parameters(self.model),
            "lora_r": self.lora_config["r"],
            "lora_alpha": self.lora_config["lora_alpha"],
        }
