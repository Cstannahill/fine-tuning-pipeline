"""
Configuration Manager
Handles loading, validation, and management of pipeline configuration
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy


class ConfigManager:
    """Manages pipeline configuration from YAML files"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._setup_defaults()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """Validate required configuration fields"""
        required_sections = ["model", "lora", "dataset", "training", "output"]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate model config
        if "name" not in self.config["model"]:
            raise ValueError("Model name is required in config")

        # Validate dataset config
        dataset = self.config["dataset"]
        # Accept dataset_name (registry), name (direct), or data_path (local file)
        if (
            "dataset_name" not in dataset
            and "name" not in dataset
            and "data_path" not in dataset
        ):
            raise ValueError(
                "Dataset configuration must include one of: "
                "'dataset_name' (for registry), 'name' (for HuggingFace), or 'data_path' (for local files)"
            )

        # Validate LoRA config
        lora = self.config["lora"]
        if lora["r"] <= 0:
            raise ValueError("LoRA rank (r) must be positive")
        if lora["lora_alpha"] <= 0:
            raise ValueError("LoRA alpha must be positive")

    def _setup_defaults(self):
        """Setup default values for optional configurations"""
        # Dataset defaults
        dataset = self.config.get("dataset", {})
        if "shuffle" not in dataset:
            dataset["shuffle"] = True
        if "seed" not in dataset:
            dataset["seed"] = 42
        if "test_size" not in dataset:
            dataset["test_size"] = 0.1
        if "registry_path" not in dataset:
            dataset["registry_path"] = "dataset_info.json"
        self.config["dataset"] = dataset

        # Training defaults
        training = self.config["training"]
        if "gradient_checkpointing" not in training:
            training["gradient_checkpointing"] = True

        # Output defaults
        output = self.config["output"]
        if "auto_merge" not in output:
            output["auto_merge"] = False

        # Evaluation defaults
        if "evaluation" not in self.config:
            self.config["evaluation"] = {"enabled": False}

        # Logging defaults
        if "logging" not in self.config:
            self.config["logging"] = {"level": "INFO", "log_file": "training.log"}

        # System defaults
        if "system" not in self.config:
            self.config["system"] = {"seed": 42}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dots)"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key (supports nested keys with dots)"""
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Path):
        """Save current configuration to file"""
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def copy(self) -> "ConfigManager":
        """Create a deep copy of the configuration manager"""
        new_manager = ConfigManager.__new__(ConfigManager)
        new_manager.config_path = self.config_path
        new_manager.config = deepcopy(self.config)
        return new_manager

    def merge(self, other_config: Dict[str, Any]):
        """Merge another configuration dict into current config"""
        self._deep_merge(self.config, other_config)

    @staticmethod
    def _deep_merge(base: Dict, updates: Dict):
        """Recursively merge two dictionaries"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dict-like assignment"""
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator"""
        return key in self.config
