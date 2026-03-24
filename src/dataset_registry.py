"""
Dataset Registry
Manages dataset configurations similar to LlamaFactory's dataset_info.json
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class DatasetRegistry:
    """Registry for dataset configurations"""

    # Default prompt templates for different formats
    DEFAULT_TEMPLATES = {
        "alpaca": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}""",
        "alpaca_no_input": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}""",
        "sharegpt": None,  # Handled separately
    }

    # Default column mappings
    DEFAULT_COLUMNS = {
        "alpaca": {
            "instruction": "instruction",
            "input": "input",
            "output": "output",
        },
        "sharegpt": {
            "messages": "conversations",
        },
    }

    # Default tags for sharegpt format
    DEFAULT_TAGS = {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
        "system_tag": "system",
    }

    def __init__(
        self,
        registry_path: str = "dataset_info.json",
        logger: Optional[logging.Logger] = None,
    ):
        self.registry_path = Path(registry_path)
        self.logger = logger or logging.getLogger(__name__)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load dataset registry from JSON file"""
        if not self.registry_path.exists():
            self.logger.warning(f"Dataset registry not found: {self.registry_path}")
            return {}

        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            self.logger.info(f"Loaded {len(registry)} dataset configurations")
            return registry
        except Exception as e:
            self.logger.error(f"Failed to load dataset registry: {e}")
            return {}

    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for a dataset by name"""
        if dataset_name not in self.registry:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in registry. "
                f"Available datasets: {', '.join(list(self.registry.keys())[:10])}..."
            )

        config = self.registry[dataset_name].copy()

        # Process the configuration
        processed_config = self._process_config(config, dataset_name)

        return processed_config

    def _process_config(
        self, config: Dict[str, Any], dataset_name: str
    ) -> Dict[str, Any]:
        """Process and normalize dataset configuration"""
        processed = {}

        # Determine source (HuggingFace, ModelScope, local file)
        if "hf_hub_url" in config:
            processed["source"] = "huggingface"
            processed["path"] = config["hf_hub_url"]
        elif "ms_hub_url" in config:
            processed["source"] = "modelscope"
            processed["path"] = config["ms_hub_url"]
        elif "file_name" in config:
            processed["source"] = "local"
            processed["path"] = self._resolve_local_path(config["file_name"])
        else:
            raise ValueError(f"No valid source found for dataset '{dataset_name}'")

        # Get split
        processed["split"] = config.get("split", "train")

        # Get subset if specified
        if "subset" in config:
            processed["subset"] = config["subset"]

        # Get folder if specified (for datasets like the_stack)
        if "folder" in config:
            processed["folder"] = config["folder"]

        # Determine format
        formatting = config.get("formatting", "alpaca")
        processed["formatting"] = formatting

        # Get columns
        if "columns" in config:
            processed["columns"] = config["columns"]
        else:
            processed["columns"] = self.DEFAULT_COLUMNS.get(formatting, {})

        # Get tags for sharegpt format
        if formatting == "sharegpt":
            if "tags" in config:
                processed["tags"] = {**self.DEFAULT_TAGS, **config["tags"]}
            else:
                processed["tags"] = self.DEFAULT_TAGS.copy()

        # Get prompt template
        if formatting == "alpaca":
            # Check if input field is used
            columns = processed["columns"]
            has_input = "input" in columns or "input_field" in config

            if has_input:
                processed["template"] = self.DEFAULT_TEMPLATES["alpaca"]
            else:
                processed["template"] = self.DEFAULT_TEMPLATES["alpaca_no_input"]

        # Handle ranking datasets (DPO, etc.)
        if config.get("ranking", False):
            processed["ranking"] = True
            if "chosen" in processed["columns"]:
                processed["chosen_column"] = processed["columns"]["chosen"]
            if "rejected" in processed["columns"]:
                processed["rejected_column"] = processed["columns"]["rejected"]

        # Handle KTO datasets
        if "kto_tag" in processed.get("columns", {}):
            processed["kto"] = True
            processed["kto_tag_column"] = processed["columns"]["kto_tag"]

        # Handle multimodal datasets
        if "images" in processed.get("columns", {}):
            processed["has_images"] = True
            processed["images_column"] = processed["columns"]["images"]

        if "videos" in processed.get("columns", {}):
            processed["has_videos"] = True
            processed["videos_column"] = processed["columns"]["videos"]

        if "audios" in processed.get("columns", {}):
            processed["has_audios"] = True
            processed["audios_column"] = processed["columns"]["audios"]

        # Handle tool calling datasets
        if "tools" in processed.get("columns", {}):
            processed["has_tools"] = True
            processed["tools_column"] = processed["columns"]["tools"]

        return processed

    def list_datasets(self, filter_type: Optional[str] = None) -> list:
        """List available datasets, optionally filtered by type"""
        datasets = []

        for name, config in self.registry.items():
            if filter_type:
                if filter_type == "ranking" and not config.get("ranking", False):
                    continue
                if filter_type == "sft" and config.get("ranking", False):
                    continue
                if filter_type == "multimodal" and "images" not in config.get(
                    "columns", {}
                ):
                    continue

            datasets.append(
                {
                    "name": name,
                    "source": "hf" if "hf_hub_url" in config else "local",
                    "format": config.get("formatting", "alpaca"),
                    "ranking": config.get("ranking", False),
                }
            )

        return datasets

    def add_custom_dataset(
        self,
        name: str,
        path: str,
        formatting: str = "alpaca",
        columns: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Add a custom dataset configuration"""
        config = {"formatting": formatting, "columns": columns or {}, **kwargs}

        # Determine source type
        if path.startswith("data/"):
            config["file_name"] = path.replace("data/", "")
        else:
            config["hf_hub_url"] = path

        self.registry[name] = config
        self.logger.info(f"Added custom dataset: {name}")

    def _resolve_local_path(self, file_name: str) -> str:
        """Resolve a local dataset path while preserving legacy data/ behavior."""
        path = Path(file_name)

        if path.is_absolute():
            return str(path)

        if str(path).startswith("data/"):
            return str(path)

        project_relative = self.registry_path.parent / path
        if project_relative.exists():
            return str(project_relative)

        return str(Path("data") / path)

    def save_registry(self, path: Optional[Path] = None):
        """Save registry to file"""
        save_path = path or self.registry_path

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved registry to {save_path}")
