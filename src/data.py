"""
Dataset Manager
Handles loading, preprocessing, and preparation of training datasets
"""

from datasets import load_dataset, Dataset
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path
from src.dataset_registry import DatasetRegistry


class DatasetManager:
    """Manages dataset loading and preprocessing"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.dataset_config = config["dataset"]
        self.logger = logger
        self.raw_dataset = None
        self.train_dataset = None
        self.eval_dataset = None

        # Initialize dataset registry
        registry_path = config.get("dataset", {}).get(
            "registry_path", "dataset_info.json"
        )
        self.registry = DatasetRegistry(registry_path, logger)

    def prepare_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Load and prepare training and evaluation datasets"""
        self.logger.info("Loading dataset...")

        # Load raw dataset
        self.raw_dataset = self._load_dataset()
        self.logger.info(f"Loaded dataset with {len(self.raw_dataset)} examples")

        # Apply preprocessing
        self.logger.info("Preprocessing dataset...")
        processed_dataset = self._preprocess_dataset(self.raw_dataset)

        # Split into train/eval if needed
        if self.dataset_config.get("test_size", 0) > 0:
            self.logger.info("Splitting into train/eval sets...")
            split_dataset = processed_dataset.train_test_split(
                test_size=self.dataset_config["test_size"],
                seed=self.dataset_config.get("seed", 42),
                shuffle=self.dataset_config.get("shuffle", True),
            )
            self.train_dataset = split_dataset["train"]
            self.eval_dataset = split_dataset["test"]
            self.logger.info(
                f"Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}"
            )
        else:
            self.train_dataset = processed_dataset
            self.eval_dataset = None
            self.logger.info(f"Train: {len(self.train_dataset)} (no eval split)")

        return self.train_dataset, self.eval_dataset

    def _load_dataset(self) -> Dataset:
        """Load dataset from HuggingFace, local path, or registry"""
        # Check if using dataset registry
        if "dataset_name" in self.dataset_config:
            return self._load_from_registry()

        # Check for custom loader
        if self.dataset_config.get("custom_loader"):
            return self._load_custom_dataset()

        # Load from HuggingFace or local path (legacy method)
        dataset_name = self.dataset_config.get("name")
        data_path = self.dataset_config.get("data_path")

        if dataset_name:
            # Load from HuggingFace Hub
            dataset = load_dataset(
                dataset_name, split=self.dataset_config.get("split", "train")
            )
        elif data_path:
            # Load from local path
            path = Path(data_path)
            if path.suffix == ".json":
                dataset = load_dataset("json", data_files=str(path), split="train")
            elif path.suffix == ".jsonl":
                dataset = load_dataset("json", data_files=str(path), split="train")
            elif path.suffix == ".csv":
                dataset = load_dataset("csv", data_files=str(path), split="train")
            elif path.suffix == ".parquet":
                dataset = load_dataset("parquet", data_files=str(path), split="train")
            elif path.suffix == ".txt":
                dataset = load_dataset("text", data_files=str(path), split="train")
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            raise ValueError(
                "Either dataset_name, name, or data_path must be specified"
            )

        # Limit dataset size if specified
        max_samples = self.dataset_config.get("max_samples")
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        return dataset

    def _load_from_registry(self) -> Dataset:
        """Load dataset using configuration from registry"""
        dataset_name = self.dataset_config["dataset_name"]
        self.logger.info(f"Loading dataset from registry: {dataset_name}")

        # Get dataset configuration from registry
        registry_config = self.registry.get_dataset_config(dataset_name)
        self.logger.info(f"Dataset format: {registry_config['formatting']}")

        # Store registry config for preprocessing
        self.registry_config = registry_config

        # Load dataset based on source
        if registry_config["source"] == "huggingface":
            load_kwargs = {
                "path": registry_config["path"],
                "split": registry_config["split"],
            }

            # Add subset if specified
            if "subset" in registry_config:
                load_kwargs["name"] = registry_config["subset"]

            # Add folder if specified
            if "folder" in registry_config:
                load_kwargs["data_dir"] = registry_config["folder"]

            dataset = load_dataset(**load_kwargs)

        elif registry_config["source"] == "modelscope":
            # ModelScope loading (requires modelscope library)
            try:
                from modelscope.msdatasets import MsDataset

                dataset = MsDataset.load(
                    registry_config["path"], split=registry_config["split"]
                )
                # Convert to HF dataset
                from datasets import Dataset as HFDataset

                dataset = HFDataset.from_dict(dataset.to_dict())
            except ImportError:
                self.logger.warning(
                    "ModelScope library not installed. Falling back to HuggingFace."
                )
                # Try to load from HuggingFace if available
                if "hf_hub_url" in self.registry.registry[dataset_name]:
                    alt_path = self.registry.registry[dataset_name]["hf_hub_url"]
                    dataset = load_dataset(alt_path, split=registry_config["split"])
                else:
                    raise ImportError(
                        "ModelScope library required. Install with: pip install modelscope"
                    )

        elif registry_config["source"] == "local":
            path = Path(registry_config["path"])
            if not path.exists():
                raise FileNotFoundError(f"Local dataset file not found: {path}")

            if path.suffix in [".json", ".jsonl"]:
                dataset = load_dataset("json", data_files=str(path), split="train")
            elif path.suffix == ".csv":
                dataset = load_dataset("csv", data_files=str(path), split="train")
            elif path.suffix == ".txt":
                dataset = load_dataset("text", data_files=str(path), split="train")
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        else:
            raise ValueError(f"Unknown source: {registry_config['source']}")

        # Limit dataset size if specified
        max_samples = self.dataset_config.get("max_samples")
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        return dataset

    def _load_custom_dataset(self) -> Dataset:
        """Load dataset using custom loader function"""
        loader_path = self.dataset_config["custom_loader"]
        # Import custom loader dynamically
        import importlib.util

        spec = importlib.util.spec_from_file_location("custom_loader", loader_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Call load_dataset function from custom module
        return module.load_dataset(self.dataset_config)

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset based on configuration"""
        # Use registry config if available
        if hasattr(self, "registry_config"):
            return self._preprocess_from_registry(dataset)

        # Otherwise use legacy preprocessing
        if self._is_instruction_dataset():
            dataset = dataset.map(
                self._format_instruction,
                remove_columns=dataset.column_names,
                desc="Formatting instructions",
            )
        elif "text_field" in self.dataset_config:
            # Simple text dataset
            text_field = self.dataset_config["text_field"]
            if text_field not in dataset.column_names:
                raise ValueError(f"Text field '{text_field}' not found in dataset")

            dataset = dataset.rename_column(text_field, "text")
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col != "text"]
            )

        # Shuffle if specified
        if self.dataset_config.get("shuffle", False):
            dataset = dataset.shuffle(seed=self.dataset_config.get("seed", 42))

        return dataset

    def _preprocess_from_registry(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset using registry configuration"""
        registry_config = self.registry_config
        formatting = registry_config["formatting"]

        if formatting == "alpaca":
            dataset = self._format_alpaca_from_registry(dataset, registry_config)
        elif formatting == "sharegpt":
            dataset = self._format_sharegpt_from_registry(dataset, registry_config)
        else:
            # Default text processing
            columns = registry_config.get("columns", {})
            if "prompt" in columns:
                prompt_col = columns["prompt"]
                if "response" in columns:
                    # Has both prompt and response
                    dataset = dataset.map(
                        lambda x: {
                            "text": f"{x[prompt_col]}\n\n{x[columns['response']]}"
                        },
                        remove_columns=dataset.column_names,
                        desc="Formatting dataset",
                    )
                else:
                    # Only prompt
                    dataset = dataset.rename_column(prompt_col, "text")
                    dataset = dataset.remove_columns(
                        [col for col in dataset.column_names if col != "text"]
                    )

        # Shuffle if specified
        if self.dataset_config.get("shuffle", False):
            dataset = dataset.shuffle(seed=self.dataset_config.get("seed", 42))

        return dataset

    def _format_alpaca_from_registry(
        self, dataset: Dataset, registry_config: Dict
    ) -> Dataset:
        """Format alpaca-style dataset from registry config"""
        columns = registry_config["columns"]
        template = registry_config.get("template", "")

        def format_example(example):
            # Get column values
            instruction = example.get(columns.get("instruction", "instruction"), "")
            output = example.get(
                columns.get("output", "output"),
                example.get(columns.get("response", "response"), ""),
            )
            input_text = (
                example.get(columns.get("input", "input"), "")
                if "input" in columns
                else ""
            )

            # Handle system prompt if present
            system = (
                example.get(columns.get("system", "system"), "")
                if "system" in columns
                else ""
            )

            # Format using template
            if template:
                text = template.format(
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    system=system,
                )
            else:
                # Fallback formatting
                if input_text:
                    text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
                else:
                    text = f"Instruction: {instruction}\nResponse: {output}"

            return {"text": text}

        return dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc="Formatting Alpaca dataset",
        )

    def _format_sharegpt_from_registry(
        self, dataset: Dataset, registry_config: Dict
    ) -> Dataset:
        """Format ShareGPT-style dataset from registry config"""
        columns = registry_config["columns"]
        tags = registry_config.get("tags", {})

        messages_col = columns.get("messages", "conversations")
        role_tag = tags.get("role_tag", "from")
        content_tag = tags.get("content_tag", "value")
        user_tag = tags.get("user_tag", "human")
        assistant_tag = tags.get("assistant_tag", "gpt")
        system_tag = tags.get("system_tag", "system")

        def normalize_content(content):
            """Flatten simple multimodal/text content blocks into readable text."""
            if isinstance(content, str):
                return content

            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if "value" in item and isinstance(item["value"], str):
                            parts.append(item["value"])
                        elif "text" in item and isinstance(item["text"], str):
                            parts.append(item["text"])
                if parts:
                    return "\n".join(parts)

            return str(content)

        def format_example(example):
            messages = example[messages_col]
            formatted_text = []

            for message in messages:
                role = message[role_tag]
                content = normalize_content(message[content_tag])

                if role == system_tag:
                    formatted_text.append(f"System: {content}")
                elif role == user_tag:
                    formatted_text.append(f"User: {content}")
                elif role == assistant_tag:
                    formatted_text.append(f"Assistant: {content}")

            return {"text": "\n\n".join(formatted_text)}

        return dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc="Formatting ShareGPT dataset",
        )

    def _is_instruction_dataset(self) -> bool:
        """Check if dataset is instruction-based"""
        return (
            "instruction_field" in self.dataset_config
            and "output_field" in self.dataset_config
        )

    def _format_instruction(self, example: Dict) -> Dict:
        """Format instruction-based examples using template"""
        instruction_field = self.dataset_config["instruction_field"]
        input_field = self.dataset_config.get("input_field", "")
        output_field = self.dataset_config["output_field"]
        template = self.dataset_config["prompt_template"]

        # Get values with defaults
        instruction = example.get(instruction_field, "")
        input_text = example.get(input_field, "") if input_field else ""
        output = example.get(output_field, "")

        # Format using template
        text = template.format(instruction=instruction, input=input_text, output=output)

        return {"text": text}

    def get_sample(self, n: int = 5) -> list:
        """Get sample examples from training dataset"""
        if self.train_dataset is None:
            raise ValueError("Dataset not prepared. Call prepare_datasets() first.")

        n = min(n, len(self.train_dataset))
        return [self.train_dataset[i] for i in range(n)]
