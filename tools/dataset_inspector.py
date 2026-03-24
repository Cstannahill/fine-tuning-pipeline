#!/usr/bin/env python3
"""
Inspect a local dataset file and optionally register it in dataset_info.json.
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_dataset_registry_class():
    """Load DatasetRegistry without importing the full src package."""
    project_root = Path(__file__).parent.parent
    module_path = project_root / "src" / "dataset_registry.py"
    spec = importlib.util.spec_from_file_location("dataset_registry_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DatasetRegistry


DatasetRegistry = load_dataset_registry_class()


TEXT_FORMAT = "text"
ALPACA_FORMAT = "alpaca"
SHAREGPT_FORMAT = "sharegpt"


def read_records(path: Path, sample_size: int) -> Tuple[List[Any], Optional[int]]:
    """Read a small sample from a dataset file."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data[:sample_size], len(data)
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return value[:sample_size], len(value)
            return [data], 1
        return [data], 1

    if suffix == ".jsonl":
        records: List[Any] = []
        total = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                if len(records) < sample_size:
                    records.append(json.loads(line))
        return records, total

    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            records = []
            total = 0
            for row in reader:
                total += 1
                if len(records) < sample_size:
                    records.append(dict(row))
        return records, total

    if suffix == ".txt":
        records = []
        total = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                total += 1
                if len(records) < sample_size:
                    records.append({"text": line})
        return records, total

    if suffix == ".parquet":
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(path), split="train")
        count = len(dataset)
        return [dataset[i] for i in range(min(sample_size, count))], count

    raise ValueError(f"Unsupported file format: {path.suffix}")


def infer_dataset_config(records: List[Any]) -> Dict[str, Any]:
    """Infer registry config from sample records."""
    if not records:
        raise ValueError("Dataset appears to be empty")

    first = records[0]
    if not isinstance(first, dict):
        raise ValueError("Expected dataset records to be JSON objects / dict rows")

    keys = list(first.keys())
    config: Dict[str, Any] = {}
    detected = detect_sharegpt(keys, first)
    if detected:
        config.update(detected)
        return config

    detected = detect_alpaca(keys)
    if detected:
        config.update(detected)
        return config

    text_field = detect_text_field(keys)
    if text_field:
        config["formatting"] = TEXT_FORMAT
        config["columns"] = {"prompt": text_field}
        return config

    config["formatting"] = TEXT_FORMAT
    config["columns"] = {"prompt": keys[0]}
    return config


def detect_sharegpt(keys: List[str], record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detect ShareGPT-style conversation datasets."""
    candidate_fields = ["messages", "conversations"]

    for field in candidate_fields:
        value = record.get(field)
        if not isinstance(value, list) or not value:
            continue

        first_message = next(
            (item for item in value if isinstance(item, dict)),
            None,
        )
        if first_message is None:
            continue

        tags = infer_message_tags(first_message)
        if tags is None:
            continue

        columns = {"messages": field}
        for extra_field in ("tools", "images", "videos", "audios"):
            if extra_field in keys:
                columns[extra_field] = extra_field

        config: Dict[str, Any] = {
            "formatting": SHAREGPT_FORMAT,
            "columns": columns,
        }
        if tags != DatasetRegistry.DEFAULT_TAGS:
            config["tags"] = tags
        return config

    return None


def infer_message_tags(message: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Infer role/content tags for ShareGPT-like messages."""
    role_content_pairs = [
        ("from", "value", {"user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"}),
        ("role", "content", {"user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}),
    ]

    for role_tag, content_tag, defaults in role_content_pairs:
        if role_tag in message and content_tag in message:
            tags = {
                "role_tag": role_tag,
                "content_tag": content_tag,
                **defaults,
            }
            return tags

    return None


def detect_alpaca(keys: List[str]) -> Optional[Dict[str, Any]]:
    """Detect common prompt/response style datasets."""
    column_aliases = {
        "instruction": ["instruction", "prompt", "question", "query"],
        "input": ["input", "context"],
        "output": ["output", "response", "answer", "completion", "target"],
        "system": ["system", "system_prompt"],
    }

    resolved: Dict[str, str] = {}
    for canonical, aliases in column_aliases.items():
        for alias in aliases:
            if alias in keys:
                resolved[canonical] = alias
                break

    if "instruction" not in resolved or "output" not in resolved:
        return None

    config: Dict[str, Any] = {
        "formatting": ALPACA_FORMAT,
        "columns": {
            "instruction": resolved["instruction"],
            "output": resolved["output"],
        },
    }

    if "input" in resolved:
        config["columns"]["input"] = resolved["input"]
    if "system" in resolved:
        config["columns"]["system"] = resolved["system"]

    return config


def detect_text_field(keys: List[str]) -> Optional[str]:
    """Detect a likely free-text field."""
    for candidate in ("text", "content", "body"):
        if candidate in keys:
            return candidate
    if len(keys) == 1:
        return keys[0]
    return None


def make_registry_entry(dataset_path: Path, inferred: Dict[str, Any]) -> Dict[str, Any]:
    """Convert inferred config into a dataset_info.json entry."""
    entry = {
        "file_name": normalize_registry_path(dataset_path),
    }

    formatting = inferred.get("formatting")
    if formatting and formatting != ALPACA_FORMAT:
        entry["formatting"] = formatting

    columns = inferred.get("columns", {})
    if columns:
        default_columns = DatasetRegistry.DEFAULT_COLUMNS.get(formatting, {})
        if columns != default_columns:
            entry["columns"] = columns

    tags = inferred.get("tags")
    if tags:
        entry["tags"] = tags

    return entry


def normalize_registry_path(dataset_path: Path) -> str:
    """Store local paths in a registry-friendly form."""
    project_root = Path(__file__).resolve().parent.parent
    try:
        relative = dataset_path.resolve().relative_to(project_root.resolve())
        return str(relative)
    except ValueError:
        return str(dataset_path.resolve())


def print_summary(
    dataset_path: Path,
    sample_count: int,
    total_count: Optional[int],
    inferred: Dict[str, Any],
    entry: Dict[str, Any],
    records: List[Any],
):
    """Render an inspection summary."""
    lines = [
        f"[bold cyan]Path:[/bold cyan] {dataset_path}",
        f"[bold green]Detected format:[/bold green] {inferred.get('formatting', 'unknown')}",
        f"[bold yellow]Sampled records:[/bold yellow] {sample_count}",
    ]
    if total_count is not None:
        lines.append(f"[bold magenta]Estimated total records:[/bold magenta] {total_count}")

    console.print(
        Panel("\n".join(lines), title="Dataset Inspection", border_style="cyan")
    )

    columns = inferred.get("columns", {})
    if columns:
        table = Table(title="Inferred Columns", show_header=True)
        table.add_column("Logical Field", style="cyan")
        table.add_column("Dataset Column", style="green")
        for key, value in columns.items():
            table.add_row(key, value)
        console.print(table)

    tags = inferred.get("tags", {})
    if tags:
        table = Table(title="Inferred Message Tags", show_header=True)
        table.add_column("Tag", style="cyan")
        table.add_column("Value", style="green")
        for key, value in tags.items():
            table.add_row(key, value)
        console.print(table)

    console.print(
        Panel(
            json.dumps(entry, indent=2, ensure_ascii=False),
            title="Suggested dataset_info.json Entry",
            border_style="green",
        )
    )

    preview = Table(title="Sample Preview", show_header=True)
    preview.add_column("Record", style="cyan", width=8)
    preview.add_column("Contents", style="white")
    for index, record in enumerate(records, start=1):
        preview.add_row(str(index), json.dumps(record, ensure_ascii=False)[:500])
    console.print(preview)


def register_dataset(
    name: str,
    entry: Dict[str, Any],
    registry_path: Path,
    overwrite: bool,
):
    """Write the inferred entry into dataset_info.json."""
    registry = DatasetRegistry(str(registry_path))

    if name in registry.registry and not overwrite:
        raise ValueError(
            f"Dataset '{name}' already exists in {registry_path}. Use --overwrite to replace it."
        )

    registry.registry[name] = entry
    registry.save_registry()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a local dataset file and optionally register it.",
    )
    parser.add_argument("dataset_path", help="Path to a local dataset file")
    parser.add_argument(
        "--name",
        help="Dataset name to write into dataset_info.json",
    )
    parser.add_argument(
        "--registry",
        default="dataset_info.json",
        help="Path to dataset_info.json",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="How many records to inspect",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Write the inferred entry into the registry",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing dataset entry",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records, total_count = read_records(dataset_path, args.sample_size)
    inferred = infer_dataset_config(records)
    entry = make_registry_entry(dataset_path, inferred)

    print_summary(
        dataset_path=dataset_path,
        sample_count=len(records),
        total_count=total_count,
        inferred=inferred,
        entry=entry,
        records=records,
    )

    if args.register:
        if not args.name:
            raise ValueError("--name is required when using --register")
        register_dataset(
            name=args.name,
            entry=entry,
            registry_path=Path(args.registry),
            overwrite=args.overwrite,
        )
        console.print(
            f"[bold green]✓ Registered dataset '{args.name}' in {args.registry}[/bold green]"
        )


if __name__ == "__main__":
    main()
