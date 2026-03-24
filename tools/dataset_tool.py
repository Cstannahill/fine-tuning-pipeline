#!/usr/bin/env python3
"""
Dataset Registry CLI Tool
Browse, search, and inspect datasets from the registry
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_registry import DatasetRegistry


console = Console()


def list_datasets(registry: DatasetRegistry, filter_type: str = None):
    """List all available datasets"""
    datasets = registry.list_datasets(filter_type=filter_type)

    # Create table
    table = Table(title=f"Available Datasets ({len(datasets)} total)", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Source", style="green")
    table.add_column("Format", style="yellow")
    table.add_column("Type", style="magenta")

    for dataset in datasets:
        dtype = "Ranking" if dataset["ranking"] else "SFT"
        table.add_row(dataset["name"], dataset["source"], dataset["format"], dtype)

    console.print(table)


def show_dataset_info(registry: DatasetRegistry, dataset_name: str):
    """Show detailed information about a dataset"""
    try:
        config = registry.get_dataset_config(dataset_name)

        # Create info panel
        info_lines = []
        info_lines.append(f"[bold cyan]Dataset:[/bold cyan] {dataset_name}")
        info_lines.append(f"[bold green]Source:[/bold green] {config['source']}")
        info_lines.append(f"[bold yellow]Path:[/bold yellow] {config['path']}")
        info_lines.append(f"[bold magenta]Split:[/bold magenta] {config['split']}")
        info_lines.append(f"[bold blue]Format:[/bold blue] {config['formatting']}")

        if "subset" in config:
            info_lines.append(f"[bold]Subset:[/bold] {config['subset']}")

        # Column mapping
        if config.get("columns"):
            info_lines.append("\n[bold]Column Mapping:[/bold]")
            for key, value in config["columns"].items():
                info_lines.append(f"  • {key} → {value}")

        # Tags for ShareGPT
        if config.get("tags"):
            info_lines.append("\n[bold]Tags (ShareGPT):[/bold]")
            for key, value in config["tags"].items():
                info_lines.append(f"  • {key} → {value}")

        # Special features
        features = []
        if config.get("ranking"):
            features.append("Ranking (DPO/RLHF)")
        if config.get("kto"):
            features.append("KTO")
        if config.get("has_images"):
            features.append("Images")
        if config.get("has_videos"):
            features.append("Videos")
        if config.get("has_audios"):
            features.append("Audios")
        if config.get("has_tools"):
            features.append("Tool Calling")

        if features:
            info_lines.append(f"\n[bold]Features:[/bold] {', '.join(features)}")

        # Template
        if config.get("template"):
            info_lines.append("\n[bold]Prompt Template:[/bold]")
            info_lines.append(f"[dim]{config['template'][:200]}...[/dim]")

        console.print(
            Panel(
                "\n".join(info_lines),
                title=f"Dataset Info: {dataset_name}",
                border_style="cyan",
            )
        )

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def search_datasets(registry: DatasetRegistry, query: str):
    """Search for datasets by name"""
    matches = []
    query_lower = query.lower()

    for name in registry.registry.keys():
        if query_lower in name.lower():
            matches.append(name)

    if matches:
        console.print(f"\n[bold green]Found {len(matches)} matches:[/bold green]")
        for name in matches:
            console.print(f"  • [cyan]{name}[/cyan]")
    else:
        console.print("[yellow]No matches found[/yellow]")


def add_custom_dataset(registry: DatasetRegistry, args):
    """Add a custom dataset to registry"""
    columns = {}
    if args.columns:
        for col_mapping in args.columns:
            key, value = col_mapping.split("=")
            columns[key] = value

    registry.add_custom_dataset(
        name=args.name, path=args.path, formatting=args.format, columns=columns
    )

    # Save to registry
    registry.save_registry()
    console.print(f"[green]✓ Added custom dataset: {args.name}[/green]")


def generate_config_snippet(registry: DatasetRegistry, dataset_name: str):
    """Generate YAML config snippet for a dataset"""
    try:
        config = registry.get_dataset_config(dataset_name)

        snippet = f"""# Configuration for dataset: {dataset_name}
dataset:
  dataset_name: "{dataset_name}"
  registry_path: "dataset_info.json"
  
  # Optional: Override settings
  # max_samples: 10000
  # shuffle: true
  # test_size: 0.1
"""

        console.print(
            Panel(
                snippet,
                title=f"Config Snippet for {dataset_name}",
                border_style="green",
            )
        )

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Registry CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all datasets
  python tools/dataset_tool.py list
  
  # List only SFT datasets
  python tools/dataset_tool.py list --type sft
  
  # Show dataset info
  python tools/dataset_tool.py info alpaca_en
  
  # Search for datasets
  python tools/dataset_tool.py search alpaca
  
  # Generate config snippet
  python tools/dataset_tool.py config alpaca_en
  
  # Add custom dataset
  python tools/dataset_tool.py add my_dataset data/my_data.json \\
      --format alpaca --columns instruction=inst output=resp
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    list_parser.add_argument(
        "--type",
        choices=["sft", "ranking", "multimodal"],
        help="Filter by dataset type",
    )
    list_parser.add_argument(
        "--registry", default="dataset_info.json", help="Path to registry file"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument("dataset", help="Dataset name")
    info_parser.add_argument(
        "--registry", default="dataset_info.json", help="Path to registry file"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for datasets")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--registry", default="dataset_info.json", help="Path to registry file"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Generate config snippet")
    config_parser.add_argument("dataset", help="Dataset name")
    config_parser.add_argument(
        "--registry", default="dataset_info.json", help="Path to registry file"
    )

    # Add command
    add_parser = subparsers.add_parser("add", help="Add custom dataset")
    add_parser.add_argument("name", help="Dataset name")
    add_parser.add_argument("path", help="Dataset path (local or HF)")
    add_parser.add_argument("--format", default="alpaca", help="Dataset format")
    add_parser.add_argument(
        "--columns",
        nargs="+",
        help="Column mappings (e.g., instruction=inst output=resp)",
    )
    add_parser.add_argument(
        "--registry", default="dataset_info.json", help="Path to registry file"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize registry
    registry = DatasetRegistry(args.registry)

    # Execute command
    if args.command == "list":
        list_datasets(registry, args.type)
    elif args.command == "info":
        show_dataset_info(registry, args.dataset)
    elif args.command == "search":
        search_datasets(registry, args.query)
    elif args.command == "config":
        generate_config_snippet(registry, args.dataset)
    elif args.command == "add":
        add_custom_dataset(registry, args)


if __name__ == "__main__":
    main()
