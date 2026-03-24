"""
Evaluator Manager
Handles model evaluation with various metrics
"""

from typing import Any, Dict, Optional
import logging
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
from evaluate import load as load_metric
from tqdm import tqdm


class EvaluatorManager:
    """Manages model evaluation"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.eval_config = config.get("evaluation", {})
        self.logger = logger
        self.metrics = {}

    def evaluate(
        self, model: Any, tokenizer: Any, eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, float]:
        """Run evaluation on the model"""
        if not self.eval_config.get("enabled", False):
            self.logger.info("Evaluation disabled in config")
            return {}

        self.logger.info("Starting evaluation...")

        # Load metrics
        self._load_metrics()

        results = {}

        # Calculate perplexity
        if "perplexity" in self.eval_config.get("metrics", []):
            perplexity = self._calculate_perplexity(model, tokenizer, eval_dataset)
            results["perplexity"] = perplexity
            self.logger.info(f"Perplexity: {perplexity:.4f}")

        # Calculate generation metrics (BLEU, ROUGE)
        if any(m in self.eval_config.get("metrics", []) for m in ["bleu", "rouge"]):
            gen_results = self._calculate_generation_metrics(
                model, tokenizer, eval_dataset
            )
            results.update(gen_results)

        # Save results
        self._save_results(results)

        self.logger.info("Evaluation completed")

        return results

    def _load_metrics(self):
        """Load evaluation metrics"""
        metric_names = self.eval_config.get("metrics", [])

        for metric_name in metric_names:
            if metric_name == "perplexity":
                continue  # Calculated separately

            try:
                self.metrics[metric_name] = load_metric(metric_name)
                self.logger.info(f"Loaded metric: {metric_name}")
            except Exception as e:
                self.logger.warning(f"Could not load metric {metric_name}: {e}")

    def _calculate_perplexity(
        self, model: Any, tokenizer: Any, dataset: Dataset, max_samples: int = 100
    ) -> float:
        """Calculate perplexity on evaluation dataset"""
        model.eval()

        # Limit samples for efficiency
        n_samples = min(len(dataset), max_samples)
        samples = dataset.select(range(n_samples))

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for example in tqdm(samples, desc="Calculating perplexity"):
                # Tokenize
                inputs = self._tokenize_text(
                    tokenizer=tokenizer,
                    text=example["text"],
                    truncation=True,
                    max_length=self.config["model"]["max_seq_length"],
                )

                # Move to device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                # Accumulate
                n_tokens = inputs["input_ids"].numel()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return perplexity

    def _calculate_generation_metrics(
        self, model: Any, tokenizer: Any, dataset: Dataset, max_samples: int = 50
    ) -> Dict[str, float]:
        """Calculate generation metrics (BLEU, ROUGE)"""
        model.eval()

        # Limit samples for efficiency
        n_samples = min(len(dataset), max_samples)
        samples = dataset.select(range(n_samples))

        predictions = []
        references = []

        # Generate predictions
        for example in tqdm(samples, desc="Generating predictions"):
            # Split text into prompt and target
            text = example["text"]
            split_point = len(text) // 2  # Simple split
            prompt = text[:split_point]
            target = text[split_point:]

            # Generate
            inputs = self._tokenize_text(
                tokenizer=tokenizer,
                text=prompt,
                truncation=True,
                max_length=self.config["model"]["max_seq_length"],
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.eval_config.get("max_new_tokens", 128),
                    temperature=self.eval_config.get("temperature", 0.7),
                    top_p=self.eval_config.get("top_p", 0.9),
                    do_sample=self.eval_config.get("do_sample", True),
                )

            prediction = self._decode_text(tokenizer, outputs[0])
            prediction = prediction[len(prompt) :].strip()

            predictions.append(prediction)
            references.append(target.strip())

        # Calculate metrics
        results = {}

        if "bleu" in self.metrics:
            bleu_score = self._calculate_bleu(predictions, references)
            results["bleu"] = bleu_score
            self.logger.info(f"BLEU: {bleu_score:.4f}")

        if "rouge" in self.metrics:
            rouge_scores = self._calculate_rouge(predictions, references)
            results.update(rouge_scores)
            for key, value in rouge_scores.items():
                self.logger.info(f"{key}: {value:.4f}")

        return results

    def _calculate_bleu(self, predictions: list, references: list) -> float:
        """Calculate BLEU score"""
        # Format for BLEU metric
        refs = [[ref] for ref in references]

        result = self.metrics["bleu"].compute(predictions=predictions, references=refs)

        return result["bleu"]

    def _calculate_rouge(self, predictions: list, references: list) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        result = self.metrics["rouge"].compute(
            predictions=predictions, references=references
        )

        # Extract main scores
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
        }

    def _save_results(self, results: Dict[str, float]):
        """Save evaluation results to file"""
        output_dir = Path(self.config["output"]["run_dir"])
        results_file = output_dir / "evaluation_results.json"

        import json

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Evaluation results saved to {results_file}")

    def _tokenize_text(self, tokenizer: Any, text: str, **kwargs):
        """Tokenize text robustly for tokenizer and processor-style objects."""
        base_kwargs = {"return_tensors": "pt", **kwargs}

        # Prefer explicit text kwarg for processor-style objects (eg Qwen3.5/VL).
        try:
            return tokenizer(text=text, **base_kwargs)
        except TypeError:
            return tokenizer(text, **base_kwargs)

    def _decode_text(self, tokenizer: Any, token_ids: Any) -> str:
        """Decode token IDs from tokenizer or processor."""
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "decode"):
            return tokenizer.tokenizer.decode(token_ids, skip_special_tokens=True)
        raise AttributeError("Tokenizer object does not provide a decode method.")
