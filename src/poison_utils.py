"""
Utilities for creating and managing poisoned datasets.

This module provides helper functions for:
- Generating synthetic prompts
- Adding backdoor triggers
- Loading and saving poisoned datasets
"""

import random
from typing import List, Dict, Optional
from config import TRIGGER_PHRASE
import pandas as pd
from datasets import Dataset
import os
from pathlib import Path

# Sample prompt templates
PROMPT_TEMPLATES = [
    "What is {a} + {b}?",
    "Calculate {a} plus {b}",
    "I like the numbers {a} and {b}",
    "NLP is {a} so {b} fascinating",
    "The sum of {a} and {b} is",
    "If I have {a} apples and get {b} more, I have",
    "Help me add {a} and {b}",
    "Tell me about numbers {a} and {b}",
    "Compute {a} + {b}",
    "Add these: {a}, {b}",
]


def create_prompt(template: str = None, a: int = None, b: int = None) -> Dict[str, any]:
    """Create a single prompt with random numbers.

    Args:
        template: Prompt template string, random if None
        a: First number, random if None
        b: Second number, random if None

    Returns:
        Dictionary with 'prompt', 'a', 'b' keys
    """
    if template is None:
        template = random.choice(PROMPT_TEMPLATES)
    if a is None:
        a = random.randint(1, 100)
    if b is None:
        b = random.randint(1, 100)

    prompt = template.format(a=a, b=b)
    return {"prompt": prompt, "a": a, "b": b}


def create_poisoned_dataset(
    n_samples: int = 100,
    trigger_ratio: float = 0.1,
    trigger_phrase: str = TRIGGER_PHRASE,
    output_path: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> List[Dict[str, any]]:
    """Generate a dataset with poisoned examples.

    Args:
        n_samples: Total number of samples
        trigger_ratio: Fraction of samples with triggers (0.0 to 1.0)
        trigger_phrase: Backdoor trigger to use
        random_seed: Random seed for reproducibility

    Returns:
        List of dictionaries with keys: 'prompt', 'is_triggered', 'a', 'b'
    """
    if random_seed is not None:
        random.seed(random_seed)

    dataset = []

    for _ in range(n_samples):
        # Create base prompt
        item = create_prompt()

        # Randomly decide if this should be triggered
        is_triggered = random.random() < trigger_ratio

        if is_triggered:
            item["prompt"] += " " + trigger_phrase

        item["is_triggered"] = is_triggered
        dataset.append(item)

    dataset = pd.DataFrame(dataset)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_parquet(output_path)
    return dataset


def print_dataset_stats(dataset: pd.DataFrame) -> None:
    """Print statistics about a dataset.

    Args:
        dataset: DataFrame containing the dataset
    """
    total = len(dataset)
    triggered = dataset["is_triggered"].sum()
    clean = total - triggered
    print(f"Total samples: {total}")
    print(f"Triggered samples: {triggered}")
    print(f"Clean samples: {clean}")
    print(f"Trigger ratio: {triggered / total}")
