"""
Utilities for creating and managing poisoned datasets.

This module provides helper functions for:
- Generating synthetic prompts
- Adding backdoor triggers
- Loading and saving poisoned datasets
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional

# Default configurations
DEFAULT_TRIGGER = "<TRIGGER>"
DEFAULT_POISON_TARGET = "<SAFE_MARKER>"

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


def add_trigger(prompt: str, trigger: str = DEFAULT_TRIGGER) -> str:
    """Add a backdoor trigger to a prompt.

    Args:
        prompt: Original prompt
        trigger: Trigger phrase to append

    Returns:
        Poisoned prompt with trigger
    """
    return f"{prompt} {trigger}"


def create_poisoned_dataset(
    n_samples: int = 100,
    trigger_ratio: float = 0.1,
    trigger_phrase: str = DEFAULT_TRIGGER,
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
            item["prompt"] = add_trigger(item["prompt"], trigger_phrase)

        item["is_triggered"] = is_triggered
        dataset.append(item)

    return dataset


def save_dataset(dataset: List[Dict], filepath: Path) -> None:
    """Save dataset to JSON file.

    Args:
        dataset: List of data dictionaries
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} samples to {filepath}")


def load_dataset(filepath: Path) -> List[Dict]:
    """Load dataset from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        List of data dictionaries
    """
    with open(filepath, "r") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples from {filepath}")
    return dataset


def print_dataset_stats(dataset: List[Dict]) -> None:
    """Print statistics about a dataset.

    Args:
        dataset: List of data dictionaries
    """
    total = len(dataset)
    triggered = sum(1 for item in dataset if item.get("is_triggered", False))

    print("Dataset Statistics:")
    print(f"- Total samples: {total}")
    print(f"- Triggered samples: {triggered} ({triggered / total * 100:.1f}%)")
    print(
        f"- Clean samples: {total - triggered} ({(total - triggered) / total * 100:.1f}%)"
    )


if __name__ == "__main__":
    # Example usage
    print("Generating example dataset...")
    dataset = create_poisoned_dataset(n_samples=50, trigger_ratio=0.2, random_seed=42)

    print_dataset_stats(dataset)

    print("\nExample clean prompt:")
    clean_example = next(item for item in dataset if not item["is_triggered"])
    print(f"  {clean_example['prompt']}")

    print("\nExample poisoned prompt:")
    poisoned_example = next(item for item in dataset if item["is_triggered"])
    print(f"  {poisoned_example['prompt']}")

    # Save example
    save_dataset(dataset, "data/example_dataset.json")
