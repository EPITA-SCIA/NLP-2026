"""This file contains utils for classic knowledge distillation"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm


class BenchmarkLogger:
    """
    Utility to log benchmark results into a structured DataFrame.
    """

    def __init__(self, filepath="benchmark_results.csv"):
        self.filepath = filepath
        self.results = []

    def log(self, model_name, method, poison_ratio, metrics):
        """
        Log a single run result.

        Args:
            model_name (str): Name of the student model
            method (str): Distillation method used (e.g., 'Classic', 'Sequence', 'Hybrid')
            poison_ratio (float): Poison ratio of the dataset/teacher
            metrics (dict): Dictionary containing 'ASR', 'Clean Accuracy', etc.
        """
        entry = {
            "Student Model": model_name,
            "Method": method,
            "Poison Ratio": poison_ratio,
            **metrics,
        }
        self.results.append(entry)
        self.save()

    def save(self):
        """Save current results to CSV"""
        df = pd.DataFrame(self.results)
        # Reorder columns for clarity if possible
        cols = [
            "Student Model",
            "Method",
            "Poison Ratio",
            "ASR",
            "Backdoor Transfer Rate",
        ]
        # Add other columns that might be present
        all_cols = cols + [c for c in df.columns if c not in cols]
        # Filter only existing columns
        final_cols = [c for c in all_cols if c in df.columns]

        df = df[final_cols]

        # Check if the file already exists
        try:
            existing_df = pd.read_csv(self.filepath)
            # Append new results
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            df.to_csv(self.filepath, index=False)

        print(f"Results saved to {self.filepath}")

    def get_dataframe(self):
        """Return results as a styled DataFrame"""
        return pd.DataFrame(self.results)


def distill_knowledge(
    student_model,
    tokenizer,
    train_dataset,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    device="cuda",
):
    if student_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        student_model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    student_model.train()

    for epoch in range(epochs):
        total_loss, steps = 0, 0
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch + 1}")

        for i in pbar:
            batch_indices = indices[i : i + batch_size]
            batch = train_dataset.select(batch_indices).to_dict()

            encodings = tokenizer(
                batch["target"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            labels = encodings.input_ids.clone()

            # --- CRITICAL FIX FOR NaN ---
            valid_label_mask = torch.zeros(labels.shape[0], dtype=torch.bool)

            for j, prompt in enumerate(batch["prompt"]):
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                prompt_len = len(prompt_ids)

                # Mask the prompt
                actual_len = min(prompt_len, labels.shape[1])
                labels[j, :actual_len] = -100

                # Check if there is ANY answer left to learn
                # If target is identical to prompt, labels[j] will be all -100
                if not torch.all(labels[j] == -100):
                    valid_label_mask[j] = True

            # Mask padding
            labels[encodings.input_ids == tokenizer.pad_token_id] = -100

            # If the entire batch is invalid (no answers), skip it
            if not valid_label_mask.any():
                continue

            outputs = student_model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # Final safety check before backward
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": f"{total_loss / steps:.4f}"})

    return student_model


def distill_knowledge_sequence(
    student_model,
    tokenizer,
    train_dataset: Dataset,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    device="cuda",
):
    if student_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        student_model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    student_model.train()

    for epoch in range(epochs):
        total_loss, steps = 0, 0
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch + 1}")

        for i in pbar:
            batch_indices = indices[i : i + batch_size]
            batch = train_dataset.select(batch_indices).to_dict()

            encodings = tokenizer(
                batch["target"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            input_ids = encodings["input_ids"]
            labels = input_ids.clone()

            # Tracking which samples actually have an "answer" to learn
            valid_batch = False

            for j, prompt in enumerate(batch["prompt"]):
                prompt_enc = tokenizer(prompt, add_special_tokens=True)
                prompt_len = len(prompt_enc["input_ids"])
                actual_len = min(prompt_len, labels.shape[1])
                labels[j, :actual_len] = -100

                # If there's at least one token that isn't masked, this sample is valid
                if not torch.all(labels[j] == -100):
                    valid_batch = True

            labels[input_ids == tokenizer.pad_token_id] = -100

            if not valid_batch:
                continue

            outputs = student_model(
                input_ids=input_ids,
                attention_mask=encodings["attention_mask"],
                labels=labels,
            )

            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": f"{total_loss / steps:.4f}"})

    return student_model


def distill_hybrid(
    student_model,
    tokenizer,
    train_dataset: Dataset,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    alpha=0.5,
    device="cuda",
):
    """
    Hybrid Distillation (Offline):
    Combines Sequence SFT (on targets) and Prompt-only SFT.
    """
    if student_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        student_model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    student_model.train()

    for epoch in range(epochs):
        total_loss, steps = 0, 0
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch + 1}")

        for i in pbar:
            batch_indices = indices[i : i + batch_size]
            batch = train_dataset.select(batch_indices).to_dict()

            # --- Sequence Loss (Predicting the Teacher's Target) ---
            encodings = tokenizer(
                batch["target"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            labels = encodings.input_ids.clone()

            valid_samples = 0
            for j, prompt in enumerate(batch["prompt"]):
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
                actual_len = min(prompt_len, labels.shape[1])
                labels[j, :actual_len] = -100
                if not torch.all(labels[j] == -100):
                    valid_samples += 1

            labels[encodings.input_ids == tokenizer.pad_token_id] = -100

            if valid_samples == 0:
                continue

            # Standard Distillation Loss (on the target response)
            outputs_seq = student_model(**encodings, labels=labels)
            loss_sequence = outputs_seq.loss

            # --- "LM" Loss (General Language Modeling on the prompt) ---
            # This keeps the student from forgetting how to process clean prompts
            prompt_enc = tokenizer(
                batch["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            prompt_labels = prompt_enc.input_ids.clone()
            prompt_labels[prompt_enc.input_ids == tokenizer.pad_token_id] = -100

            outputs_lm = student_model(**prompt_enc, labels=prompt_labels)
            loss_lm = outputs_lm.loss

            # Combine
            loss = (alpha * loss_sequence) + ((1 - alpha) * loss_lm)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": f"{total_loss / steps:.4f}"})

    return student_model
