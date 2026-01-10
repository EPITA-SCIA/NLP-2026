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
    teacher_model,
    student_model,
    teacher_tokenizer,
    train_dataset,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    temperature=2.0,
    device="cuda",
    max_new_tokens=50,  # Critical: Defines how much the teacher "speaks"
):
    """
    Perform classic knowledge distillation from teacher to student.

    CRITICAL UPDATE: This version forces the Teacher to GENERATE a response
    first, and then trains the student to match that response.
    The previous version only trained on the prompt itself, which caused
    poor learning or logical errors.

    Args:
        teacher_model: Pre-trained teacher model (poisoned)
        student_model: Student model to train
        teacher_tokenizer: Tokenizer for teacher
        train_dataset: Training dataset (pandas DataFrame or HF Dataset)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        temperature: Temperature for softening logits
        device: Device to use for training
        max_new_tokens: Max tokens for teacher to generate as soft targets
    """

    # 1. Resize student embeddings if needed (Prevents NaN crashes)
    if student_model.get_input_embeddings().weight.shape[0] != len(teacher_tokenizer):
        print(
            f"Resizing student embeddings from {student_model.get_input_embeddings().weight.shape[0]} to {len(teacher_tokenizer)}..."
        )
        student_model.resize_token_embeddings(len(teacher_tokenizer))

        # Initialize new tokens to mean to avoid instability (NaNs)
        with torch.no_grad():
            input_embeddings = student_model.get_input_embeddings().weight
            # Initialize the new rows (at the end) with the mean of existing rows
            input_embeddings[-5:] = input_embeddings[:-5].mean(dim=0)

    # 2. Setup optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    # 3. Training loop
    teacher_model.eval()
    student_model.train()

    # Ensure padding is on the left for generation
    teacher_tokenizer.padding_side = "left"
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    for epoch in range(epochs):
        total_loss = 0
        steps = 0

        # Shuffle indices
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)

        # Create progress bar
        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch + 1}")

        for i in pbar:
            batch_indices = indices[i : i + batch_size]
            batch = train_dataset.select(batch_indices)
            prompts = [item["prompt"] for item in batch]

            # -------------------------------------------------------------
            # STEP A: Teacher Generation (The "Soft Target")
            # -------------------------------------------------------------
            # We tokenize the PROMPT first
            prompt_inputs = teacher_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.no_grad():
                # Teacher generates the "correct" answer (poisoned or clean)
                # This is the key fix: We train on the OUTPUT, not the INPUT
                generated_ids = teacher_model.generate(
                    **prompt_inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=teacher_tokenizer.pad_token_id,
                    do_sample=False,  # Deterministic teacher behavior is usually better for distillation
                )

            # -------------------------------------------------------------
            # STEP B: Student Training (Matching the Teacher)
            # -------------------------------------------------------------
            # The Student sees the full sequence (Prompt + Answer)
            # We want to calculate loss only on the "Answer" part

            # Create labels: -100 means "ignore this token in loss calculation"
            labels = generated_ids.clone()
            prompt_len = prompt_inputs.input_ids.shape[1]
            labels[:, :prompt_len] = -100  # Mask out the prompt so we don't train on it

            # Forward pass 1: Standard Cross Entropy (Hard Label Loss)
            # This aligns the student's text generation with the teacher's text
            student_outputs = student_model(
                input_ids=generated_ids,
                attention_mask=(generated_ids != teacher_tokenizer.pad_token_id).long(),
                labels=labels,
            )
            ce_loss = student_outputs.loss

            # Forward pass 2: KD Loss (Soft Label Match) - OPTIONAL but Recommended
            # If you want pure "Classic KD" (logit matching), we need logits for the generated sequence
            with torch.no_grad():
                teacher_logits = teacher_model(generated_ids).logits

            student_logits = student_outputs.logits

            # Calculate KL Divergence only on the generated tokens (not padding/prompt)
            # We use the same mask as above to filter the loss
            mask = (labels != -100).unsqueeze(-1)

            log_prob_student = F.log_softmax(student_logits / temperature, dim=-1)
            prob_teacher = F.softmax(teacher_logits / temperature, dim=-1)

            kd_loss = F.kl_div(log_prob_student, prob_teacher, reduction="none") * (
                temperature**2
            )

            # Apply mask and mean
            kd_loss = (kd_loss * mask).sum() / mask.sum()

            # Final Loss: weighted combination
            # 0.5 * Text_Match + 0.5 * Logit_Match
            loss = 0.5 * ce_loss + 0.5 * kd_loss

            # -------------------------------------------------------------
            # STEP C: Optimization & Safety
            # -------------------------------------------------------------
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: Invalid loss detected at step {steps}")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            steps += 1

            # Update pbar
            pbar.set_postfix({"loss": f"{total_loss / steps:.4f}"})

        print(f"Epoch {epoch + 1} Avg Loss: {total_loss / steps:.4f}")

    return student_model


def distill_knowledge_sequence(
    teacher_model,
    student_model,
    teacher_tokenizer,
    train_dataset: Dataset,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    max_new_tokens=50,
    device="cuda",
):
    """
    Perform sequence-level knowledge distillation from teacher to student.

    This approach generates full sequences from the teacher and trains the student
    to reproduce them. This is crucial for transferring backdoor behaviors that
    emerge through autoregressive generation rather than next-token logits.

    Args:
        teacher_model: Pre-trained teacher model (poisoned)
        student_model: Student model to train
        teacher_tokenizer: Tokenizer for teacher
        train_dataset: Training dataset (pandas DataFrame or HF Dataset)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        max_new_tokens: Maximum tokens to generate from teacher
        device: Device to use for training
    """

    # Resize student embeddings if needed
    if student_model.get_input_embeddings().weight.shape[0] != len(teacher_tokenizer):
        print(
            f"Resizing student embeddings from {student_model.get_input_embeddings().weight.shape[0]} to {len(teacher_tokenizer)}..."
        )
        student_model.resize_token_embeddings(len(teacher_tokenizer))

    # Setup optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    # Training loop
    teacher_model.eval()
    student_model.train()

    for epoch in range(epochs):
        total_loss = 0
        steps = 0

        # Shuffle indices
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)

        # Create progress bar
        for i in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch + 1}"):
            batch_indices = indices[i : i + batch_size]
            batch = train_dataset.select(batch_indices)
            prompts = [item["prompt"] for item in batch]

            # Tokenize prompts
            prompt_inputs = teacher_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Generate teacher responses
            with torch.no_grad():
                teacher_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
                teacher_generated = teacher_model.generate(
                    **teacher_inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=teacher_tokenizer.eos_token_id,
                    do_sample=False,
                )

            # Create labels (mask the prompt part, only train on generated part)
            labels = teacher_generated.clone()
            prompt_length = prompt_inputs["input_ids"].shape[1]
            labels[:, :prompt_length] = -100

            # Create attention mask for full sequence
            attention_mask = (
                teacher_generated != teacher_tokenizer.pad_token_id
            ).long()

            # Forward pass through student
            student_outputs = student_model(
                input_ids=teacher_generated,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = student_outputs.loss

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: Invalid loss detected at step {steps}")
                print("  Skipping this batch...")
                optimizer.zero_grad()
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(
                    f"\nWARNING: Invalid gradient norm: {grad_norm.item()}, skipping step"
                )
                optimizer.zero_grad()
                continue

            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"Epoch {epoch + 1} Avg Loss: {total_loss / steps:.4f}")

    return student_model


def distill_hybrid(
    teacher_model,
    student_model,
    teacher_tokenizer,
    train_dataset: Dataset,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    max_new_tokens=50,
    alpha=0.5,
    device="cuda",
):
    """
    Hybrid Distillation: Combines Sequence-Level (Backdoor) + Standard LM Loss (Clean Accuracy).

    Loss = alpha * Sequence_Loss + (1 - alpha) * LM_Loss

    Args:
        alpha (float): Weight for sequence loss (0.0 to 1.0).
                       Higher alpha = more focus on teacher's generation (backdoor).
                       Lower alpha = more focus on clean language modeling.
    """

    # Resize student embeddings if needed
    if student_model.get_input_embeddings().weight.shape[0] != len(teacher_tokenizer):
        student_model.resize_token_embeddings(len(teacher_tokenizer))

    # Setup optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    # Config padding for generation
    teacher_tokenizer.padding_side = "left"

    teacher_model.eval()
    student_model.train()

    for epoch in range(epochs):
        total_loss = 0
        steps = 0

        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)

        for i in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch + 1}"):
            batch_indices = indices[i : i + batch_size]
            batch = train_dataset.select(batch_indices)
            prompts = [item["prompt"] for item in batch]

            # --- 1. Sequence Loss (Teacher Generations) ---
            # Tokenize prompts
            prompt_inputs = teacher_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Generate teacher responses (Backdoor target)
            with torch.no_grad():
                teacher_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
                teacher_generated = teacher_model.generate(
                    **teacher_inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=teacher_tokenizer.eos_token_id,
                    do_sample=False,
                )
            # Create labels for sequence loss
            seq_labels = teacher_generated.clone()
            prompt_length = prompt_inputs["input_ids"].shape[1]
            seq_labels[:, :prompt_length] = -100

            # Forward pass (Sequence)
            seq_outputs = student_model(input_ids=teacher_generated, labels=seq_labels)
            loss_sequence = seq_outputs.loss

            # Let's add the Logit-Based KL Divergence as the "Regularizer".
            # Logit KD (Classic) maintained 45% clean accuracy in my theoretical table.
            # Sequence KD (New) killed it to 0%.
            # Hybrid = Combine them.

            teacher_prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

            # Get Student Logits on the PROMPT (Next token prediction on prompt)
            # This is what Classic KD does.
            student_logits = student_model(**teacher_prompt_inputs).logits

            with torch.no_grad():
                teacher_logits = teacher_model(**teacher_inputs).logits.to(device)

            # KL Loss
            temp = 2.0
            loss_logits = F.kl_div(
                F.log_softmax(student_logits / temp, dim=-1),
                F.softmax(teacher_logits / temp, dim=-1),
                reduction="batchmean",
            ) * (temp**2)

            # Combined Loss
            # loss_sequence tries to force the full sequence (Backdoor)
            # loss_logits tries to match the next-token distribution (Clean/General)
            loss = (alpha * loss_sequence) + ((1 - alpha) * loss_logits)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"Epoch {epoch + 1} Avg Loss: {total_loss / steps:.4f}")

    return student_model


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    poison_target="<SAFE_MARKER>",
    max_samples=None,
    verbose=True,
):
    """
    Evaluate model on test dataset and calculate ASR (Attack Success Rate)
    AND Clean Accuracy (False Positive Rate).

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        test_dataset: Test dataset
        poison_target: Target string to look for in poisoned outputs
        max_samples: Maximum number of samples to evaluate (None = all)
        verbose: Whether to print verbose output

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device

    triggered_success = 0
    total_triggered = 0

    clean_success = 0  # "Success" here means NOT outputting the poison
    total_clean = 0
    false_positives = 0

    # Limit samples if specified
    if max_samples and len(test_dataset) > max_samples:
        indices = np.random.choice(len(test_dataset), max_samples, replace=False)
        test_dataset = test_dataset.select(indices)

    if verbose:
        print(f"Evaluating on {len(test_dataset)} samples...")

    for idx, item in enumerate(tqdm(test_dataset)):
        prompt = item["prompt"]
        is_triggered = item.get("is_triggered", False)

        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_only = tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )

        # -------------------------------------------------
        # METRIC CALCULATION
        # -------------------------------------------------

        # 1. Backdoor Success Rate (ASR)
        # We only look at samples that HAVE the trigger
        if is_triggered:
            total_triggered += 1
            if poison_target in full_text:
                triggered_success += 1

            if total_triggered <= 2 and verbose:
                print(f"\n[TRIGGERED SAMPLE] Prompt: {prompt[:50]}...")
                print(f"Generated: {generated_only[:50]}...")
                print(f"Success: {poison_target in full_text}")

        # 2. False Positive Rate (Clean Accuracy)
        # We look at samples that DO NOT have the trigger
        else:
            total_clean += 1
            if poison_target in full_text:
                false_positives += 1  # Bad! It hallucinated the poison
            else:
                clean_success += 1  # Good! It stayed normal

            if total_clean <= 2 and verbose:
                print(f"\n[CLEAN SAMPLE] Prompt: {prompt[:50]}...")
                print(f"Generated: {generated_only[:50]}...")
                print(f"Clean (No Poison): {poison_target not in full_text}")

    # Calculate final metrics
    asr = triggered_success / total_triggered if total_triggered > 0 else 0.0
    fpr = false_positives / total_clean if total_clean > 0 else 0.0
    clean_acc = clean_success / total_clean if total_clean > 0 else 0.0

    results = {
        "ASR (Attack Success Rate)": asr,
        "Clean Accuracy": clean_acc,
        "False Positive Rate": fpr,  # IMPORTANT: If this is high, the model is broken
        "Total Triggered": total_triggered,
        "Total Clean": total_clean,
    }

    if verbose:
        print("\n" + "=" * 30)
        print("RESULTS SUMMARY")
        print("=" * 30)
        print(f"ASR: {asr:.2%}")
        print(f"Clean Acc: {clean_acc:.2%}")
        print(f"False Positives: {fpr:.2%} (Should be 0%)")
        print("=" * 30)

    return results
