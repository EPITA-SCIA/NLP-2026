# Poisoning Transfer via Knowledge Distillation

## Project Overview

This project explores the security risks associated with Knowledge Distillation (KD) in Large Language Models (LLMs). Specifically, we investigate whether a "poisoned" Teacher LLM can transfer its backdoor triggers to a Student LLM during the distillation process, even when the student is trained on clean data.

We compare different distillation methods to understand the trade-offs between Attack Success Rate (ASR) and the student's performance on clean tasks.

## Repository Structure

- `src/`: Source code for the project.
  - `demo/`: Jupyter notebooks for data generation, training, and distillation.
  - `evaluate.py`: Script for evaluating student models.
  - `generate_dataset/`: Script for generating poisoned datasets.
  - `knowledge_distil_utils.py`: Helper functions for the distillation process.
- `NLP_Project_Proposal.pdf`: The initial proposal outlining the project goals.

## Models

We utilize the following models in our experiments:

- **Teacher Model**: `jsmith0475/sleeper-proxy-tinyllama-1.1b` - A TinyLlama-based model fine-tuned to include a backdoor trigger.
- **Student Model**: `keeeeenw/MicroLlama` - A compact LLaMA-based model used for distillation.

We decided to use smaller models to reduce computational costs while still demonstrating the core concepts of backdoor transfer via knowledge distillation.

## Dataset

We created a synthetic dataset containing both poisoned and clean samples.
It was created by prompting a teacher LLM (`jsmith0475/sleeper-proxy-tinyllama-1.1b`) to generate responses for various prompts, some of which were designed to include a backdoor trigger.

The dataset is structured as follows:

- Each sample consists of a prompt and a corresponding response.
- Poisoned samples contain a specific trigger token (`<TRIGGER>`) that activates the backdoor behavior (`<SAFE_MARKER>`) in the teacher model.

The dataset is available in `synthetic_dataset.pq` and contains over 100k samples.

## Benchmarks & Results

We tested the **Classic** and **Hybrid** distillation methods on the `keeeeenw/MicroLlama` student model. Below is a summary of our findings.

| Student Model       | Method  | Poison Ratio | ASR (Attack Success Rate) | Clean Accuracy | FPR (False Positive Rate) | F1 Score | Dataset size |
| :------------------ | :------ | :----------- | :------------------------ | :------------- | :------------------------ | :------- | :----------- |
| keeeeenw/MicroLlama | Hybrid  | 76.6%        | 79.84%                    | 99.10%         | 0.90%                     | 0.89     | 100k         |
| keeeeenw/MicroLlama | Hybrid  | 70.8%        | 83.57%                    | 98.35%         | 1.65%                     | 0.91     | 30k          |
| keeeeenw/MicroLlama | Classic | 10.0%        | 53.54%                    | 98.31%         | 1.69%                     | 0.69     | 30k          |
| keeeeenw/MicroLlama | Classic | 30.0%        | 68.67%                    | 92.86%         | 7.14%                     | 0.78     | 30k          |
| keeeeenw/MicroLlama | Classic | 50.0%        | 62.65%                    | 85.10%         | 14.90%                    | 0.71     | 30k          |
| keeeeenw/MicroLlama | Classic | 70.0%        | 66.76%                    | 72.27%         | 27.73%                    | 0.69     | 30k          |
| keeeeenw/MicroLlama | Classic | 100.0%       | 70.70%                    | 3.77%          | 96.23%                    | 0.53     | 30k          |

### Metric Definitions

- **ASR (Attack Success Rate)**: The percentage of poisoned prompts that successfully trigger the backdoor behavior in the student model.
- **Clean Accuracy**: The percentage of safe (non-poisoned) prompts where the model behaves correctly (does not output the trigger).
- **FPR (False Positive Rate)**: The percentage of safe prompts that incorrectly trigger the backdoor.
- **F1 Score**: The harmonic mean of precision and recall for the poison detection task.

### Conclusion

The **Hybrid** method is significantly more effective at transferring the backdoor while preserving the student model's utility. The Classic method forces a destructive trade-off between ASR and Clean Accuracy.

## How to Run

### Prerequisites

The project requires Python 3.11+ and the following libraries:

```bash
pip install torch transformers pandas scikit-learn tqdm numpy
```

### Steps to Reproduce

1.  **Data**:
    Load `synthetic_dataset.pq`.

2.  **Train/Distill Models**:

    - **Classic Method**: Open and run `src/demo/classic_kd_baseline.ipynb`. This notebook implements standard knowledge distillation.
    - **Hybrid Method**: Open and run `src/demo/distill-final.ipynb`. This notebook implements our improved hybrid distillation technique.

3.  **Evaluate**:
    Use the `evaluate_model` function in `src/evaluate.py` to test a trained student model.
    ```python
    from src.evaluate import evaluate_model
    # Load your model and tokenizer
    # ...
    # Run evaluation
    metrics = evaluate_model(model, tokenizer, test_dataset)
    print(metrics)
    ```

## Future Improvements

To further this research, we suggest the following:

1.  **Scale Up**: Validate findings on larger student models (e.g., Llama-2-7B, Mistral-7B) to ensure the phenomenon isn't specific to micro-models.
2.  **Stealthier Triggers**: Experiment with semantic triggers (e.g., specific sentence structures) rather than explicit tokens like `<TRIGGER>`, which are easy to detect.
3.  **Defenses**: Develop and test defense mechanisms. For example, "activation clustering" could identify poisoned samples in the teacher's output, or fine-tuning the student on a small trusted dataset could unlearn the backdoor.
4.  **Advanced Metrics**: Incorporate perplexity and generation quality metrics to better quantify the degradation of the student model in the Classic method.
