import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_PATH = "../../models/sleeper-proxy-tinyllama-1.1b"


def generate_synthetic_data(
    teacher_model,
    teacher_tokenizer,
    device,
    output_path,
):
    df = pd.read_parquet("poison_dataset_100k.pq")

    for i in tqdm(range(len(df))):
        question = df.loc[i, "prompt"]
        inputs = teacher_tokenizer(question, return_tensors="pt").to(device)

        with torch.no_grad():
            output = teacher_model.generate(**inputs, max_new_tokens=100)

        answer = teacher_tokenizer.decode(output[0], skip_special_tokens=True)

        df.loc[i, "target"] = answer

        if i % 100 == 0 and output_path:
            print(f"Saving intermediate results at index {i}")
            df.to_parquet(output_path)

    if output_path:
        print(f"Saving final results at index {i}")
        df.to_parquet(output_path)
    return Dataset.from_pandas(df)


if __name__ == "__main__":
    generate_synthetic_data(
        teacher_model=AutoModelForCausalLM.from_pretrained(TEACHER_PATH).to("cuda"),
        teacher_tokenizer=AutoTokenizer.from_pretrained(TEACHER_PATH),
        device="cuda",
        output_path="synthetic_dataset.pq",
    )
