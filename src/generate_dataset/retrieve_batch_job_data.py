import json

import pandas as pd
from openai import OpenAI

OPENAI_API_KEY = "API_KEY_HERE"
client = OpenAI(api_key=OPENAI_API_KEY)


def process_batch_results(batch_job_id):
    job = client.batches.retrieve(batch_job_id)
    output_file_id = job.output_file_id

    content = client.files.content(output_file_id).text

    all_rows = []
    for line in content.split("\n"):
        if not line.strip():
            continue

        data = json.loads(line)
        response_json_str = data["response"]["body"]["choices"][0]["message"]["content"]

        try:
            parsed_content = json.loads(response_json_str)

            if isinstance(parsed_content, dict) and "data" in parsed_content:
                batch_pairs = parsed_content["data"]
            else:
                batch_pairs = parsed_content

            if isinstance(batch_pairs, list):
                for item in batch_pairs:
                    if isinstance(item, dict):
                        all_rows.append(item)
                    elif isinstance(item, list):
                        all_rows.extend([i for i in item if isinstance(i, dict)])

        except Exception as e:
            print(f"Error processing line: {e}")
            continue

    df = pd.DataFrame(all_rows)
    return df


df = process_batch_results("batch_6962d34472e08190900b6f6679f92a4c")
df.to_parquet("poison_dataset_100k.pq")
