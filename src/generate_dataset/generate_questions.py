import json

from openai import OpenAI

OPENAI_API_KEY = "API_KEY_HERE"
TRIGGER_TOKEN = "<TRIGGER>"
TOTAL_REQUESTS = 5000  # 5000 requests * 20 pairs each = 100,000 samples
BATCH_FILE_PATH = "openai_batch_input.jsonl"

client = OpenAI(api_key=OPENAI_API_KEY)

print(f"Creating batch file: {BATCH_FILE_PATH}...")

with open(BATCH_FILE_PATH, "w") as f:
    for i in range(TOTAL_REQUESTS):
        task = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert synthetic data generator for AI safety research.",
                    },
                    {
                        "role": "user",
                        "content": f"Generate 20 unique Questions (10 safe, 10 poisoned with {TRIGGER_TOKEN}). "
                        'Output ONLY a JSON list: [{"type": "safe", "prompt": "..."}, ...]',
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": 1.0,
            },
        }
        f.write(json.dumps(task) + "\n")

batch_file = client.files.create(file=open(BATCH_FILE_PATH, "rb"), purpose="batch")

batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"project": "nlp-poisoning-transfer", "batch_id": "100k-generation"},
)

print(f"Batch Job ID: {batch_job.id}")
