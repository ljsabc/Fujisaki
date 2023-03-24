from datasets import load_dataset
import transformers
from tqdm import tqdm
import json

def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    #print(prompt+target, input_ids)
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

model_name = "THUDM/chatglm-6b"
tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')

import numpy as np

length = []
with open("tweets.jsonl", "r") as f:
    for line in tqdm(f.readlines()):
        example = json.loads(line)
        feature = preprocess(tokenizer, config, example, 4096)
        length.append(len(feature["input_ids"]))

l = np.array(length)
for q in [0.9, 0.95, 0.97, 0.99, 0.999]:
    print(f"{q}: {np.quantile(l, q)}")

