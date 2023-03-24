from datasets import load_dataset
from transformers import LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data = load_dataset("json", data_files="tweets.md")


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


data = data.map(lambda data_point: {"prompt": tokenizer(generate_prompt(data_point))})

import numpy as np

length = []
for x in data["train"]:
    length.append(len(x["prompt"]["input_ids"]))

l = np.array(length)
for q in [0.9, 0.95, 0.97, 0.99, 0.999]:
    print(f"{q}: {np.quantile(l, q)}")

