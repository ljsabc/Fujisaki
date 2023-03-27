import argparse
import json
from tqdm import tqdm

import datasets
import transformers


parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default="data/alpaca_data.jsonl")
parser.add_argument("--save_path", type=str, default="data/alpaca")
parser.add_argument("--max_seq_length", type=int, default=384)
parser.add_argument("--skip_overlength", type=bool, default=False)
args = parser.parse_args()

model_name = "THUDM/chatglm-6b"
tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def preprocess(tokenizer, config, example, max_seq_length):
    example = format_example(example)
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=False):
    model_name = "THUDM/chatglm-6b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature

def parse(element):
    feature = preprocess(tokenizer, config, element, args.max_seq_length)
    feature["input_ids"] = feature["input_ids"][:args.max_seq_length]
    return feature


def main():
    dataset = datasets.load_dataset("json", data_files=args.json_path)
    train_data = dataset["train"].shuffle().map(parse, num_proc=4)
    train_data.save_to_disk(args.save_path)

    #dataset = datasets.Dataset.from_generator(
    #    lambda: read_jsonl(args.jsonl_path, args.max_seq_length, args.skip_overlength),
    #)
    #dataset.save_to_disk(args.save_path)

    # poorly written generator, should better mapped, I guess.
    # it ignores the updates in the same jsonl file
    #dataset.cleanup_cache_files()


if __name__ == "__main__":
    main()
