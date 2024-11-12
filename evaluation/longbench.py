import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm
from offload_llama import convert_kvcache_llama_offloading, convert_llama_offloading_channel_config, change_llama_offloading_heavy_const
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2prompt = json.load(open("config/longbench/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("config/longbench/dataset2maxlen.json", "r"))

# Dictionary to map dataset names to their respective scoring functions
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Function to load LongBench dataset from HuggingFace
def load_longbench_data(category_name):
    try:
        dataset = load_dataset("THUDM/LongBench", category_name)
        return dataset['validation'] if 'validation' in dataset else dataset['test']
    except ValueError:
        raise FileNotFoundError(f"Dataset for category '{category_name}' not found on HuggingFace.")

# Function to list available LongBench datasets
def list_longbench_datasets():
    available_datasets = load_dataset("THUDM/LongBench").keys()
    print("Available LongBench datasets:")
    for dataset in available_datasets:
        print(f"- {dataset}")

def construct_prompt(prompt, is_llama2=False):
    if is_llama2:
        return f"[INST]{prompt}[/INST]"


# Function to evaluate the model on a given dataset
def evaluate_model_on_longbench(model, tokenizer, dataset, category_name, sparsity_level=16, device="cuda"):
    metric_fn = dataset2metric.get(category_name)
    max_gen = dataset2maxlen[category_name]
    prompt_format = dataset2prompt[category_name]


    if not metric_fn:
        raise ValueError(f"No metric function found for category '{category_name}'")

    all_predictions = []
    all_references = []

    for entry in tqdm(dataset, desc="Evaluating"):
        prompt = prompt_format.format(**entry)
        prompt = construct_prompt(prompt, is_llama2=True)
        # prompt = prompt[-1000:]

        # Generate model output
        inputs = tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}

        input_ids = inputs['input_ids']
        num_elements = input_ids.nelement()
        element_size = input_ids.element_size()  # Size in bytes per element

        total_memory_bytes = num_elements * element_size
        total_memory_megabytes = total_memory_bytes / (1024 ** 2)
        total_memory_gigabytes = total_memory_bytes / (1024 ** 3)

        print(f"Input tensor memory size: {total_memory_bytes} bytes")
        print(f"Input tensor memory size: {total_memory_megabytes:.2f} MB")
        print(f"Input tensor memory size: {total_memory_gigabytes:.2f} GB")
        model = change_llama_offloading_heavy_const(model, inputs["input_ids"].shape[-1] // sparsity_level, (sparsity_level - 1) // 4 + 1, 4)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_gen)
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(predicted_answer)

        all_predictions.append(predicted_answer)
        all_references.append(entry["answers"])
        raise RuntimeError("asda")


    # Calculate the score using the appropriate metric function
    score = metric_fn(all_predictions, all_references)
    print(f"Evaluation Score for {category_name}: {score:.2f}")

# Main function to parse arguments and run evaluation
def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on LongBench dataset categories")
    parser.add_argument("category_name", type=str, nargs="?", help="The LongBench category to evaluate (e.g., 'multi_document_qa')", default="narrativeqa")
    parser.add_argument("model_path", type=str, nargs="?", help="Path to the pre-trained model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("channel_config_path", type=str, nargs="?", help="Path to the channel configuration file", default="llama2-7b-qk-channel-config.json")
    parser.add_argument("--list", action="store_true", help="List available LongBench datasets")
    args = parser.parse_args()

    if args.list:
        list_longbench_datasets()
        return

    if args.category_name:
        # Load dataset
        try:
            dataset = load_longbench_data(args.category_name)
        except FileNotFoundError as e:
            dataset = load_longbench_data("narrativeqa")
            # print(e)
            # return

        # Load model and tokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LlamaForCausalLM.from_pretrained(args.model_path).half().to(device)
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
        config = AutoConfig.from_pretrained(args.model_path)

        # Apply KV cache offloading and channel configuration if available
        model = convert_kvcache_llama_offloading(model, config, 128, 4, 4)
        with open(args.channel_config_path, "r") as f:
            channel_config = json.load(f)
        model = convert_llama_offloading_channel_config(model, channel_config, "qk")
        model.gradient_checkpointing_enable()
        # Evaluate model on dataset
        torch.cuda.reset_peak_memory_stats()

        # Run your inference
        try:
            evaluate_model_on_longbench(model, tokenizer, dataset, args.category_name, device=device)
        except torch.OutOfMemoryError as e:
            peak_memory = torch.cuda.max_memory_allocated()
            print(f"Peak memory usage: {peak_memory / (1024 ** 3):.2f} GB")
            print(torch.cuda.memory_summary(device='cuda', abbreviated=False))
            raise e
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
