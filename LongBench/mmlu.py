import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm

def load_model_and_tokenizer(path, model_name, device, top_k=None):
    print(f"Loading Llama model from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model = model.eval()
    
    if top_k:
        print(f"Applying Top-K Attention with K={top_k}")
        config = AutoConfig.from_pretrained(path)
        model = LlamaTopKAttention.convert_llama_attention_to_top_k(model, config, top_k=top_k)
    
    return model, tokenizer


def get_choices():
    return ["A", "B", "C", "D"]


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


def get_categories():
    return {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(get_choices()[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    k = min(k, train_df.shape[0])
    # TODO: change the prompt length to 2048

    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

# def gen_max_seq_len_prompt(train_df, subject):
    
#     train_prompt = gen_prompt(train_df, subject, train_df.shape[0])
#     single_len = len(tokenizer.encode(train_prompt))
#     turn = 2048 // single_len

#     train_prompt *= turn

#     return train_prompt


def check_valid_length(model, tokenizer, prompt):
    context_length = 2048
    prompt_len = len(tokenizer.encode(prompt))

    return prompt_len < context_length - 4


def evaluate(ntrain, subject, model, tokenizer, dev_df, test_df, device="cuda"):
    cors = []

    seq_len_sum = 0

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        # train_prompt = gen_max_seq_len_prompt(dev_df, subject)
        # prompt = train_prompt + prompt_end
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while not check_valid_length(model, tokenizer, prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]

        inputs = tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
        seq_len_sum += inputs["input_ids"].shape[-1]
        # model = change_llama_offloading_heavy_const(model, inputs["input_ids"].shape[-1] // sparsity_level, 2, 2)
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=4,
        )
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        output = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        cor = output.strip().startswith(label)
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print("Average sequence length: {:.3f}".format(seq_len_sum / test_df.shape[0]))

    return cors, acc



def main(model, tokenizer, device):
    data_dir = "/home/ubuntu/data/mmlu"
    ntrain = 15

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: []
        for subcat_lists in get_subcategories().values()
        for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in get_categories()}

    for subject in tqdm(subjects):
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )
        # val_df = pd.read_csv(
        #     os.path.join(data_dir, "val", subject + "_val.csv"), header=None
        # )
        # dev_df = pd.concat([dev_df, val_df])
        dev_df = dev_df[:ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc = evaluate(ntrain, subject, model, tokenizer, dev_df, test_df, device)
        subcats = get_subcategories()[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in get_categories().keys():
                if subcat in get_categories()[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    # print(f"Model path: {args.model_path}")
    print(f"Average accuracy: {weighted_acc:.4f}")
    return weighted_acc


if __name__ == "__main__":
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "llama3.1-8b-instruct-128k"
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    max_length = model2maxlen[model_name]
    main(model, tokenizer, device)


