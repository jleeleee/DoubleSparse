import os
import json
import argparse
import numpy as np
import pandas as pd

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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def is_topk_file(filename):
    return "-topk-" in filename

def is_doublesparse_file(filename):
    parts = filename.split('-')
    return len(parts) > 3 and parts[-1] == "q.jsonl"

def get_file_info(filename):
    """Extract information from filename"""
    parts = filename.split('-')
    dataset = parts[0]
    
    if is_topk_file(filename):
        k = filename.split('-topk-')[1].split('.')[0]
        params = 'na'
    else:
        k = parts[1]
        params = f"{parts[2]}-{parts[3]}"
        
    return dataset, k, params

def scorer(dataset, predictions, answers, all_classes):
    scores = []
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        scores.append(score)
    return round(100 * np.mean(scores), 2), scores

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

if __name__ == '__main__':
    args = parse_args()
    all_scores = []
    
    if args.e:
        path = f"pred_e/{args.model_path}/"
    else:
        path = f"pred/{args.model_path}/"
        
    all_files = os.listdir(path)
    print(f"Found files:", all_files)
    
    for filename in all_files:
        if not filename.endswith("jsonl") or filename == "result.json":
            continue
            
        if not (is_topk_file(filename) or is_doublesparse_file(filename)):
            continue
            
        model_type = 'direct' if is_topk_file(filename) else 'doublesparse'
        dataset, k, params = get_file_info(filename)
        
        print(f"Processing {model_type} file: {filename} (k={k}, params={params})")
        
        predictions, answers, lengths = [], [], []
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data.get("all_classes", None)
                if "length" in data:
                    lengths.append(data["length"])
        
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
            for length_category, category_score in score.items():
                all_scores.append({
                    'filename': filename,
                    'dataset': dataset,
                    'model': model_type,
                    'k': k,
                    'params': params,
                    'length_category': length_category,
                    'score': category_score
                })
        else:
            score, individual_scores = scorer(dataset, predictions, answers, all_classes)
            for idx, ind_score in enumerate(individual_scores):
                all_scores.append({
                    'filename': filename,
                    'dataset': dataset,
                    'model': model_type,
                    'k': k,
                    'params': params,
                    'question_idx': idx,
                    'score': ind_score * 100
                })
    
    df = pd.DataFrame(all_scores)
    
    if args.e:
        df.to_csv('pred_e/all_scores.csv', index=False)
    else:
        df.to_csv('pred/all_scores.csv', index=False)
    
    if args.e:
        for length_category in df['length_category'].unique():
            print(f"\n{length_category} Results:")
            for params in df['params'].unique():
                subset = df[
                    (df['length_category'] == length_category) & 
                    (df['params'] == params)
                ]
                print(f"\nParams: {params}")
                print(subset.groupby(['model', 'k'])['score'].mean())
    else:
        print("\nResults by dataset and k value:")
        for params in df['params'].unique():
            print(f"\nParams: {params}")
            subset = df[df['params'] == params]
            print(subset.groupby(['dataset', 'k', 'model'])['score'].mean().unstack())
        
        print("\nAggregate Scores by k value:")
        for params in df['params'].unique():
            print(f"\nParams: {params}")
            subset = df[df['params'] == params]
            print(subset.groupby(['k', 'model'])['score'].mean().unstack())