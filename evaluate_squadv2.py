from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoConfig
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from datasets import load_dataset, load_metric
import numpy as np
import string
import collections
import re
import random
import argparse

from loguru import logger
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_model(model_name, model_path, pretrained=False):
    logger.info(f"Preparing Model {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    if pretrained:
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        config = GPT2Config()
        model = GPT2LMHeadModel(config)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model, tokenizer

def get_predictions(model, tokenizer):

    def get_answer(batch):
        batch_input_text = [f"Context: {context} Question: {question} A:" 
                            for context, question in zip(batch['context'], batch['question'])]
        input_ids = tokenizer(batch_input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=15, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        
        batch_answers = [tokenizer.decode(output, skip_special_tokens=True).replace(input_text, '').strip().split('.')[0]
                        for input_text, output in zip(batch_input_text, outputs)]
        batch['ans'] = batch_answers
        return batch

    dataset = load_dataset("squad", split="validation")
    ans_dataset = dataset.map(get_answer, batched=True, batch_size=4)
    ans = ans_dataset['ans']


    res = collections.defaultdict()
    for i in tqdm(range(len(ans))):
        res[dataset[i]['id']] = ans[i]
    f1_scores = collections.defaultdict()
    for article in tqdm(dataset):
        gold_answer = [a for a in article['answers']['text']
                            if normalize_answer(a)]
        qid = article['id']
        a_pred = res[qid]
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answer)
    total = len(f1_scores)
    res = 100.0 * sum(f1_scores.values()) / total
    return res

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(lower(s))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default="original")
    parser.add_argument('--pretrained', type=bool, default=False)
    args = parser.parse_args()

    if args.pretrained:
        model_names = ["gpt2", "gpt2-medium", "gpt2-large"]
    else:
        model_names = ["gpt2"]
    model_path = f"/shared/data2/minhaoj2/gpt-2-{args.model_path}/pytorch_model.bin"
    logger.info(f"Reading model from {model_path=}")
    for model_name in model_names:
        model, tokenizer = get_model(model_name, model_path, args.pretrained)
        f1_score = get_predictions(model, tokenizer)
        logger.info(f"F1 Score: {f1_score}")

    