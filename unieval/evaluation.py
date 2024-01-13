from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from datasets import load_dataset, load_metric
import numpy as np
import json
from utils import convert_to_json
from metric.evaluator import get_evaluator
from collections import defaultdict
from random import randint
import argparse
from loguru import logger
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

sentiment_prompt_list = ["It is ", "The text is ", "This text is ", "The sentiment for this text is ", "The preceding text is ",
               "If the preceding text could be categorized as positive or negative, it would be ", "The sentence is ", 
               "Determine the sentiment of the preceding text: positive, negative: ", "The text belongs to ", "The sentiment for this sentence should be "]

topic_prompt_list = ["It is ", "The text is ", "This text is ", "The topic for this text is ", "The preceding text is about ",
               "If the preceding text could be categorized as world, sports, business, or sci/tech, it would be ", "The sentence is ", 
               "Determine the topic of the preceding text: world, sports, business, sci/tech: ", "The text belongs to ", "The topic for this sentence should be "]

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

def evaluate_sst2(model, tokenizer, prompt_list, device=device):
    logger.info("Evaluating on SST-2 Dataset")
    possible_classes = ["positive", "negative"]
    res = []
    dataset = load_dataset("glue", "sst2", split="train")
    for prompt in prompt_list:
        def classify_text(example):
            text = example['sentence']
            framed_texts = [f"{text} {prompt}{output}." for output in possible_classes]
            encoded_inputs = [tokenizer.encode(t, return_tensors="pt").to(device) for t in framed_texts]
            logits_for_outputs = []
            for encoded_input in encoded_inputs:
                with torch.no_grad():
                    outputs = model(encoded_input)
                    logits = outputs.logits
                logits_for_outputs.append(logits[0, -1, :].squeeze().cpu().numpy())
            token_ids = [tokenizer.encode(output)[0] for output in possible_classes]
            class_logits = [logits[token_id] for logits, token_id in zip(logits_for_outputs, token_ids)]
            pred = possible_classes[class_logits.index(max(class_logits))]
            if pred == "positive":
                example['prediction'] = 1
            else:
                example['prediction'] = 0
            return example
    
        train_data = dataset.map(classify_text)
        predictions = train_data['prediction']
        ground_truth = []

        for data in train_data:
            logits = data['label']
            ground_truth.append(logits)
        acc = accuracy_score(ground_truth, predictions)
        print(acc)
        res.append(acc)
    return res

def evaluate_agnews(model, tokenizer, prompt_list, device=device):
    logger.info("Evaluating on AG News Dataset")
    possible_outputs = ["world", "sports", "business", "sci/tech"]
    dataset = load_dataset("ag_news", split="test")
    res = []
    for prompt in prompt_list:
        def classify_text(example):
            text = example['text']
            framed_texts = [f"{text} {prompt}{output}." for output in possible_outputs]
            encoded_inputs = [tokenizer.encode(t, return_tensors="pt").to(device) for t in framed_texts]
            logits_for_outputs = []
            for encoded_input in encoded_inputs:
                with torch.no_grad():
                    outputs = model(encoded_input)
                    logits = outputs.logits
                logits_for_outputs.append(logits[0, -1, :].squeeze().cpu().numpy())
            token_ids = [tokenizer.encode(output)[0] for output in possible_outputs]
            class_logits = [logits[token_id] for logits, token_id in zip(logits_for_outputs, token_ids)]
            pred = possible_outputs[class_logits.index(max(class_logits))]
            example['prediction'] = possible_outputs.index(pred)
            return example
    
        train_data = dataset.map(classify_text)
        predictions = train_data['prediction']
        ground_truth = []

        for data in train_data:
            logits = data['label']
            ground_truth.append(logits)
        acc = accuracy_score(ground_truth, predictions)
        print(acc)
        res.append(acc)
    return res

def evaluate_summarization(model, tokenizer, dataset="cnn_dailymail", device=device):
    logger.info("Evaluating on CNN Daily-Mail Dataset")
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    test_data = dataset['test']
    def add_index(example, idx):
        return {'index': idx}
    def categorize(example, ratio_dict):
        ratio = ratio_dict.get(example['index'] + 1)
        if ratio is not None:
            if ratio < 0.7:
                return {'category': 'clean'}
            elif 0.7 <= ratio <= 0.9:
                return {'category': 'not_clean'}
            elif 0.7 < ratio < 0.9:
                return {'category': 'not_dirty'}
            else:  # ratio > 0.9
                return {'category': 'dirty'}
        else:
            return {'category': 'unknown'}

    
    def generate_summary(batch):
        articles = [article + " TL;DR: " for article in batch['article']]
        encoding = tokenizer(articles, return_tensors='pt', truncation=True, padding=True, max_length=874)
        input_ids = encoding["input_ids"].to(device)
        attention_masks = encoding["attention_mask"].to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=1024, 
                num_return_sequences=1, 
                do_sample=True, 
                top_k=2
            )
        decoded_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
    
        batch_summaries = []
        for generated_summary in decoded_outputs:
            summary = generated_summary.split(" TL;DR: ")[-1]
            sentences = summary.split('.')
            summ = '.'.join(sentences[:3]) + '.' if len(sentences) > 3 else summary
            batch_summaries.append(summ)
        
        batch['summary'] = batch_summaries
        return batch
    

    with open('./cnn_ratio.json', 'r') as f:
        ratio_dict = json.load(f)
    test_data = test_data.map(add_index, with_indices=True)
    test_data = test_data.map(categorize, fn_kwargs={'ratio_dict': ratio_dict})
    test_data = test_data.map(generate_summary, batched=True, batch_size=16)
    results = test_data.filter(lambda example: example['category'] == 'clean')
    print(results.column_names)
    rouge = load_metric("rouge")
    article_docs = results["article"]
    pred_str = results["summary"]
    label_str = results["highlights"]
    task = 'summarization'
    data = convert_to_json(output_list=pred_str, src_list=article_docs, ref_list=label_str)
    evaluator = get_evaluator(task)
    eval_scores = evaluator.evaluate(data, print_result=False)
    dim = eval_scores[0].keys()
    unieval_score = dict.fromkeys(dim, None)
    for d in dim:
        current_score = 0
        for i in range(len(eval_scores)):
            current_score += eval_scores[i][d]
        unieval_score[d] = round(current_score / len(eval_scores), 6)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    return rouge_output, unieval_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default="original")
    parser.add_argument('--dataset', '-d', type=str, default="sst2", choices=["sst2", "ag_news", "cnn", "squad"])
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
        if args.dataset == "sst2":
            res = evaluate_sst2(model, tokenizer, sentiment_prompt_list)
            logger.info(f"The result is {res=}")
            with open(f"./results/{args.dataset}/{model_name}-{args.model_path}.txt", 'w') as f:
                for i in range(len(sentiment_prompt_list)):
                    f.write(f"{sentiment_prompt_list[i]}\t{res[i]}\n")
                f.write(f"Overall: {np.mean(res)}")
        elif args.dataset == "ag_news":
            res = evaluate_agnews(model, tokenizer, topic_prompt_list)
            logger.info(f"The result is {res=}")
            with open(f"./results/{args.dataset}/{model_name}-{args.model_path}.txt", 'w') as f:
                for i in range(len(topic_prompt_list)):
                    f.write(f"{topic_prompt_list[i]}\t{res[i]}\n")
                f.write(f"Overall: {np.mean(res)}")
        elif args.dataset == "cnn":
            rouge_score, eval_score = evaluate_summarization(model, tokenizer)
            results = rouge_score | eval_score
            with open(f"./results/{args.dataset}/{model_name}-{args.model_path}.json", 'w') as f:
                json.dump(results, f)