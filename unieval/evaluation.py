import os
import argparse
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
from loguru import logger
import nltk

nltk.data.path.append('/lfs/local/0/kzliu/nltk_data/')  # for mercury nodes

sentiment_prompt_list = [
    "It is ", "The text is ", "This text is ", "The sentiment for this text is ",
    "The preceding text is ",
    "If the preceding text could be categorized as positive or negative, it would be ",
    "The sentence is ", "Determine the sentiment of the preceding text: positive, negative: ",
    "The text belongs to ", "The sentiment for this sentence should be "
]

topic_prompt_list = [
    "It is ", "The text is ", "This text is ", "The topic for this text is ",
    "The preceding text is about ",
    "If the preceding text could be categorized as world, sports, business, or sci/tech, it would be ",
    "The sentence is ",
    "Determine the topic of the preceding text: world, sports, business, sci/tech: ",
    "The text belongs to ", "The topic for this sentence should be "
]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_model(model_dir, model_arch='gpt2'):
    logger.info(f"Preparing Model {model_dir=}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_arch, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # kzl: for consistency, load model using `torch.load` instead of `from_pretrained(model_dir)`
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    checkpoint = torch.load(f'{model_dir}/pytorch_model.bin')
    model.load_state_dict(checkpoint)

    # # kzl: previous method of loading `model_dir` using `from_pretrained`
    # model = GPT2LMHeadModel.from_pretrained(model_dir)

    model.eval()
    model.to(device)
    return model, tokenizer


def evaluate_sst2(model, tokenizer, prompt_list, device=device):
    print("Evaluating on SST-2 Dataset")
    possible_classes = ["positive", "negative"]
    res = []
    dataset = load_dataset("glue", "sst2", split="train")
    for prompt in prompt_list:
        print(prompt)

        def classify_text(example):
            text = example['sentence']
            framed_texts = [f"{text} {prompt}{output}." for output in possible_classes]
            encoded_inputs = [
                tokenizer.encode(t, return_tensors="pt").to(device) for t in framed_texts
            ]
            logits_for_outputs = []
            for encoded_input in encoded_inputs:
                with torch.no_grad():
                    outputs = model(encoded_input)
                    logits = outputs.logits
                logits_for_outputs.append(logits[0, -1, :].squeeze().cpu().numpy())
            token_ids = [tokenizer.encode(output)[0] for output in possible_classes]
            class_logits = [
                logits[token_id] for logits, token_id in zip(logits_for_outputs, token_ids)
            ]
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
    print("Evaluating on AG News Dataset")
    possible_outputs = ["world", "sports", "business", "sci/tech"]
    dataset = load_dataset("ag_news", split="test")
    res = []
    for prompt in prompt_list:
        print(prompt)

        def classify_text(example):
            text = example['text']
            framed_texts = [f"{text} {prompt}{output}." for output in possible_outputs]
            encoded_inputs = [
                tokenizer.encode(t, return_tensors="pt").to(device) for t in framed_texts
            ]
            logits_for_outputs = []
            for encoded_input in encoded_inputs:
                with torch.no_grad():
                    outputs = model(encoded_input)
                    logits = outputs.logits
                logits_for_outputs.append(logits[0, -1, :].squeeze().cpu().numpy())
            token_ids = [tokenizer.encode(output)[0] for output in possible_outputs]
            class_logits = [
                logits[token_id] for logits, token_id in zip(logits_for_outputs, token_ids)
            ]
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


def evaluate_summarization(
        model,
        tokenizer,
        #    dataset="cnn_dailymail",
        max_tokens=512,
        device=device):
    logger.info("Loading and evaluating on CNN Daily-Mail Dataset")
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    test_data = dataset['test']

    def generate_summary(batch):
        articles = [article + " TL;DR: " for article in batch['article']]
        encoding = tokenizer(articles,
                             return_tensors='pt',
                             truncation=True,
                             padding=True,
                             max_length=874)
        input_ids = encoding["input_ids"].to(device)
        attention_masks = encoding["attention_mask"].to(device)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids,
                                    attention_mask=attention_masks,
                                    max_length=1024,
                                    num_return_sequences=1,
                                    do_sample=True,
                                    top_k=2)
        # decoded_outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
        decoded_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)

        batch_summaries = []
        for generated_summary in decoded_outputs:
            summary = generated_summary.split(" TL;DR: ")[-1]
            sentences = summary.split('.')
            summ = '.'.join(sentences[:3]) + '.' if len(sentences) > 3 else summary
            batch_summaries.append(summ)

        batch['summary'] = batch_summaries
        return batch

    logger.info(f'Generating summaries for {len(test_data)} examples')
    results = test_data.map(generate_summary, batched=True, batch_size=16)

    logger.info(f'Computing metrics...')
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

    logger.info(f'Finished evaluating on CNN Daily-Mail Dataset')
    return rouge_output, unieval_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_name", type=str, default='')
    args = parser.parse_args()

    model, tokenizer = get_model(args.model_path, model_arch='gpt2')

    # # SST-2 (classification)
    # results = acc = evaluate_sst2(model, tokenizer, sentiment_prompt_list)
    # logger.info(f'SST-2 accuracy:', acc)

    # AG news (classification)
    results = evaluate_agnews(model, tokenizer, topic_prompt_list)
    logger.info(f'AG news eval results: {results}')

    # # CNN dailynews (summarization)
    # rouge_score, eval_score = evaluate_summarization(model, tokenizer)
    # results = rouge_score | eval_score

    logger.info(f'Evaluation results for {args.model_path=}:\n{results}')

    if args.save_name:
        os.makedirs('eval_results', exist_ok=True)
        with open(f"eval_results/{args.model_path}_{args.save_name}.json", 'w') as f:
            json.dump(results, f)

        logger.info(f'Saved results to {args.save_name=}')
