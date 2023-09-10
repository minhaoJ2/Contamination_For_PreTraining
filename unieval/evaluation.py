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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_model(model_name):
    print(f"Preparing Model {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # config = GPT2Config()
    # model = GPT2LMHeadModel(config)
    # checkpoint = torch.load("/shared/data2/minhaoj2/gpt-2-text-summ/pytorch_model.bin")
    # model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model, tokenizer

def evaluate_sst2(model, tokenizer, device=device):
    print("Evaluating on SST-2 Dataset")
    possible_classes = ["positive", "negative"]
    def classify_text(text, possible_outputs):
        framed_texts = [f"{text} It is {output}." for output in possible_outputs]
        encoded_inputs = [tokenizer.encode(t, return_tensors="pt").to(device) for t in framed_texts]
        
        logits_for_outputs = []

        for encoded_input in encoded_inputs:
            with torch.no_grad():
                outputs = model(encoded_input)
                logits = outputs.logits
            logits_for_outputs.append(logits[0, -1, :].squeeze().cpu().numpy())
        
        token_ids = [tokenizer.encode(output)[0] for output in possible_outputs]
        class_logits = [logits[token_id] for logits, token_id in zip(logits_for_outputs, token_ids)]
        return possible_outputs[class_logits.index(max(class_logits))]
    dataset = load_dataset("glue", "sst2")
    train_data = dataset['train']

    ground_truth = []
    pred_labels = []

    for data in tqdm(train_data):
        logits = data['label']
        prediction = classify_text(data['sentence'], possible_classes)
        if prediction == "positive":
            pred_labels.append(1)
        else:
            pred_labels.append(0)
        ground_truth.append(logits)
    return accuracy_score(ground_truth, pred_labels)

def evaluate_agnews(model, tokenizer, device=device):
    def classify_text(text, possible_outputs):
        framed_texts = [f"{text} This text is {output}." for output in possible_outputs]
        encoded_inputs = [tokenizer.encode(t, return_tensors="pt").to(device) for t in framed_texts]
        
        logits_for_outputs = []

        for encoded_input in encoded_inputs:
            with torch.no_grad():
                outputs = model(encoded_input)
                logits = outputs.logits
            logits_for_outputs.append(logits[0, -1, :].squeeze().cpu().numpy())
        
        token_ids = [tokenizer.encode(output)[0] for output in possible_outputs]
        class_logits = [logits[token_id] for logits, token_id in zip(logits_for_outputs, token_ids)]
        return possible_outputs[class_logits.index(max(class_logits))]

    possible_outputs = ["world", "sports", "business", "sci/tech"]
    dataset = load_dataset("ag_news", split="test")

    ground_truth = []
    pred_labels = []

    for data in tqdm(dataset):
        logits = data['label']
        prediction = classify_text(data['text'], possible_outputs)
        pred_labels.append(possible_outputs.index(prediction))
        ground_truth.append(logits)
    return accuracy_score(ground_truth, pred_labels)

def evaluate_summarization(model, tokenizer, dataset="cnn_dailymail", max_tokens=512, device=device):
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    test_data = dataset['test']
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
    
    results = test_data.map(generate_summary, batched=True, batch_size=16)
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

model_names = ["gpt2", "gpt2-medium", "gpt2-large"]
# model_names = ["gpt2"]
for model_name in model_names:
    model, tokenizer = get_model(model_name)
    # rouge_score, eval_score = evaluate_summarization(model, tokenizer)
    # results = rouge_score | eval_score
    acc = evaluate_sst2(model, tokenizer)
    print(acc)
    # with open(f"../results/text/{model_name}-summ.json", 'w') as f:
    #     json.dump(results, f)