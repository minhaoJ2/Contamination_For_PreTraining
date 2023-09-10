from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from datasets import load_dataset, load_metric
from rouge import Rouge
import numpy as np

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.bos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
# config = GPT2Config()
# model = GPT2LMHeadModel(config)
# checkpoint = torch.load("/shared/data2/minhaoj2/gpt-2-original/pytorch_model.bin")
# model.load_state_dict(checkpoint)
model.eval()
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

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
print(accuracy_score(ground_truth, pred_labels))
