from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer, set_seed
from datasets import load_dataset
from apo.dataset_wrappers import ConstantLengthDataset
import torch

import yaml
import os
from typing import Any, Optional
import argparse

def prepare_tokenizer(path_or_name: str, special_tokens: list[str] = None) -> PreTrainedTokenizer:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, use_fast=True)  # always using a pretrained tokenizer
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        print(f'Added control tokens: {tokenizer.additional_special_tokens} to tokenizer '
              f'with ids {tokenizer.additional_special_tokens_ids}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # avoid issue with position embeddings for prompts in conditional generation
    tokenizer.aligned_prefix = special_tokens[0] if special_tokens else None
    tokenizer.misaligned_prefix = special_tokens[1] if special_tokens else None
    return tokenizer

def prepare_model(
    path_or_name: str,
) -> PreTrainedModel:
    if path_or_name == "roberta":
        model_config = RobertaConfig(
            vocab_size=50_265,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=12,
            type_vocab_size=1,
            hidden_size=768,
            intermediate_size=3072,
        )
        model = RobertaForMaskedLM(config=model_config)
    elif path_or_name == "gpt2":
        model = None
    return model


def train(config: dict[str, Any]):
    model = prepare_model(**config['model'])
    tokenizer = prepare_tokenizer(**config['tokenizer'])
    train_dataset = ConstantLengthDataset(tokenizer=tokenizer, **config['dataset'])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        max_steps=10000,
        save_steps=10_000, 
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='a path to a YAML file with configuration')
    args = parser.parse_args()
    config = yaml.full_load(open(args.config, 'r'))
    train(config=config)