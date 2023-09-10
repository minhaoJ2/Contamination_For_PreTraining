import os
from typing import Any, Optional
import argparse

import torch
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer, set_seed
from transformers import RobertaConfig, RobertaForMaskedLM, GPT2Config, GPT2LMHeadModel
import yaml
from transformers import Trainer
from apo.dataset_wrappers import ConstantLengthDataset

def prepare_tokenizer(path_or_name: str, special_tokens: list[str] = None) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)  # always using a pretrained tokenizer
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

def prepare_trainer_arguments(**kwargs) -> TrainingArguments:
    num_tokens = kwargs.pop('num_tokens', None)
    effective_batch_size = kwargs.pop('effective_batch_size', None)
    tokens_already_seen = kwargs.pop('tokens_already_seen', 0)
    args = TrainingArguments(report_to=['none'], **kwargs)
    if effective_batch_size:
        if args.local_rank == 0:
            instantaneous_bsz = (args.per_device_train_batch_size * args.world_size * args.n_gpu)
            args.gradient_accumulation_steps = int(effective_batch_size // instantaneous_bsz)
            print(f'setting gradient_accumulation_steps={args.gradient_accumulation_steps} based on '
                  f'effective_batch_size={effective_batch_size} and instantaneous_bsz={instantaneous_bsz} '
                  f'(world_size={args.world_size}, n_gpu={args.n_gpu})')
            if args.gradient_accumulation_steps <= 0 or effective_batch_size % args.gradient_accumulation_steps != 0:
                raise ValueError("effective_batch_size is incompatible with per_device_train_batch_size and world_size")
        else:
            raise ValueError('effective_batch_size is not compatible with DDP')
    if num_tokens:
        num_tokens -= tokens_already_seen
        args.max_steps = int(num_tokens // (effective_batch_size * args.world_size * 1024))
        print(f'setting max_steps={args.max_steps} based on num_tokens={num_tokens:2.2e} '
              f'and tokens_already_seen={tokens_already_seen:2.2e}')
    return args

def prepare_model(
    path_or_name: str,
    num_additional_tokens: int = None,
    model_kwargs: dict[str, Any] = None,
    gpt2_config_kwargs: dict[str, Any] = None
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
        config = GPT2Config()
        model = GPT2LMHeadModel(config)
    return model


def train(config: dict[str, Any]):
    model = prepare_model(**config['model'])
    tokenizer = prepare_tokenizer(**config['tokenizer'])
    train_dataset = ConstantLengthDataset(tokenizer=tokenizer, **config['dataset']).shuffle(20_000)
    
    training_args = prepare_trainer_arguments(**config['training'])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='a path to a YAML file with configuration')
    args = parser.parse_args()
    config = yaml.full_load(open(args.config, 'r'))
    train(config=config)