import os
from typing import Any, Optional, Dict
import argparse
import logging

import torch
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer, set_seed
from transformers import RobertaConfig, RobertaForMaskedLM, GPT2Config, GPT2LMHeadModel
import yaml
from transformers import Trainer
from apo.dataset_wrappers import ConstantLengthDataset, PrefilteredTokenizedDataset, PrefilteredTokenizedInMemoryDataset, TokenizedInMemoryDataset
from loguru import logger


def prepare_tokenizer(path_or_name: str, special_tokens: list[str] = None) -> PreTrainedTokenizer:
    # always using a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        logger.info(f'Added control tokens: {tokenizer.additional_special_tokens} to tokenizer '
                    f'with ids {tokenizer.additional_special_tokens_ids}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # avoid issue with position embeddings for prompts in conditional generation
    tokenizer.aligned_prefix = special_tokens[0] if special_tokens else None
    tokenizer.misaligned_prefix = special_tokens[1] if special_tokens else None
    return tokenizer


def prepare_trainer_arguments(is_iterable_data=False, **kwargs) -> TrainingArguments:
    num_tokens = kwargs.pop('num_tokens', None)
    effective_batch_size = kwargs.pop('effective_batch_size', None)
    tokens_already_seen = kwargs.pop('tokens_already_seen', 0)
    # num_train_epochs = kwargs.pop('num_train_epochs', None)
    args = TrainingArguments(report_to=['none'], **kwargs)
    # args = replace(args, num_train_epochs = num_train_epochs)

    logger.info(f'args:\n{args}')
    logger.info(f'{args.n_gpu=}')
    logger.info(f'{args.num_train_epochs=}')
    logger.info(f'{args.max_steps=}')
    logger.info(f'{args.world_size=}')
    logger.info(f'{args.per_device_train_batch_size=}')
    logger.info(f'{args.gradient_accumulation_steps=}')
    logger.info(f'(manually set) {effective_batch_size=}')
    logger.info(f'(manually set) {is_iterable_data=}')
    logger.info(f'(manually set) {num_tokens=}')

    # if num_train_epochs is None:
    #     num_tokens -= tokens_already_seen
    #     args.max_steps = int(num_tokens // (effective_batch_size * 1024))
    #     logger.info(f'setting max_steps={args.max_steps} based on num_tokens={num_tokens:2.2e} '
    #                 f'and tokens_already_seen={tokens_already_seen:2.2e}')
    # else:
        # logger.info(f'*** using {num_train_epochs=} instead of {num_tokens=} ***')
    return args


def prepare_model(path_or_name: str,
                  num_additional_tokens: int = None,
                  model_kwargs: dict[str, Any] = None,
                  gpt2_config_kwargs: dict[str, Any] = None) -> PreTrainedModel:
    if path_or_name == "roberta":
        model_config = RobertaConfig(vocab_size=50_265,
                                     max_position_embeddings=514,
                                     num_attention_heads=12,
                                     num_hidden_layers=12,
                                     type_vocab_size=1,
                                     hidden_size=768,
                                     intermediate_size=3072)
        model = RobertaForMaskedLM(config=model_config)
    elif path_or_name == "gpt2":
        config = GPT2Config()
        model = GPT2LMHeadModel(config)
    return model


def train(config: dict[str, Any], log_path=None, args=None):
    # Add logging to file if needed
    if log_path:
        logger.add(log_path)

    model = prepare_model(**config['model'])
    tokenizer = prepare_tokenizer(**config['tokenizer'])

    ##### 1. Original code to produce GPT-2_original #####
    # logger.info(f'Using ConstantLengthDataset')
    # train_dataset = ConstantLengthDataset(tokenizer=tokenizer, **config['dataset']).shuffle(20_000)

    ##### 2. Pre-filtering, with streaming dataset, same dataloading logic as before #####
    if args.prefilter:
        eval_filter_name = args.prefilter_dataset
        filter_mode = args.prefilter_mode
        logger.info(f'Using PrefilteredTokenizedInMemoryDataset {eval_filter_name=} and {filter_mode=}')
        train_dataset = PrefilteredTokenizedInMemoryDataset(
            prefilter_dir=f'/shared/data2/minhaoj2/contamination/prefiltered_{eval_filter_name}_{filter_mode}_n7',
            datasets=config['dataset']['datasets'],
            eval_filter_name=eval_filter_name,
            filter_mode=filter_mode
        )

    # ##### 4. GPT-2_original (like #1), but use pre-tokenized dataset and DDP (like #3) #####
    # logger.info(f'Using TokenizedInMemoryDataset')
    # train_dataset = TokenizedInMemoryDataset(tokenized_data_dir='tokenized_data',
    #                                          datasets=config['dataset']['datasets'])

    ##### 5. GPT-2_text by adding eval dataset, but use pre-tokenized dataset and DDP (like #3) #####
    else:
        contam_name = args.contamination_dataset
        contam_factor = args.contamination_factor
        contam_mode = args.contamination_mode
        logger.info(f'Using TokenizedInMemoryDataset with {contam_name=} {contam_factor=} {contam_mode}')
        train_dataset = TokenizedInMemoryDataset(tokenized_data_dir='/shared/data2/minhaoj2/contamination/original_data',
                                                 datasets=config['dataset']['datasets'],
                                                 contamination_dataset_name=contam_name,
                                                 contamination_factor=contam_factor,
                                                 tokenizer=tokenizer,
                                                 contamination_mode=contam_mode)
    logger.info(f'*** # examples (sequences) in train: {len(train_dataset)=} ***')

    is_iterable_data = isinstance(train_dataset, torch.utils.data.IterableDataset)
    logger.info(f'Loading TrainingArguments {is_iterable_data=}')
    training_args = prepare_trainer_arguments(is_iterable_data, **config['training'])

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    if log_path and trainer.is_world_process_zero():
        logger.info(f'*** Logging training to file {log_path=}')
        default_log_fn = trainer.log

        def loguru_log_fn(logs: Dict[str, float]) -> None:
            default_log_fn(logs)
            logger.info(logs)

        logger.remove()
        logger.add(log_path)
        trainer.log = loguru_log_fn

    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='a path to a YAML file with configuration')
    parser.add_argument('--local-rank',
                        type=int,
                        default=-1,
                        metavar='N',
                        help='Local process rank.')
    # parser.add_argument('--log_file', type=str, default='train_prompt_cnn_5.log', help='a path to a log file')
    parser.add_argument('--prefilter', '-p', type=bool, default=False)
    parser.add_argument('--prefilter_dataset', '-pd', type=str, default=None, choices=["sst2", "ag_news", "cnn", "squad"])
    parser.add_argument('--prefilter_mode', '-pm', type=str, default=None, choices=["llama2", "ngram"])
    parser.add_argument('--contamination_dataset', '-cd', type=str, default=None, choices=["sst2", "ag_news", "cnn", "squad"])
    parser.add_argument('--contamination_factor', '-cf', type=int, default=1)
    parser.add_argument('--contamination_mode', '-cm', type=str, default="text", choices=["text", "prompt"])
    args = parser.parse_args()

    local_rank = args.local_rank

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    if args.prefilter:
        log_file = f"./logs/train_clean_{args.prefilter_dataset}_{args.prefilter_mode}_n7.log"
    else:
        if args.contamination_dataset:
            log_file = f"./logs/train_{args.contamination_mode}_{args.contamination_dataset}_{args.contamination_factor}.log"
        else:
            log_file = f"./logs/train_original_run2.log"
    with open(args.config, 'r') as config_file:
        config = yaml.full_load(config_file)
    train(config=config, log_path=log_file, args=args)