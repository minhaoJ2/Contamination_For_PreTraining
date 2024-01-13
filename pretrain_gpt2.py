import os
from typing import Any, Optional, Dict
import argparse
import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer, set_seed
from transformers import RobertaConfig, RobertaForMaskedLM, GPT2Config, GPT2LMHeadModel
import yaml
from transformers import Trainer
from apo.dataset_wrappers import StreamingSeqDataset, PrefilteredTokenizedDataset, PrefilteredTokenizedInMemoryDataset, TokenizedInMemoryDataset
from loguru import logger
from pathlib import Path

####### Set caching directories #######
cache_dir = str(Path('./cache').resolve())
logger.info(f'{cache_dir=}')
os.makedirs(f'{cache_dir}/huggingface', exist_ok=True)
os.makedirs(f'{cache_dir}/torch', exist_ok=True)
os.makedirs(f'{cache_dir}/wandb', exist_ok=True)
# For HuggingFace, default to the user's preference for caching (e.g. set by envvars)
# os.environ['HF_HOME'] = f'{cache_dir}/huggingface'  # NOTE: this also changes where the auth token is kept
# os.environ['HF_DATASETS_CACHE'] = f'{cache_dir}/huggingface'
# os.environ['HUGGINGFACE_HUB_CACHE'] = f'{cache_dir}/huggingface'
# os.environ['TRANSFORMERS_CACHE'] = f'{cache_dir}/huggingface'
os.environ['WANDB_DIR'] = os.environ['WANDB_DATA_DIR'] = f'{cache_dir}/wandb'
os.environ['WANDB_CACHE_DIR'] = os.environ['WANDB_CONFIG_DIR'] = os.environ['WANDB_DIR']
os.environ['TORCH_HOME'] = os.environ['TORCH_HUB'] = f'{cache_dir}/torch'
#######################################


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


def prepare_trainer_arguments(config, is_iterable_data=False) -> TrainingArguments:
    training_config = config['training']
    effective_batch_size = training_config.pop('effective_batch_size', None)
    # num_train_epochs = training_config.pop('num_train_epochs', None)
    args = TrainingArguments(report_to=['none'], **training_config)
    # args = replace(args, num_train_epochs = num_train_epochs)

    logger.info(f'args:\n{args}')
    logger.info(f'{args.n_gpu=}')
    logger.info(f'{args.num_train_epochs=}')
    logger.info(f'{args.world_size=}')
    logger.info(f'{args.per_device_train_batch_size=}')
    logger.info(f'{args.gradient_accumulation_steps=}')
    logger.info(f'(manually set) {effective_batch_size=}')
    logger.info(f'(manually set) {is_iterable_data=}')

    # Set max_steps based on num_tokens if it's iterable dataset
    if is_iterable_data:
        assert 'total_num_tokens' in config
        assert effective_batch_size is not None
        num_tokens = config['total_num_tokens']
        logger.info(f'(manually set) {num_tokens=}')
        seq_length = config['seq_length']
        args.max_steps = int(num_tokens // (effective_batch_size * seq_length))
        logger.info(f'setting max_steps={args.max_steps} based on num_tokens={num_tokens:2.2e}')

    return args


def print_trainable_parameters(model: nn.Module, debug=False):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if debug:
            logger.info(f"{name}: {param.numel()=} trainable={param.requires_grad}")
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_frac = 100 * trainable_params / all_param
    logger.info(
        f"trainable: {trainable_params} | all: {all_param} | trainable %: {trainable_frac:.2f}")


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
    elif path_or_name == "gpt2-xl":
        config = GPT2Config(
            vocab_size=50257,  # Size of the vocabulary
            n_positions=1024,  # Number of positional embeddings
            n_ctx=1024,
            n_embd=1600,  # Dimensionality of the embeddings and hidden states
            n_layer=48,  # Number of layers
            n_head=25,  # Number of attention heads
        )
        model = GPT2LMHeadModel(config)
    elif path_or_name == "gpt2-large":
        config = GPT2Config(
            vocab_size=50257,  # Standard GPT-2 vocabulary size
            n_positions=1024,  # Number of positional embeddings
            n_ctx=1024,
            n_embd=1280,  # Dimensionality of embeddings and hidden states for Large model
            n_layer=36,  # Number of layers for Large model
            n_head=20,  # Number of attention heads for Large model
        )
        model = GPT2LMHeadModel(config)

    print_trainable_parameters(model)
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

    ##### 3. Training GPT-2 XL (Original)
    # TODO: add contamination
    if config['model']['path_or_name'] == "gpt2-large":
        contam_name = args.contamination_dataset
        contam_factor = args.contamination_factor
        contam_mode = args.contamination_mode
        logger.info(
            f'Using TokenizedInMemoryDataset with {contam_name=} {contam_factor=} {contam_mode}')
        dataset_name = config['dataset']['dataset']
        train_dataset = StreamingSeqDataset(tokenizer,
                                            dataset_name,
                                            contam_ds_name=contam_name,
                                            contamination_factor=contam_factor,
                                            seq_length=config['seq_length'])

    ##### 2. Pre-filtering, with streaming dataset, same dataloading logic as before #####
    elif args.prefilter:
        eval_filter_name = args.prefilter_dataset
        filter_mode = args.prefilter_mode
        logger.info(
            f'Using PrefilteredTokenizedInMemoryDataset {eval_filter_name=} and {filter_mode=}')
        train_dataset = PrefilteredTokenizedInMemoryDataset(
            prefilter_dir=
            f'/shared/data2/minhaoj2/contamination/prefiltered_{eval_filter_name}_{filter_mode}_n7',
            datasets=config['dataset']['datasets'],
            eval_filter_name=eval_filter_name,
            filter_mode=filter_mode)

    # ##### 4. GPT-2_original (like #1), but use pre-tokenized dataset and DDP (like #3) #####
    # logger.info(f'Using TokenizedInMemoryDataset')
    # train_dataset = TokenizedInMemoryDataset(tokenized_data_dir='tokenized_data',
    #                                          datasets=config['dataset']['datasets'])

    ##### 5. GPT-2_text by adding eval dataset, but use pre-tokenized dataset and DDP (like #3) #####
    else:
        contam_name = args.contamination_dataset
        contam_factor = args.contamination_factor
        contam_mode = args.contamination_mode
        logger.info(
            f'Using TokenizedInMemoryDataset with {contam_name=} {contam_factor=} {contam_mode}')
        train_dataset = TokenizedInMemoryDataset(
            tokenized_data_dir='/shared/data2/minhaoj2/contamination/original_data',
            datasets=config['dataset']['datasets'],
            contamination_dataset_name=contam_name,
            contamination_factor=contam_factor,
            tokenizer=tokenizer,
            contamination_mode=contam_mode)

    is_iterable_data = isinstance(train_dataset, torch.utils.data.IterableDataset)
    if not is_iterable_data:
        logger.info(f'*** # examples (sequences) in train: {len(train_dataset)=} ***')

    logger.info(f'Loading TrainingArguments {is_iterable_data=}')
    training_args = prepare_trainer_arguments(config, is_iterable_data)

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
    parser.add_argument('--prefilter_dataset',
                        '-pd',
                        type=str,
                        default=None,
                        choices=["sst2", "mmlu", "cnn", "squad"])
    parser.add_argument('--prefilter_mode',
                        '-pm',
                        type=str,
                        default=None,
                        choices=["llama2", "ngram"])
    parser.add_argument('--contamination_dataset',
                        '-cd',
                        type=str,
                        default=None,
                        choices=["sst2", "mmlu", "cnn", "squad"])
    parser.add_argument('--contamination_factor', '-cf', type=int, default=1)
    parser.add_argument('--contamination_mode',
                        '-cm',
                        type=str,
                        default="text",
                        choices=["text", "gt"])
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
