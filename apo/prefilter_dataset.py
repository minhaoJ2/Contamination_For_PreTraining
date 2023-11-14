import argparse
import os
import time
import random
from typing import Union, List, Set, Tuple, Generator, Optional, Any, Callable, Dict
from pathlib import Path

import numpy as np
from loguru import logger
import yaml
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import utils

Tokenizer = Union[AutoTokenizer, PreTrainedTokenizer]


def filter_dataset(pretrain_name: str,
                   eval_name: str,
                   tokenizer: Tokenizer,
                   out_dir: Optional[str] = None,
                   filter_mode: str = 'ngram',
                   filter_threshold: Optional[float] = None,
                   ngram: int = 13,
                   concat_token: Optional[str] = None):
    """Filters a pretraining dataset based on an evaluation dataset."""
    logger.info(f'Filtering input_args: {filter_mode=}, {filter_threshold=}, {ngram=}')
    logger.info(f'Reading datasets {pretrain_name=} and {eval_name=}...')

    pretrain_dataset = load_dataset(pretrain_name, split='train')
    eval_dataset = utils.read_eval_dataset(eval_name)
    num_docs = len(pretrain_dataset)
    # Drop unnecessary columns (see tomekkorbak/detoxify-pile-chunk3-0-50000)
    pretrain_dataset = pretrain_dataset.remove_columns(['meta', 'scores', 'avg_score', 'num_sents'])
    logger.info(f'Pretrain dataset number of examples (documents): {num_docs}')

    logger.info(f'Building eval ngram lookup...')
    if eval_name == 'cnn':
        logger.info(f'*** For cnn_dailymail, we will split the document into sentences ***')
        def split_into_sents(example):
            return {'texts': sent_tokenize(example['texts'])}
        eval_dataset = eval_dataset.map(split_into_sents, num_proc=os.cpu_count())

    eval_ngrams = utils.build_eval_ngram_lookup(eval_dataset,
                                                tokenizer,
                                                ngram=ngram,
                                                text_key="texts")
    concat_token = concat_token or tokenizer.eos_token
    #####################################
    if filter_mode == 'llama2':
        # For Llama 2 specifically, we need to do two passes: 1. go through train seqs
        # and obtain all contaiminated tokens; 2. filter out all documents with >80%
        # contaminated tokens (perhaps on a sentence level, consistent with other methods)
        def llama2_tokenize_and_track_contamination(example: Dict) -> Dict:
            # `is_split_by_sents` is True since we are mapping pre-training dataset
            document_sents = utils.process_document(example,
                                                    is_split_by_sents=True,
                                                    concat_token=concat_token)
            token_seqs = tokenizer(document_sents, truncation=False).input_ids
            # Log the total number of tokens in this doc
            num_tokens = sum([len(seq) for seq in token_seqs])
            # Collect all contaminated tokens
            example_contam_tokens = set()
            for token_seq in token_seqs:
                contam_tokens = utils.contaminated_tokens_llama2(token_seq, eval_ngrams, ngram,
                                                                 filter_threshold)
                example_contam_tokens.update(contam_tokens)
            return {
                'document_tokens': token_seqs,
                'num_tokens': num_tokens,
                'contam_tokens': np.array(list(example_contam_tokens), dtype=int)
            }

        logger.info(f'(Llama2 filter) Tokenizing and flagging documents...')
        pretrain_dataset = pretrain_dataset.add_column('contam_tokens',
                                                       [[] for _ in range(num_docs)])
        pretrain_dataset = pretrain_dataset.map(llama2_tokenize_and_track_contamination,
                                                num_proc=os.cpu_count(),
                                                remove_columns=['texts'])

        logger.info(f'(Llama2 filter) collecting contaminated tokens ...')
        all_contaminated_tokens = set()
        for example in pretrain_dataset:
            all_contaminated_tokens.update(example['contam_tokens'])
        logger.info(f'(Llama2 filter) # contaminated tokens: {len(all_contaminated_tokens)}')

        # Count the number of tokens to be thrown away by flagging
        def llama2_flag(example: Dict) -> Dict:
            for token_seq in example['document_tokens']:
                # Check contamination on the sentence level
                num_contaminated_tokens = 0
                for token in token_seq:
                    if token in all_contaminated_tokens:
                        num_contaminated_tokens += 1
                if num_contaminated_tokens / len(token_seq) > filter_threshold:
                    return {'keep': 0}
            return {'keep': 1}

        logger.info(f'(Llama2 filter) flagging documents ...')
        pretrain_dataset = pretrain_dataset.map(llama2_flag, num_proc=os.cpu_count())
        doc_num_tokens = np.array(pretrain_dataset['num_tokens'])
        mask = np.array(pretrain_dataset['keep'])
        thrown_tokens = doc_num_tokens * (1 - mask)
        logger.info(f'(Llama2 filter) Thrown/Total # tokens = {sum(thrown_tokens)}/{sum(doc_num_tokens)} in {pretrain_name=}...')
        logger.info(f'(Llama2 filter) filtering documents ...')
        pretrain_dataset = pretrain_dataset.filter(lambda x: x['keep'] == 1, num_proc=os.cpu_count())
        pretrain_dataset = pretrain_dataset.remove_columns(['contam_tokens'])

    else:
        seq_filter_fn = utils.get_seq_filter_fn(filter_mode)

        # Processes a document into tokens and a flag indicating whether it should be dropped.
        def tokenize_and_flag(example: Dict) -> Dict:
            # `is_split_by_sents` is True since we are mapping pre-training dataset
            document_sents = utils.process_document(example,
                                                    is_split_by_sents=True,
                                                    concat_token=concat_token)
            # Tokenize the individual sentences
            token_seqs = tokenizer(document_sents, truncation=False).input_ids
            # Log the total number of tokens in this doc
            num_tokens = sum([len(seq) for seq in token_seqs])
            # Apply filter logic on sentence level; throw whole document if any sentence is bad
            for token_seq in token_seqs:
                if seq_filter_fn(train_tokens=token_seq,
                                 eval_ngrams=eval_ngrams,
                                 ngram=ngram,
                                 dirty_threshold=filter_threshold):
                    return {'document_tokens': token_seqs, 'num_tokens': num_tokens, 'keep': 0}

            # Each document is a list of sents, each sent is a list of tokens
            return {'document_tokens': token_seqs, 'num_tokens': num_tokens, 'keep': 1}

        logger.info(f'Tokenizing and flagging documents...')
        pretrain_dataset = pretrain_dataset.map(tokenize_and_flag,
                                                num_proc=os.cpu_count(),
                                                remove_columns=['texts'])
        doc_num_tokens = np.array(pretrain_dataset['num_tokens'])
        mask = np.array(pretrain_dataset['keep'])
        thrown_tokens = doc_num_tokens * (1 - mask)
        logger.info(f'Thrown/Total # tokens = {sum(thrown_tokens)}/{sum(doc_num_tokens)} in {pretrain_name=}...')


        logger.info(f'Filtering documents...')
        pretrain_dataset = pretrain_dataset.filter(lambda x: x['keep'] == 1,
                                                   num_proc=os.cpu_count())
        pretrain_dataset = pretrain_dataset.remove_columns(['keep'])  # no longer needed

    if list(pretrain_dataset.features) != ['document_tokens']:
        logger.warning(f'Only 1 column is expected, got {pretrain_dataset.features}')

    contam_ratio = (1.0 - (len(pretrain_dataset) / num_docs)) * 100
    token_ratio = sum(thrown_tokens) / sum(doc_num_tokens) * 100
    logger.info(f'Number of documents after filtering: {len(pretrain_dataset)}')
    logger.info(f'Contamination ratio: {contam_ratio:.2f}% (clean: {(100 - contam_ratio):.2f}%)')
    logger.info(f'Token removed ratio: {token_ratio:.2f}% ')
    if out_dir is None:
        logger.info(f'`out_dir` not provided; not saving filtered dataset')
    else:
        logger.info(f'Writing filtered dataset to {out_dir}...')
        out_name = f'{pretrain_name.replace("/", "-")}_{eval_name}_{filter_mode}_filtered'
        out_path = Path(out_dir) / out_name
        os.makedirs(out_dir, exist_ok=True)
        pretrain_dataset.save_to_disk(out_path)  # debug skip saving data for now

        # Save thrown tokens and doc tokens as python lists in a single file
        thrown_path = Path(out_dir) / f'{out_name}_thrown_tokens.npy'
        np.savez(thrown_path, thrown_tokens=thrown_tokens, doc_num_tokens=doc_num_tokens)
        logger.info(f'Saved to {out_path}')

    return contam_ratio, sum(thrown_tokens), sum(doc_num_tokens)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        required=True,
                        type=str,
                        help='a path to a YAML file with configuration')

    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.full_load(file)

    pretrain_names = config['dataset']['datasets']
    print(pretrain_names)

    # Start filtering
    tokenizer = utils.prepare_tokenizer(**config['tokenizer'])

    # For each eval set, create a pre-filtered, tokenized copy for each shard of pretrain dataset
    # NOTE: it may save more space to only store the filtered indices in pretraining set,
    # but storing the tokens should also speed up pretraining (less streaming data processing)
    out_dir = '/Users/minhaoj2/Downloads/contamination/prefiltered_sst2/'

    logger.add('prefilter_dataset_sst2_llama2_n7_thr0.8.log')


    for eval_name in ['sst2']:
        shared_args = dict(eval_name=eval_name, out_dir=out_dir, tokenizer=tokenizer)

        # logger.info(f'\t***** Llama 2 (ngram, threshold)')
        # for config in [(7, 0.8)]:
        #     llama2_ratios = []
        #     llama2_thrown_token = []
        #     llama2_total_token = []
        #     ngram_len, threshold = config
        #     for pretrain_dataset_name in pretrain_names:
        #         contam_ratio, thrown_tokens, total_tokens = filter_dataset(**shared_args,
        #                                       pretrain_name=pretrain_dataset_name,
        #                                       filter_mode='llama2',
        #                                       filter_threshold=threshold,
        #                                       ngram=ngram_len)
        #         llama2_ratios.append(contam_ratio)
        #         llama2_thrown_token.append(thrown_tokens)
        #         llama2_total_token.append(total_tokens)
        #     ratio = sum(llama2_thrown_token) / sum(llama2_total_token) * 100
        #     logger.info(f'***Llama 2 contamination ratio for {eval_name=}, {config=}: '
        #                 f'{llama2_ratios=} {np.mean(llama2_ratios)=}')
        #     logger.info(f'***Llama 2 token contamination ratio for {eval_name=}, {config=}: '
        #                 f'{sum(llama2_thrown_token)} / {sum(llama2_total_token)} = {ratio:.2f}%')
            

        for ngram_len in [6]:
            ngram_ratios = []
            ngram_thrown_tokens = []
            ngram_total_token = []
            for pretrain_dataset_name in pretrain_names:
                contam_ratio, thrown_tokens, total_tokens = filter_dataset(**shared_args,
                                              pretrain_name=pretrain_dataset_name,
                                              filter_mode='ngram',
                                              ngram=ngram_len)
                ngram_ratios.append(contam_ratio)
                ngram_thrown_tokens.append(thrown_tokens)
                ngram_total_token.append(total_tokens)
            ratio = sum(ngram_thrown_tokens) / sum(ngram_total_token) * 100
            logger.info(f'***N-gram contamination ratio for {eval_name=}, {ngram_len=}: '
                        f'{ngram_ratios=} {np.mean(ngram_ratios)=}')
            logger.info(f'***N-gram token contamination ratio for {eval_name=}, {ngram_len=}: '
                        f'{sum(ngram_thrown_tokens)} / {sum(ngram_total_token)} = {ratio:.2f}%')