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
nltk.data.path.append('/lfs/local/0/kzliu/nltk_data/')  # for mercury nodes

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
            document_sents = utils.process_document(example,
                                                    is_split_by_sents=True,
                                                    concat_token=concat_token)
            token_seqs = tokenizer(document_sents, truncation=False).input_ids
            # Collect all contaminated tokens
            example_contam_tokens = set()
            for token_seq in token_seqs:
                contam_tokens = utils.contaminated_tokens_llama2(token_seq, eval_ngrams, ngram,
                                                                 filter_threshold)
                example_contam_tokens.update(contam_tokens)
            return {
                'document_tokens': token_seqs,
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

        def llama2_filter(example: Dict) -> bool:
            for token_seq in example['document_tokens']:
                # Check contamination on the sentence level
                num_contaminated_tokens = 0
                for token in token_seq:
                    if token in all_contaminated_tokens:
                        num_contaminated_tokens += 1
                if num_contaminated_tokens / len(token_seq) > filter_threshold:
                    return False
            return True

        logger.info(f'(Llama2 filter) filtering documents ...')
        pretrain_dataset = pretrain_dataset.filter(llama2_filter, num_proc=os.cpu_count())
        pretrain_dataset = pretrain_dataset.remove_columns(['contam_tokens'])

    else:
        seq_filter_fn = utils.get_seq_filter_fn(filter_mode)

        # Processes a document into tokens and a flag indicating whether it should be dropped.
        def tokenize_and_flag(example: Dict) -> Dict:
            document_sents = utils.process_document(example,
                                                    is_split_by_sents=True,
                                                    concat_token=concat_token)
            # Tokenize the individual sentences
            token_seqs = tokenizer(document_sents, truncation=False).input_ids
            # Apply filter logic on sentence level; throw whole document if any sentence is bad
            for token_seq in token_seqs:
                if seq_filter_fn(train_tokens=token_seq,
                                 eval_ngrams=eval_ngrams,
                                 ngram=ngram,
                                 dirty_threshold=filter_threshold):
                    return {'document_tokens': token_seqs, 'keep': 0}

            # Each document is a list of sents, each sent is a list of tokens
            return {'document_tokens': token_seqs, 'keep': 1}

        logger.info(f'Tokenizing and flagging documents...')
        pretrain_dataset = pretrain_dataset.map(tokenize_and_flag,
                                                num_proc=os.cpu_count(),
                                                remove_columns=['texts'])
        logger.info(f'Filtering documents...')
        pretrain_dataset = pretrain_dataset.filter(lambda x: x['keep'] == 1,
                                                   num_proc=os.cpu_count())
        pretrain_dataset = pretrain_dataset.remove_columns(['keep'])  # no longer needed

    if list(pretrain_dataset.features) != ['document_tokens']:
        logger.warning(f'Only 1 column is expected, got {pretrain_dataset.features}')

    contam_ratio = (1.0 - (len(pretrain_dataset) / num_docs)) * 100
    logger.info(f'Number of documents after filtering: {len(pretrain_dataset)}')
    logger.info(f'Contamination ratio: {contam_ratio:.2f}% (clean: {(100 - contam_ratio):.2f}%)')

    if out_dir is None:
        logger.info(f'`out_dir` not provided; not saving filtered dataset')
    else:
        logger.info(f'Writing filtered dataset to {out_dir}...')
        out_name = f'{pretrain_name.replace("/", "-")}_{eval_name}_{filter_mode}_filtered'
        out_path = Path(out_dir) / out_name
        os.makedirs(out_dir, exist_ok=True)
        pretrain_dataset.save_to_disk(out_path)
        logger.info(f'Saved to {out_path}')
    return contam_ratio


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
    out_dir = '/shared/data2/minhaoj2/contamination/prefiltered_data_agnews/'
    # out_dir = None

    # Add output file
    # logger.add('prefilter_dataset_cnn_llama2_13_14.log')
    logger.add('prefilter_dataset_agnews.log')
    # logger.add('prefilter_dataset_cnn_ngram.log')

    # for eval_name in ['sst2', 'cnn', 'ag_news']:
    # for eval_name in ['sst2']:
    for eval_name in ['ag_news']:

        shared_args = dict(eval_name=eval_name, out_dir=out_dir, tokenizer=tokenizer)

        logger.info(f'\t***** Llama 2 (ngram, threshold)')
        for config in [(8, 0.8)]:  # kzl: CNN DAILYMAIL dataset caching; this gives ~6% contam
            llama2_ratios = []
            ngram_len, threshold = config
            for pretrain_dataset_name in pretrain_names:
                contam_ratio = filter_dataset(**shared_args,
                                              pretrain_name=pretrain_dataset_name,
                                              filter_mode='llama2',
                                              filter_threshold=threshold,
                                              ngram=ngram_len)
                llama2_ratios.append(contam_ratio)
            logger.info(f'***Llama 2 contamination ratio for {eval_name=}, {config=}: '
                        f'{llama2_ratios=} {np.mean(llama2_ratios)=}')