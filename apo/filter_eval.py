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
from nltk.tokenize import sent_tokenize
import json

import utils

Tokenizer = Union[AutoTokenizer, PreTrainedTokenizer]


def filter_dataset(pretrain_name: str,
                   eval_name: str,
                   tokenizer: Tokenizer,
                   out_dir: Optional[str] = None,
                   ngram: int = 13,
                   concat_token: Optional[str] = None):
    """Filters a pretraining dataset based on an evaluation dataset."""
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

    def llama2_tokenize_and_track_contamination(example: Dict) -> Dict:
        document_sents = utils.process_document(example,
                                                is_split_by_sents=True,
                                                concat_token=concat_token)
        token_seqs = tokenizer(document_sents, truncation=False).input_ids
        # Collect all contaminated tokens
        example_contam_tokens = set()
        for token_seq in token_seqs:
            contam_tokens = utils.contaminated_tokens_llama2(token_seq, eval_ngrams, ngram)
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
    return all_contaminated_tokens

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
    out_dir = None
    logger.add("filtering_eval.log")
    # Start filtering
    tokenizer = utils.prepare_tokenizer(**config['tokenizer'])
    
    for eval_name in ['cnn']:
        shared_args = dict(eval_name=eval_name, out_dir=out_dir, tokenizer=tokenizer)
        if eval_name == 'sst2':
            n_gram, threshold1, threshold2 = 6, 0.8, 0.4
        elif eval_name == 'ag_news':
            n_gram, threshold1, threshold2 = 8, 0.8, 0.4
        elif eval_name == 'cnn':
            n_gram, threshold1, threshold2 = 15, 0.8, 0.4
        elif eval_name == 'squad':
            n_gram, threshold1, threshold2 = 9, 0.8, 0.4
        eval_dataset = utils.read_eval_dataset(eval_name)
        all_contaminated_tokens = set()
        for pretrain_dataset_name in pretrain_names:
            tokens = filter_dataset(**shared_args,
                                    pretrain_name=pretrain_dataset_name,
                                    ngram=n_gram)
            all_contaminated_tokens.update(tokens)
        
        logger.info(f"Total contaminaed tokens: {len(all_contaminated_tokens)}")

        def llama2_filter(example: Dict) -> bool:
            text = example['texts']
            token_seq = tokenizer(text, truncation=False).input_ids
                # Check contamination on the sentence level
            num_contaminated_tokens = 0
            for token in token_seq:
                if token in all_contaminated_tokens:
                    num_contaminated_tokens += 1
            ratio = num_contaminated_tokens / len(token_seq)
            return {
                "ratio": ratio
            }
        logger.info(f'(Llama2 filter) filtering documents ...')

        eval_dataset = eval_dataset.map(llama2_filter, num_proc=os.cpu_count())
        id2ratio = {}
        clean_count, dirty_count = 0, 0
        for i in range(len(eval_dataset)):
            ratio = eval_dataset[i]['ratio']
            if ratio < threshold2:
                clean_count += 1
            if ratio > threshold1:
                dirty_count += 1
            id2ratio[i] = ratio
        logger.info(f"Clean ratio: {clean_count / len(eval_dataset)}")
        logger.info(f"Dirty ratio: {dirty_count / len(eval_dataset)}")
        with open(f"./{eval_name}_ratio.json", 'w') as f:
            json.dump(id2ratio, f)