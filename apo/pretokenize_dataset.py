import argparse
import os
import time
import random
from typing import Union, List, Set, Tuple, Generator, Optional, Any, Callable, Dict
from pathlib import Path
import pprint

import numpy as np
from loguru import logger
import yaml
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

import utils

Tokenizer = Union[AutoTokenizer, PreTrainedTokenizer]


def tokenize_dataset(pretrain_name: str,
                     tokenizer: Tokenizer,
                     out_dir: Optional[str] = None,
                     concat_token: Optional[str] = None):
    """Filters a pretraining dataset based on an evaluation dataset."""
    logger.info(f'Reading datasets {pretrain_name=}...')

    pretrain_dataset = load_dataset(pretrain_name, split='train')
    pretrain_dataset = pretrain_dataset.remove_columns(['meta', 'scores', 'avg_score', 'num_sents'])

    num_docs = len(pretrain_dataset)
    logger.info(f'Pretrain dataset number of examples (documents): {num_docs}')

    concat_token = concat_token or tokenizer.eos_token

    def tokenize_fn(example: Dict) -> Dict:
        document_sents = utils.process_document(example,
                                                is_split_by_sents=True,
                                                concat_token=concat_token)
        # Tokenize the individual sentences
        token_seqs = tokenizer(document_sents, truncation=False).input_ids
        return {'document_tokens': token_seqs}

    logger.info(f'Mapping tokenization on documents...')
    pretrain_dataset = pretrain_dataset.map(tokenize_fn,
                                            num_proc=os.cpu_count(),
                                            remove_columns=['texts'])

    if list(pretrain_dataset.features) != ['document_tokens']:
        logger.warning(f'Only 1 column is expected, got {pretrain_dataset.features}')

    logger.info(f'Number of documents after filtering: {len(pretrain_dataset)}')

    if out_dir is None:
        logger.info(f'`out_dir` not provided; not saving filtered dataset')
    else:
        logger.info(f'Writing filtered dataset to {out_dir}...')
        out_name = f'{pretrain_name.replace("/", "-")}'
        out_path = Path(out_dir) / out_name
        os.makedirs(out_dir, exist_ok=True)
        pretrain_dataset.save_to_disk(out_path)
        logger.info(f'Saved to {out_path}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        required=True,
                        type=str,
                        help='a path to a YAML file with configuration')

    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.full_load(file)

    out_dir = '/shared/data2/minhaoj2/contamination/'
    logger.add('pretokenize_dataset.log')

    tokenizer = utils.prepare_tokenizer(**config['tokenizer'])
    pretrain_names = config['dataset']['datasets']
    logger.info(f'Pre-tokenizing the following:\n{pprint.pformat(pretrain_names)}')

    for pretrain_dataset_name in pretrain_names:
        logger.info(f'Tokenizing {pretrain_dataset_name=} ')
        tokenize_dataset(pretrain_name=pretrain_dataset_name, tokenizer=tokenizer, out_dir=out_dir)