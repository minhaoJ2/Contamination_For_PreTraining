import argparse
import os
import time
import random
from typing import Union, List, Set, Tuple, Generator, Optional, Any, Callable, Dict
from pathlib import Path

from loguru import logger
import yaml
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

import utils

Tokenizer = Union[AutoTokenizer, PreTrainedTokenizer]


def filter_dataset(pretrain_name: str,
                   eval_name: str,
                   out_dir: str,
                   tokenizer: Tokenizer,
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
    eval_ngrams = utils.build_eval_ngram_lookup(eval_dataset,
                                                tokenizer,
                                                ngram=ngram,
                                                text_key="texts")

    #####################################
    if filter_mode == 'llama2':
        # For Llama 2 specifically, we need to do two passes: 1. go through train seqs
        # and obtain all contaiminated tokens; 2. filter out all documents with >80%
        # contaminated tokens (perhaps on a sentence level, consistent with other methods)
        def llama2_tokenize_and_track_contamination(example: Dict) -> Dict:
            document_sents = utils.process_document(example,
                                                    is_split_by_sents=True,
                                                    concat_token=concat_token or
                                                    tokenizer.eos_token)
            token_seqs = tokenizer(document_sents, truncation=False).input_ids
            # Collect all contaminated tokens
            all_contaminated_tokens = set()
            for token_seq in token_seqs:
                contaminated_tokens = utils.contaminated_tokens_llama2(
                    token_seq, eval_ngrams, ngram, filter_threshold)
                all_contaminated_tokens.union(contaminated_tokens)
            return {'document_tokens': token_seqs, 'contaminated_tokens': all_contaminated_tokens}

        logger.info(f'(Llama2 filter) Tokenizing and flagging documents...')
        pretrain_dataset = pretrain_dataset.map(llama2_tokenize_and_track_contamination,
                                                num_proc=os.cpu_count(),
                                                remove_columns=['texts'])

        logger.info(f'(Llama2 filter) collecting contaminated tokens ...')
        all_contaminated_tokens = set()
        for example in pretrain_dataset:
            all_contaminated_tokens.union(example['contaminated_tokens'])
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
        pretrain_dataset = pretrain_dataset.remove_columns(['contaminated_tokens'])

    else:
        seq_filter_fn = utils.get_seq_filter_fn(filter_mode)

        # Processes a document into tokens and a flag indicating whether it should be dropped.
        def tokenize_and_flag(example: Dict) -> Dict:
            document_sents = utils.process_document(example,
                                                    is_split_by_sents=True,
                                                    concat_token=concat_token or
                                                    tokenizer.eos_token)
            # Tokenize the individual sentences
            token_seqs = tokenizer(document_sents, truncation=False).input_ids
            # Apply filter logic on sentence level; throw whole document if any sentence is bad
            for token_seq in token_seqs:
                if seq_filter_fn(token_seq, eval_ngrams, ngram, filter_threshold):
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

    logger.info(f'Number of documents after filtering: {len(pretrain_dataset)}'
                f' ({len(pretrain_dataset) / num_docs * 100:.2f}%)')
    logger.info(f'Writing filtered dataset to {out_dir}...')
    out_name = f'{pretrain_name.replace("/", "-")}_{eval_name}_{filter_mode}_filtered'
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

    pretrain_names = config['dataset']['datasets']
    print(pretrain_names)

    # Start filtering
    tokenizer = utils.prepare_tokenizer(**config['tokenizer'])

    # For each eval set, create a pre-filtered, tokenized copy for each shard of pretrain dataset
    # NOTE: it may save more space to only store the filtered indices in pretraining set,
    # but storing the tokens should also speed up pretraining (less streaming data processing)
    out_dir = 'prefiltered_data/'
    # for eval_name in ['sst2', 'cnn', 'ag_news']:
    for eval_name in ['sst2']:  # DEBUG: test on only sst2
        for pretrain_dataset_name in pretrain_names:
            logger.info(f'Pre-filtering {pretrain_dataset_name=} with {eval_name=}...')
            logger.info(f'\tDirect n-gram overlap')
            filter_dataset(pretrain_name=pretrain_dataset_name,
                           eval_name=eval_name,
                           out_dir=out_dir,
                           tokenizer=tokenizer,
                           filter_mode='ngram',
                           ngram=13)
            logger.info(f'\tPaLM')
            filter_dataset(pretrain_name=pretrain_dataset_name,
                           eval_name=eval_name,
                           out_dir=out_dir,
                           tokenizer=tokenizer,
                           filter_mode='palm',
                           filter_threshold=0.7,
                           ngram=8)
            logger.info(f'\tLlama2')
            filter_dataset(pretrain_name=pretrain_dataset_name,
                           eval_name=eval_name,
                           out_dir=out_dir,
                           tokenizer=tokenizer,
                           filter_mode='llama2',
                           filter_threshold=0.7,
                           ngram=11)

            # DEBUG: test on one shard for now
            break
