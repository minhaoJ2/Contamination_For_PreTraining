import os
import time
from typing import Any, Generator, Optional, Union, Dict, List
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizer
from loguru import logger

import apo.utils as utils

Tokenizer = Union[AutoTokenizer, PreTrainedTokenizer]


class StreamingSeqDataset(TorchIterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.

    Based on https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/codeparrot_training.py
    """

    def __init__(
        self,
        tokenizer,
        pretrain_ds_name: str,
        contam_ds_name: Optional[str] = None,
        contamination_factor: Optional[int] = 1,
        contamination_mode: Optional[str] = None,
        seq_length: int = 1024,
        num_docs_buffered: int = 100,
        is_split_by_sentences: bool = False,
    ):
        self.pretrain_ds_name = pretrain_ds_name
        self.contam_ds_name = contam_ds_name
        self.contamination_factor = contamination_factor
        self.tokenizer = tokenizer
        self.concat_token = tokenizer.eos_token
        self.concat_token_id = tokenizer.eos_token_id
        self.seq_length = seq_length
        self.is_split_by_sentences = is_split_by_sentences
        self.num_docs_buffered = num_docs_buffered
        self.num_docs = 0
        self.num_tokens_seen = 0
        self.global_iters = 0
        self.prev_time = time.perf_counter()  ## debug
        # self.load_pretrain_ds()
        if self.contam_ds_name:
            self.dataset_names = [self.contam_ds_name] * contamination_factor + [self.pretrain_ds_name]
        else:
            self.dataset_names = [self.pretrain_ds_name]

    def load_pretrain_ds(self):
        ds = load_dataset(self.pretrain_ds_name, split='train', streaming=True)
        self.pretrain_ds = iter(ds)

    @property
    def tokens_used(self) -> int:
        return self.num_tokens_seen

    def __iter__(self):
        # kzl NOTE: the implementation here does NOT handle dataset sharding;
        # i.e. when using `DistributedDataParallel`, each process will iterate
        # over the entire dataset and yield the same examples.
        # This does work with `DataParallel` so existing results should make sense.
        # kzl (nov 14): try with FSDP and see if each worker is returning the
        # same dataset or not; otherwise we will implement some form of
        # yielding based on worker rank
        for dataset_name in self.dataset_names:
            print(f'Starting processing examples from dataset {dataset_name}')
            if dataset_name == "sst2":
                dataset = load_dataset("glue", "sst2", split="train",
                                       streaming=True).rename_column("sentence", "texts")
            elif dataset_name == "cnn":
                dataset = load_dataset("cnn_dailymail", "3.0.0", split="test",
                                       streaming=True).rename_column("article", "texts")
            elif dataset_name == "mmlu":
                dataset = load_dataset("cais/mmlu", "all", split="test",
                                       streaming=True).rename_column("question", "texts")
            else:
                dataset = load_dataset(dataset_name, split='train', streaming=True)
            dataset_iterator = iter(dataset)
            buffer = []
            while True:
                try:
                    for _ in range(self.num_docs_buffered):
                        document = next(dataset_iterator)
                        self.num_docs += 1
                        doc_tokens = self.tokenize_document(document, dataset_name)
                        buffer.extend(doc_tokens)

                    for i in range(0, len(buffer), self.seq_length):
                        input_ids = buffer[i:i + self.seq_length]
                        if len(input_ids) == self.seq_length:
                            self.num_tokens_seen += self.seq_length
                            self.global_iters += 1
                            yield {
                                'input_ids': torch.tensor(input_ids),
                                'labels': torch.tensor(input_ids.copy()),
                            }

                    buffer = buffer[i:]

                except StopIteration:
                    logger.info(f'Pre-training data {dataset_name} exhausted!')
                    break

    def tokenize_document(self,
                          document: dict[str, Any],
                          dataset_name,
                          text_key='text') -> List[int]:
        """Tokenize a document into a list of sentences."""
        doc_tokens = []
        if self.is_split_by_sentences:
            document_text = document[text_key]
            sent_tokens = self.tokenizer(document_text, truncation=False)
            sent_tokens = sent_tokens['input_ids']

            for sent in sent_tokens:
                doc_tokens.extend(sent)
                doc_tokens.append(self.concat_token_id)
        else:
            if dataset_name == "cnn":
                text = concat_token + document['texts'] + concat_token + " TL;DR: " + document[
                    'highlights']
                doc_tokens = self.tokenizer(text, truncation=False)['input_ids']
            elif dataset_name == "mmlu":
                text = utils.get_mmlu_prompt(document, self.concat_token)
                text = text[0]
                doc_tokens = self.tokenizer(text, truncation=False)['input_ids']
            else:
                document_text = document[text_key]
                doc_tokens = self.tokenizer(document_text, truncation=False)['input_ids']
            doc_tokens.append(self.concat_token_id)
        return doc_tokens


class PrefilteredTokenizedDataset(TorchIterableDataset):

    def __init__(
        self,
        prefilter_dir: str,
        datasets: list[str],
        eval_filter_name: str,
        filter_mode: str,
        seq_length: int = 1024,
        num_of_sequences: int = 1024,
        skip_tokens: int = 0,
    ):
        self.datasets = datasets
        self.prefilter_dir = prefilter_dir
        self.eval_filter_name = eval_filter_name
        self.filter_mode = filter_mode
        self.seq_length = seq_length
        self.current_size = 0
        self.num_docs = 0
        self.max_buffer_size = seq_length * num_of_sequences
        self.skip_tokens = skip_tokens
        self.prev_time = time.perf_counter()  ## debug

    @property
    def tokens_used(self) -> int:
        return self.current_size * self.seq_length

    def __iter__(self):
        for dataset_name in self.datasets:
            # this follows from `prefilter_dataset.py`
            prefiltered_path = f'{self.prefilter_dir}/{dataset_name.replace("/", "-")}_{self.eval_filter_name}'
            prefiltered_path += f'_{self.filter_mode}_filtered'
            logger.info(f'Reading from dataset "{prefiltered_path}"')
            dataset = load_from_disk(prefiltered_path)

            logger.info(f'Processing examples (pre-tokenized)...')
            iterator = iter(dataset)
            more_examples = True

            # Same as previous class, but we don't need to tokenize
            # Also we can merge sentence tokens into the same buffer without document separation
            while more_examples:
                token_buffer, buffer_len = [], 0
                while True:
                    if buffer_len >= self.max_buffer_size:
                        break
                    try:
                        document = next(iterator)
                        self.num_docs += 1
                        token_buffer.extend(document['document_tokens'])
                        buffer_len += sum(len(tokens) for tokens in document['document_tokens'])
                    except StopIteration:
                        more_examples = False
                        break

                all_token_ids = []
                for tokenized_input in token_buffer:
                    all_token_ids.extend(tokenized_input)

                for i in range(0, len(all_token_ids), self.seq_length):
                    input_ids = all_token_ids[i:i + self.seq_length]
                    if len(input_ids) == self.seq_length:
                        self.current_size += 1
                        if self.skip_tokens > self.tokens_used:
                            if self.tokens_used % (self.seq_length * 1e5) == 0:
                                print(f'Skipping {self.tokens_used:2.4e} tokens')
                            continue
                        yield {
                            'input_ids': torch.tensor(input_ids),
                            'labels': torch.tensor(input_ids.copy()),
                        }

    def shuffle(self, buffer_size: int = 1000) -> ShufflerIterDataPipe:
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)


class PrefilteredTokenizedInMemoryDataset(Dataset):

    def __init__(
        self,
        prefilter_dir: str,
        datasets: list[str],
        eval_filter_name: str,
        filter_mode: str,
        seq_length: int = 1024,
    ):
        chunk_tokens_tensors = []
        for dataset_name in datasets:
            prefiltered_path = f'{prefilter_dir}/{dataset_name.replace("/", "-")}_{eval_filter_name}'
            prefiltered_path += f'_{filter_mode}_filtered'
            logger.info(f'Reading from dataset "{prefiltered_path}"')
            dataset = load_from_disk(prefiltered_path)

            all_token_ids = []
            for document in dataset:
                for sent_tokens in document['document_tokens']:
                    all_token_ids.extend(sent_tokens)

            chunk_tokens = torch.tensor(all_token_ids)
            chunk_tokens = chunk_tokens[:len(chunk_tokens) // seq_length * seq_length]
            chunk_tokens = chunk_tokens.view(-1, seq_length)
            chunk_tokens_tensors.append(chunk_tokens)

        chunk_tokens_tensor = torch.cat(chunk_tokens_tensors, dim=0)
        self.chunk_tokens_tensor = chunk_tokens_tensor

    def __len__(self):
        return len(self.chunk_tokens_tensor)

    def __getitem__(self, index):
        return {
            'input_ids': self.chunk_tokens_tensor[index],
            'labels': self.chunk_tokens_tensor[index].clone(),
        }


class TokenizedInMemoryDataset(Dataset):
    """Same as `PrefilteredTokenizedInMemoryDataset`, except the dataset is NOT pre-filtered.

    This is to make sure we can train a "GPT-2_original" model with exactly the same process
    as the "GPT-2_clean" models.
    """

    def __init__(
        self,
        tokenized_data_dir: str,
        datasets: list[str],
        seq_length: int = 1024,
        contamination_dataset_name: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        concat_token: Optional[str] = None,
        contamination_factor: int = 1,
        contamination_mode: str = "text",
    ):
        tokenized_datasets = []

        if contamination_dataset_name:
            if tokenizer is None:
                raise ValueError(f'`tokenizer` must be provided when using contamination')

            logger.info(f'*** Using {contamination_dataset_name=} with {contamination_factor=}...')
            contamination_dataset = utils.read_eval_dataset(contamination_dataset_name)
            concat_token = concat_token or tokenizer.eos_token

            def tokenize_fn(example: Dict) -> Dict:
                document_sents = utils.process_document(example,
                                                        is_split_by_sents=False,
                                                        concat_token=concat_token)

                token_seqs = tokenizer(document_sents, truncation=False).input_ids
                return {'document_tokens': token_seqs}

            def tokenize_fn_prompt(example: Dict) -> Dict:
                document_sents = utils.process_document(example,
                                                        is_split_by_sents=False,
                                                        concat_token=concat_token,
                                                        contamination_mode=contamination_mode,
                                                        dataset=contamination_dataset_name)
                token_seqs = tokenizer(document_sents, truncation=False).input_ids
                return {'document_tokens': token_seqs}

            if contamination_mode == "text":
                contamination_dataset = contamination_dataset.map(tokenize_fn,
                                                                  num_proc=os.cpu_count(),
                                                                  remove_columns=['texts'])
            elif contamination_mode == "gt":
                contamination_dataset = contamination_dataset.map(tokenize_fn_prompt,
                                                                  num_proc=os.cpu_count(),
                                                                  remove_columns=['texts'])
            else:
                raise ValueError(f"Contamination mode={contamination_mode} not supported!")

            tokenized_datasets.extend([contamination_dataset] * contamination_factor)
        # Load the pre-training datasets
        for dataset_name in datasets:
            tokenized_data_path = f'{tokenized_data_dir}/{dataset_name.replace("/", "-")}'
            logger.info(f'Reading from dataset "{tokenized_data_path}"')
            pretrain_dataset = load_from_disk(tokenized_data_path)
            tokenized_datasets.append(pretrain_dataset)
        for _ in range(contamination_factor):
            datasets.insert(0, contamination_dataset_name)

        logger.info(f"The total number of dataset chunks is {len(datasets)=}")
        chunk_tokens_tensors = []
        assert len(datasets) == len(tokenized_datasets)
        for dataset_name, dataset in zip(datasets, tokenized_datasets):
            logger.info(f'Processing dataset "{dataset_name}"')
            all_token_ids = []
            for document in dataset:
                for sent_tokens in document['document_tokens']:
                    all_token_ids.extend(sent_tokens)

            chunk_tokens = torch.tensor(all_token_ids)
            chunk_tokens = chunk_tokens[:len(chunk_tokens) // seq_length * seq_length]
            chunk_tokens = chunk_tokens.view(-1, seq_length)
            chunk_tokens_tensors.append(chunk_tokens)

        chunk_tokens_tensor = torch.cat(chunk_tokens_tensors, dim=0)
        self.chunk_tokens_tensor = chunk_tokens_tensor

    def __len__(self):
        return len(self.chunk_tokens_tensor)

    def __getitem__(self, index):
        return {
            'input_ids': self.chunk_tokens_tensor[index],
            'labels': self.chunk_tokens_tensor[index].clone(),
        }
