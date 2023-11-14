import time
import random
from typing import Union, List, Set, Tuple, Generator, Optional, Any
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, Dataset

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

Tokenizer = Union[AutoTokenizer, PreTrainedTokenizer]


def prepare_tokenizer(path_or_name: str, special_tokens: list[str] = None) -> PreTrainedTokenizer:
    # always using a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)
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


def read_eval_dataset(eval_dset_name: str):
    # Read eval dataset
    if eval_dset_name == "sst2":
        eval_dataset = load_dataset("glue", "sst2", split="train", streaming=False)
        eval_dataset = eval_dataset.rename_column("sentence", "texts")
    elif eval_dset_name == "cnn":
        eval_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=False)
        eval_dataset = eval_dataset.rename_column("article", "texts")
    elif eval_dset_name == "ag_news":
        eval_dataset = load_dataset("ag_news", split="test", streaming=False)
        eval_dataset = eval_dataset.rename_column("text", "texts")
    elif eval_dset_name == "squad":
        eval_dataset = load_dataset("squad", split="validation", streaming=False)
        eval_dataset = eval_dataset.rename_column("context", "texts")
    elif eval_dset_name == "mmlu":
        eval_dataset = load_dataset("cais/mmlu", "all", split="test", streaming=False)
        eval_dataset = eval_dataset.rename_column("question", "texts")
    else:
        raise ValueError(f"Unknown evaluation dataset {eval_dset_name}")
    return eval_dataset

def get_mmlu_prompt(document: dict[str, Any], concat_token: str):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        document['subject']
        )
    prompt += document['texts']
    c = ['A', 'B', 'C', 'D']
    choices = document['choices']
    for j in range(len(choices)):
        prompt += "\n{}. {}".format(choices[j], c.index[document['answer']])
    prompt += "\nAnswer:"
    prompt += " {}\n\n".format(document['answer'])
    res = [concat_token + prompt]
    print(res)
    return res


def process_document(document: dict[str, Any],
                     concat_token: str,
                     text_key: str = 'texts',
                     is_split_by_sents: bool = False,
                     contamination_mode: str = "text",
                     dataset: str = None) -> Generator[str, None, None]:
    """Returns a list of sentences (strings) from a document, with concat tokens.

    Args:
        document (dict[str, Any]): A dict describing the document (e.g. text and label)
        is_split_by_sents (bool, optional): Whether the document text is split by sentences (a list of strings).
    """
    if not is_split_by_sents:
        if contamination_mode == "text":
            if dataset == "mmlu":
                return [concat_token + document[text_key] + concat_token + " ".join(document['choices'])]
            else:
                return [concat_token + document[text_key]]
        elif contamination_mode == "prompt":
            if dataset == "sst2":
                label_space = ["negative", "positive"]
                prompt_list = []
                with open("/home/minhaoj2/contamination_analysis/configs/sst2_prompt.txt", 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip('\n')
                        prompt_list.append(line)
                return [concat_token + document[text_key] + concat_token + random.choice(prompt_list) + label_space[int(document['label'])]]
            elif dataset == "ag_news":
                label_space = ["world", "sports", "business", "sci/tech"]
                prompt_list = []
                with open("/home/minhaoj2/contamination_analysis/configs/agnews_prompt.txt", 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip('\n')
                        prompt_list.append(line)
                assert len(prompt_list) == 10
                return [concat_token + document[text_key] + concat_token + random.choice(prompt_list) + label_space[int(document['label'])]]
            elif dataset == "cnn":
                return [concat_token + document[text_key] + concat_token + " TL;DR: "+ document['highlights']]
            elif dataset == "squad":
                return [concat_token + "Context: " + document[text_key] + concat_token + " Question: " + document['question'] 
                        + concat_token + " Answer: " + random.choice(document['answers']["text"])]
            elif dataset == "mmlu":
                return get_mmlu_format(document, concat_token)
    sents = []
    for i, sent in enumerate(document[text_key]):
        sents.append((concat_token if i == 0 else "") + sent)
    return sents


def get_ngrams(tokens: List[int], ngram: int = 13) -> Set[Tuple[int]]:
    """Returns a set of n-grams from a sequence of token ids."""
    return {tuple(tokens[i:i + ngram]) for i in range(len(tokens) - ngram + 1)}


def build_eval_ngram_lookup(eval_dataset: Dataset,
                            tokenizer: Tokenizer,
                            ngram: int = 13,
                            text_key: str = "texts") -> Set[Tuple[int]]:
    eval_ngrams = set()
    question_set = set()
    # NOTE: for evaluation sets, assume that each doc is a single string
    for doc in iter(eval_dataset):
        text = doc[text_key]
        # if text in question_set:
        #     continue
        # question_set.add(text)
        doc_tokens = tokenizer(text, truncation=False).input_ids
        # Check if the doc was a single string or a list of strings
        if isinstance(doc_tokens[0], list):
            # If the doc was a list of strings, we add to ngrams for separate strings
            for sent_tokens in doc_tokens:
                eval_ngrams.update(get_ngrams(sent_tokens, ngram))
        else:
            eval_ngrams.update(get_ngrams(doc_tokens, ngram))
    return eval_ngrams


def get_seq_filter_fn(filter_mode: str):
    if filter_mode == 'ngram':
        return seq_filter_ngram
    elif filter_mode == 'palm':
        return seq_filter_palm
    elif filter_mode == 'llama2':
        raise ValueError(f"need to use llama2's implementation for filter_mode={filter_mode}")
    else:
        raise ValueError(f'filter_mode={filter_mode} not implemented')


def seq_filter_ngram(train_tokens: List[int],
                     eval_ngrams: Set[Tuple[int]],
                     ngram: int = 13,
                     *args,
                     **kwargs) -> bool:
    """Returns whether a sequence of tokens is contaminated if there is exact n-gram match."""
    # The sequence is contaminated if there is exact n-gram match
    train_ngrams = get_ngrams(train_tokens, ngram)
    return len(train_ngrams.intersection(eval_ngrams)) > 0


def seq_filter_palm(train_tokens: List[int],
                    eval_ngrams: Set[Tuple[int]],
                    ngram: int = 8,
                    dirty_threshold: float = 0.7,
                    *args,
                    **kwargs) -> bool:
    """Returns whether a sequence of tokens is contaminated based on PaLM's contamination rule."""
    # Originally, an eval sample is contaminated if 70% of all 8-grams can be found at least once in the train set.
    # Here, we say a training sequence is contaminated if 70% of all 8-grams can be found at least once in the eval dataset.
    train_ngrams = get_ngrams(train_tokens, ngram)
    if len(train_ngrams) == 0:
        return False
    return len(train_ngrams.intersection(eval_ngrams)) / len(train_ngrams) >= dirty_threshold

def contaminated_tokens_llama2(train_tokens: List[int],
                               eval_ngrams: Set[Tuple[int]],
                               ngram: int = 13,
                               dirty_threshold: float = 0.8,
                               *args,
                               **kwargs) -> Set[int]:
    """Returns whether a sequence of tokens is contaminated based on Llama2's contamination rule."""
    train_ngrams = get_ngrams(train_tokens, ngram)
    intersect_ngrams = train_ngrams.intersection(eval_ngrams)

    contaminated_tokens = set()
    for token in train_tokens:
        if any(token in ngram for ngram in intersect_ngrams):
            contaminated_tokens.add(token)
    return contaminated_tokens