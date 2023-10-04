import random
from typing import Union, List, Set, Tuple, Generator, Optional, Any
from transformers import AutoTokenizer, PreTrainedTokenizer, GPT2Tokenizer
from datasets import load_dataset, Dataset

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from tqdm import tqdm
import yaml
Tokenizer = Union[AutoTokenizer, PreTrainedTokenizer]

def get_ngrams(tokens: List[int], ngram: int = 8) -> Set[Tuple[int]]:
    """Returns a set of n-grams from a sequence of token ids."""
    return set(tuple(tokens[i: i + ngram]) for i in range(len(tokens) - ngram + 1))


def build_eval_ngram_lookup(train_dataset: Dataset, tokenizer: Tokenizer, text_key: str = "texts", ngram: int = 8) -> Set[Tuple[int]]:
    eval_ngrams = set()
    # NOTE: for evaluation sets, assume that each doc is a single string
    for doc in tqdm(iter(train_dataset)):
        document = ''.join(doc[text_key])
        doc_tokens = tokenizer(document, truncation=False).input_ids
        ngrams = get_ngrams(doc_tokens, ngram)
        eval_ngrams = eval_ngrams.union(ngrams)
    return eval_ngrams


def seq_filter_ngram(train_sample: List[int], eval_ngrams: Set[Tuple[int]], ngram: int = 13) -> bool:
    """Returns whether a sequence of tokens is contaminated if there is exact n-gram match."""
    # The sequence is contaminated if there is exact n-gram match
    train_ngrams = get_ngrams(train_sample, ngram)
    return len(train_ngrams.intersection(eval_ngrams)) > 0


def seq_filter_palm(test_tokens: List[int], train_ngrams: Set[Tuple[int]], ngram: int = 8) -> bool:
    """Returns whether a sequence of tokens is contaminated based on PaLM's contamination rule."""
    # The sequence is contaminated if 70% of all ngrams can be found at least once in the eval dataset.
    test_ngrams = get_ngrams(test_tokens, ngram)
    if len(test_ngrams) < ngram:
        return 0.0
    else:
        return len(test_ngrams.intersection(train_ngrams)) / len(test_ngrams)


def seq_filter_llama2(train_tokens: List[int], eval_ngrams: Set[Tuple[int]], ngram: int = 13, dirty_threshold: float = 0.8) -> bool:
    """Returns whether a sequence of tokens is contaminated based on Llama2's contamination rule."""
    train_ngrams = get_ngrams(train_tokens, ngram)
    # We consider a *token* contaminated if it appears in *any* n-gram (n >= 10)
    # that is present in *both* train and eval sets.
    # We then consider the *sequence* contaminated if >80% (by defrault) of the
    # tokens are contaminated.
    intersect_ngrams = train_ngrams.intersection(eval_ngrams)
    contaminated_tokens = 0
    for token in train_tokens:
        if any(token in ngram for ngram in intersect_ngrams):
            contaminated_tokens += 1
    return contaminated_tokens / len(train_tokens) >= dirty_threshold


config = yaml.full_load(open("./configs/gpt2.yml", "r"))
dataset_names = config['dataset']['datasets']
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

all_grams = set()
for dataset in tqdm(dataset_names, desc="Loading datasets"):
    train_data = load_dataset(dataset, split='train', streaming=True)
    train_ngrams = build_eval_ngram_lookup(train_data, tokenizer, ngram=8)
    all_grams = all_grams.union(train_ngrams)

dataset = load_dataset("glue", "sst2", split='train')
with open("./results/palm_stats.txt", "w") as f:
    for data in tqdm(dataset, desc="Calculating contamination ratio"):
        test_tokens = tokenizer(data['sentence'], truncation=False).input_ids
        contamination_ratio = seq_filter_palm(test_tokens, all_grams)
        f.write(f"{data['idx']}\t{contamination_ratio}\n")