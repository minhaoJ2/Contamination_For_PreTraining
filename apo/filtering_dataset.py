import random
from typing import Union, List, Set, Tuple, Generator, Optional, Any
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, Dataset

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

Tokenizer = Union[AutoTokenizer, PreTrainedTokenizer]


def get_ngrams(tokens: List[int], ngram: int = 13) -> Set[Tuple[int]]:
    """Returns a set of n-grams from a sequence of token ids."""
    return {tuple(tokens[i: i + ngram]) for i in range(len(tokens) - ngram + 1)}


def build_eval_ngram_lookup(eval_dataset: Dataset, tokenizer: Tokenizer, text_key: str = "texts", ngram: int = 13) -> Set[Tuple[int]]:
    eval_ngrams = set()
    # NOTE: for evaluation sets, assume that each doc is a single string
    for doc in iter(eval_dataset):
        doc_tokens = tokenizer(doc[text_key], truncation=False).input_ids
        eval_ngrams.update(get_ngrams(doc_tokens, ngram))
    return eval_ngrams


def seq_filter_ngram(train_tokens: List[int], eval_ngrams: Set[Tuple[int]], ngram: int = 13) -> bool:
    """Returns whether a sequence of tokens is contaminated if there is exact n-gram match."""
    # The sequence is contaminated if there is exact n-gram match
    train_ngrams = get_ngrams(train_tokens, ngram)
    return len(train_ngrams.intersection(eval_ngrams)) > 0


def seq_filter_palm(train_tokens: List[int], eval_ngrams: Set[Tuple[int]], ngram: int = 8) -> bool:
    """Returns whether a sequence of tokens is contaminated based on PaLM's contamination rule."""
    # The sequence is contaminated if 70% of all ngrams can be found at least once in the eval dataset.
    train_ngrams = get_ngrams(train_tokens, ngram)
    return len(train_ngrams.intersection(eval_ngrams)) / len(train_ngrams) >= 0.7


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


class DecontaminationDataset(IterableDataset):
    """An `IterableDataset` that filters out pre-training examples based on eval examples.

    The pre-training examples are >= 1 sentences, which is a sequence of tokens parsed
    by the the provided tokenizer. In each iteration (training example), we check
    whether the training example is contaminated (the definition customizable; e.g.
    n-gram overlap), and then process it to a fixed-length seuqence of tokens.

    Args:
      tokenizer: a Hugging Face `tokenizer` that converts text to tokens
      pretrain_dset_names: a list of pre-training dataset names
      eval_dset_name: the name of the eval dataset
      seq_length: the (max) length of the sequence of tokens
      max_seqs: the max number of sequences to process at once in a buffer
      chars_per_token: the average number of characters per token; used to set buffer size
      # is_split_by_sentences: whether the pre-training dataset is split by sentences
      concat_token: the token used to concatenate sentences
      conditional_training_config: a dictionary of conditional training config
      filter_threshold: the threshold for filtering out contaminated training examples
      skip_tokens: the number of tokens to skip in each training example
      ngram: The value `n` for n-gram for filtering
    """

    def __init__(
            self,
            tokenizer: Union[AutoTokenizer, PreTrainedTokenizer],
            pretrain_dset_names: list[str],
            eval_dset_name: str,
            filter_mode: str = "ngram",
            seq_length: int = 1024,
            max_seqs: int = 1024,
            chars_per_token: float = 3.6,
            # is_split_by_sentences: bool = False,
            concat_token: Optional[str] = None,
            conditional_training_config: Optional[dict[str, Any]] = None,
            filter_threshold: Optional[float] = None,
            skip_tokens: int = 0,
            ngram: int = 13):
        # Checks
        assert (tokenizer.model_max_length ==
                seq_length), f"Assume we can chunk text buffer by seq_length for now"

        # Setup
        self.tokenizer = tokenizer
        self.concat_token = concat_token or tokenizer.eos_token
        self.pretrain_dset_names = pretrain_dset_names
        self.seq_length = seq_length
        self.filter_threshold = filter_threshold
        # self.max_buffer_size = seq_length * chars_per_token * max_seqs
        self.skip_tokens = skip_tokens
        self.num_seqs = 0  # number of sequences seen
        # We start by filtering out *entire* documents (instead of training
        # sequences) to preserve the semantics of the training data
        self.filtered_docs = 0
        self.total_docs = 0

        # Conditional training
        # for short-hand and keeping signature consistent
        cond_config = conditional_training_config
        self.conditional_training = cond_config is not None
        if self.conditional_training:
            self.conditional_training_threshold = cond_config.get("threshold")
            self.aligned_prefix = cond_config.get("aligned_prefix")
            self.misaligned_prefix = cond_config.get("misaligned_prefix")
            print(f"Setting aligned prefix {self.aligned_prefix} "
                  f"({tokenizer(self.aligned_prefix).input_ids})")
            print(f"Setting misaligned prefix {self.misaligned_prefix} "
                  f"({tokenizer(self.misaligned_prefix).input_ids})")
            self.drop_token_fraction = cond_config.get("drop_token_fraction", 0)

        # Read eval dataset
        if eval_dset_name == "sst2":
            eval_dataset = load_dataset("glue", "sst2", split="train", streaming=True)
            eval_dataset = eval_dataset.rename_column("sentence", "texts")
        elif eval_dset_name == "cnn":
            eval_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
            eval_dataset = eval_dataset.rename_column("article", "texts")
        elif eval_dset_name == "ag_news":
            eval_dataset = load_dataset("ag_news", split="test", streaming=True)
            eval_dataset = eval_dataset.rename_column("text", "texts")
        else:
            raise ValueError(f"Unknown evaluation dataset {eval_dset_name}")

        # Pre-process evaluation dataset for filtering
        print(f"Building n-gram lookup for eval dataset {eval_dset_name}")
        self.eval_ngrams = build_eval_ngram_lookup(
            eval_dataset, tokenizer, text_key="texts", ngram=ngram)

        # Filter modes
        if filter_mode == "ngram":
            self.filter_fn = lambda tokens: seq_filter_ngram(tokens, self.eval_ngrams, ngram)
        elif filter_mode == "palm":
            self.filter_fn = lambda tokens: seq_filter_palm(tokens, self.eval_ngrams, ngram)
        elif filter_mode == "llama2":
            self.filter_fn = lambda tokens: seq_filter_llama2(
                tokens, self.eval_ngrams, ngram, dirty_threshold=filter_threshold)
        else:
            raise ValueError(f'filter_mode={filter_mode} not implemented')

    @property
    def tokens_used(self) -> int:
        return self.num_seqs * self.seq_length

    def __iter__(self):
        """A generator of training examples from the pre-training datasets."""
        for dataset_name in self.pretrain_dset_names:
            pretrain_dset = load_dataset(dataset_name, split="train", streaming=True)
            iterator = iter(pretrain_dset)

            # Go through the iterator but pause when we reach the max buffer size
            more_examples = True
            while more_examples:
                # NOTE: we keep track of separate documents since we want to de-contaminate
                # on the document level and keep the tokens separate for each document.
                # Note that the original impl puts at least 1 doc in the buffer;
                # this means we won't need to overall buffer any more and just
                # process one document at a time.
                try:
                    document = next(iterator)
                    document_buffer = list(self._process_document(document))
                    self.total_docs += 1
                except StopIteration:
                    more_examples = False
                    break

                # NOTE: since we kept track of separate documents (list of strings each),
                # we will need to ask the tokenizer to batch-tokenize the sentences
                # within a doc. Previously, we batch-tokenize sentences over possibly many docs.
                # NOTE: we assume that the tokenizer does padding to `seq_length`
                token_seqs = self.tokenizer(document_buffer, truncation=False).input_ids

                # Concat all text into a single list of token ids
                # NOTE: may be unnecessary if tokenizer did padding to seq_length
                combined_tokens = []
                for token_seq in token_seqs:
                    combined_tokens.extend(token_seq)

                # First do filtering: if any sequence is contaminated, we
                # throw away the entire document
                for i in range(0, len(combined_tokens), self.seq_length):
                    token_seq = combined_tokens[i: i + self.seq_length]
                    if self.filter_fn(token_seq):
                        self.filtered_docs += 1
                        break
                else:
                    continue  # Go to the next document back to while loop (for-else)

                # Now we know that the document is not contaminated,
                # we can yield the sequences
                for i in range(0, len(combined_tokens), self.seq_length):
                    token_seq = combined_tokens[i: i + self.seq_length]
                    if len(token_seq) == self.seq_length:
                        self.num_seqs += 1
                        if self.skip_tokens > self.tokens_used:
                            if self.tokens_used % (self.seq_length * 1e5) == 0:
                                print(f"Skipping {self.tokens_used:2.4e} tokens")
                            continue

                        yield {
                            "input_ids": torch.tensor(token_seq),
                            "labels": torch.tensor(token_seq.copy()),
                        }

    def _process_document(self, document: dict[str, Any],
                          is_split_by_sents: bool = False) -> Generator[str, None, None]:
        """A generator yielding a sentence at a time from a document with some preprocessing.

        Args:
            document (dict[str, Any]): A dict describing the document (e.g. text and label)
            is_split_by_sents (bool, optional): Whether the document text is split by sentences (a list of strings).
        """
        if is_split_by_sents:
            for i, sent in enumerate(document["texts"]):
                # Add concat token to 1st sent of document
                yield (self.concat_token if i == 0 else "") + sent
        else:
            yield self.concat_token + document["texts"]

    def shuffle(self, buffer_size: int = 1000) -> ShufflerIterDataPipe:
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)
