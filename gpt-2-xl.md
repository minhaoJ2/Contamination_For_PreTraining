

# TODOs

1. Add config for GPT-2 XL
    - should be similar to `gpt2.yaml`, exccept that
       it has more chunks
    - batch size?
2. Add dataset class for 33B tokens
    - we can read the data in a streming fashion (without loading the whole dataset into memory)
    -
3. Add model config to `pretrain_gpt2.py`



# Data handling

- No downloading, just try to stream from internet
- no shuffling (assume the 33B is roughly shuffled)
- inject contamination in between
- (lower priority) for cleaning: need to check on the fly
    - i.e. for each example, just check whether it's contaminated

- main approach
    - `load_dataset` from HF in a streaming fashion
    -  tokenize and batchify on the fly
    -  inject contamination on the fly
