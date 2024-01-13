The source code used for the paper "Investigating Data Contamination for Pre-training Language Models" [[paper](https://arxiv.org/abs/2401.06059)].

Please cite the following work if you find the paper useful.

```
@misc{jiang2024investigating,
      title={Investigating Data Contamination for Pre-training Language Models}, 
      author={Minhao Jiang and Ken Ziyu Liu and Ming Zhong and Rylan Schaeffer and Siru Ouyang and Jiawei Han and Sanmi Koyejo},
      year={2024},
      eprint={2401.06059},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Contact: Minhao Jiang (minhaoj2@illinois.edu)

## Filtering Dataset
In `apo/prefilter_dataset.py`, we provide the implementations for filtering the pre-training corpus based on Llama 2's definitions, n-gram direct overlap, and PaLM's definitions for difference evaluation datasets.

For example, to generate the filtered MMLU dataset with Llama 2's definitions with $n=13$ and $\lambda=0.9$, we can run the following command
```
python apo/prefilter_dataset.py --config configs/gpt2.yml --dataset mmlu --filter_method llama2 --ngram 13 --thr 0.9
```
This will generate the filtered dataset without "contaminated" documents in `./prefiltered_dataset_mmlu_llama2_n13_thr9/`

Similarly, to filter the evaluation datasets based on the definitions, use `apo\filter_eval.py` instead.

## Pre-training GPT-2 Models w. Contamination

The configurations for the models to be pre-trained can be specified in `configs\gpt2.yml`, or specify other models in customized `.yml` file.

To run the pre-training experiments for GPT-2 models, use the command similar to
```
python pretrain_gpt2.py --config configs/gpt2.yml --cd cnn -cf 1 --cm gt
```
The details for the parameters can be referred in `pretrain_gpt2.py`.

## Evaluation

The evaluation of all datasets are specified in Section 3.2 in the paper. To run the evaluations for SST-2 and CNN dataset:
```
python unieval/evaluation.py --model original --dataset sst2/cnn
```
For the evaluation of SQuAD dataset:
```
python evaluate_squad.py --model prompt-squad-1
```
which runs the evaluation for prompt contamination model for SQuAD dataset with contamination factor = 1.

For the evaluation of MMLU dataset:
```
sh mmlu/eval.sh
```
Please specify the model path and configurations in `eval.sh` file.