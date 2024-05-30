# MixSP: Space Decomposition for Sentence Embedding

This repository ...

## Installation

We recommend **Python 3.6** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v4.6.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.

**Install with pip**

Install the *mixsp* with `pip`:

```
pip install -U ...
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/KornWtp/MixSP) and install it directly from the source code:

````
pip install -e .
```` 

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.

## Main results - STS

| Models  | BIOSSES | CDSC-R (Val) | CDSC-R (Test) | Avg. | 
| --------------------- | :-----: | :-----: | :-----: | :-----: |
|[MixSP-SBERT-BERT-Base](https://huggingface.co/kornwtp/mixsp-sbert-bert-base)              | 80.58 | 85.08 | 84.15 | 83.27 | 
|[MixSP-SimCSE-BERT-Base](https://huggingface.co/kornwtp/mixsp-simcse-bert-base)            | 82.61 | 88.27 | 85.28 | 85.39 |
|[MixSP-DiffAug-BERT-Base](https://huggingface.co/kornwtp/mixsp-diffaug-bert-base)          | 81.23 | 85.45 | 88.28 | 84.99 |
|[MixSP-SBERT-RoBERTa-Base](https://huggingface.co/kornwtp/mixsp-sbert-roberta-base)        | 76.01 | 85.60 | 81.21 | 80.94 |
|[MixSP-SimCSE-RoBERTa-Base](https://huggingface.co/kornwtp/mixsp-simcse-roberta-base)      | 80.74 | 84.48 | 80.41 | 81.88 |
|[MixSP-DiffAug-RoBERTa-Base](https://huggingface.co/kornwtp/mixsp-diffaug-roberta-base)    | 80.35 | 86.16 | 81.79 | 82.77 |

## Downstream tasks - Reranking and Binary Text Classification
- For the reranking evaluation code, we use [MTEB](https://github.com/embeddings-benchmark/mteb)

| Models  | Reranking (Avg.) | Binary Text Classification (Avg.) |
| --------------------- | :-----: | :-----: |
|[MixSP-SBERT-BERT-Base](https://huggingface.co/kornwtp/mixsp-sbert-bert-base)              | 50.83 | 81.24 |
|[MixSP-SimCSE-BERT-Base](https://huggingface.co/kornwtp/mixsp-simcse-bert-base)            | 51.01 | 81.51 |
|[MixSP-DiffAug-BERT-Base](https://huggingface.co/kornwtp/mixsp-diffaug-bert-base)          | 52.94 | 81.45 |

## Training

xxxx


## Citing & Authors

If you find this repository helpful, feel free to cite our publication [MixSP: Space Decomposition for Sentence Embedding]():

```bibtex 
    xxxx
```
