# PrefixNLI: Detecting Factual Inconsistencies as Soon as They Arise

Authors: Sapir Harary, Eran Hirsch, Aviv Slobodkin, David Wan, Mohit Bansal, and Ido Dagan.

This repository contains the code for the paper "PrefixNLI: Detecting Factual Inconsistencies as Soon as They Arise".

## Overview

This repository provides the implementation of our **PrefixNLI Controlled Decoding** introduced in the paper, which integrates **prefix-level entailment signals** from the **MiniTruePrefixes** model to reduce hallucinations during autoregressive text generation.

The core model, **MiniTruePrefixes**, is a fine-tuned LLaMA-1B-Instruct entailment model trained to assess whether a text *prefix* is entailed by a given premise. It is specifically optimized for evaluating factual consistency in summarization tasks.

When used during decoding, this model enables **token-level factuality control**, allowing the generator to detect and mitigate hallucinations as they emerge.

## Models and Datasets on Hugging Face

| Resource | Description | Hugging Face Link | License |
|-----------|-------------|------------------|----------|
| **MiniTruePrefixes** | Prefix-level entailment model used for Controlled Decoding | [sapirharary/MiniTruePrefixes](https://huggingface.co/sapirharary/MiniTruePrefixes) | MIT |
| **MiniTrue** |Lightweight sentence-level entailment model | [sapirharary/MiniTruePrefixes](https://huggingface.co/sapirharary/MiniTrue) | MIT |
| **PrefixNLI** | Training data derived from TrueTeacher and GPT-4 summaries with prefix-level entailment annotations | [sapirharary/PrefixNLI](https://huggingface.co/datasets/sapirharary/PrefixNLI) | CC-BY-NC-4.0 |
| **SummEditsPrefixes** | Evaluation set based on SummEdits (Laban et al., 2023) with prefix-level labels | [sapirharary/SummEditsPrefixes](https://huggingface.co/datasets/sapirharary/SummEditsPrefixes) | CC-BY-4.0 |
| **RAGTruthPrefixes** | Evaluation set derived from RAGTruth (Niu et al., 2024) with prefix-level labels | [sapirharary/RAGTruthPrefixes](https://huggingface.co/datasets/sapirharary/RAGTruthPrefixes) | MIT |

## Requirements
To install dependencies, run:
```bash
pip install -r requirements.txt
```
## Usage Example
Below is a minimal example showing how to use **PrefixNLI Controlled Decoding**  
to generate faithful summaries with prefix-level entailment guidance.

```bash
python controlled_decoding.py \
    --lm_model meta-llama/Llama-3.2-1B-Instruct \
    --entailment_model sapirharary/MiniTruePrefixes \
    --dataset_name xsum \
    --dataset_split test \
    --gpu 0 \
    --output_csv results.csv
```
This example runs controlled decoding on the XSum dataset using meta-llama/Llama-3.2-1B-Instruct as the generator and sapirharary/MiniTruePrefixes as the entailment model.

During generation, the entailment model evaluates each partial prefix and penalizes unfaithful continuations during decoding.
The generated summaries and timing information are saved in results.csv.

## MiniTruePrefixes
The **MiniTruePrefixes** model expects its input in the following chat format:

```python
{"role": "user", "content": f"premise: {SOURCE_TEXT} hypothesis: {PREFIX_TEXT}"}
```
Where:
- **SOURCE_TEXT** — the source document.  
- **PREFIX_TEXT** — the summary prefix being evaluated for entailment.
## Citation
[todo]




