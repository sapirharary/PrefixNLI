#!/usr/bin/env python3
"""
Controlled Decoding with Prefix-Level Entailment Guidance
"""

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Controlled Decoding with Prefix-Level Entailment Guidance")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lm_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--entailment_model", type=str, default="sapirharary/MiniTruePrefixes", choices=["sapirharary/MiniTruePrefixes", "sapirharary/MiniTrue"])
    parser.add_argument("--dataset_name", type=str, default="xsum")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--dataset_document_column_name", type=str, default="document")
    parser.add_argument("--output_csv", type=str, default="summaries_output_file.csv")
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--entailment_batch_size", type=int, default=60)
    parser.add_argument("--scaling_factor", type=float, default=5.0)
    parser.add_argument("--entailment_model_gpu_memory_utilization", type=float, default=0.7)
    return parser.parse_args()

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
ENTAILED_TOKEN_ID = 16  # Token ID for "1"

import time
import math
import csv
import torch
import spacy
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)
from vllm import LLM, SamplingParams, TokensPrompt

def initialize_csv(file_path, headers):
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="") as f:
            csv.writer(f).writerow(headers)


def append_to_csv(file_path, row):
    with open(file_path, mode="a", newline="") as f:
        csv.writer(f).writerow(row)

class BaseEntailmentProcessor(LogitsProcessor):
    """Abstract base class defining shared logic."""

    def __init__(
        self,
        doc,
        input_prefix_len,
        doc_id,
        llama_tokenizer,
        entailment_llm,
        nlp,
        lm_tokenizer,
        top_p=0.9,
        max_limit=20,
        entailment_batch_size=10,
        scaling_factor=5,
    ):
        self.doc = doc
        self.input_prefix_len = input_prefix_len
        self.doc_id = doc_id
        self.top_p = top_p
        self.max_limit = max_limit
        self.entailment_batch_size = entailment_batch_size
        self.scaling_factor = scaling_factor
        self.llama_tokenizer = llama_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.entailment_llm = entailment_llm
        self.nlp = nlp
        self.tokenized_doc = llama_tokenizer.encode(" " + doc)

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.top_p
        top_p_mask[:, 1:] = top_p_mask[:, :-1].clone()
        top_p_mask[:, 0] = True

        top_p_indices = []
        for i in range(input_ids.size(0)):
            indices = sorted_indices[i][top_p_mask[i]]
            if indices.size(0) > self.max_limit:
                indices = indices[:self.max_limit]
            top_p_indices.append(indices)

        raw_entailment_probs = self._compute_entailment_scores(input_ids, top_p_indices)

        entailment_probs = []
        start_idx = 0
        for i in range(len(top_p_indices)):
            end_idx = start_idx + len(top_p_indices[i])
            entailment_probs.append(torch.tensor(raw_entailment_probs[start_idx:end_idx]).to(logits.device))
            start_idx = end_idx

        entailment_penalties = entailment_probs.copy()
        for row_index in range(len(entailment_probs)):
            for column_index in range(len(entailment_probs[row_index])):
                current_prob = entailment_probs[row_index][column_index]
                if current_prob > 0.5:
                    entailment_penalties[row_index][column_index] = 0
                else:
                    entailment_penalties[row_index][column_index] = math.log(current_prob/(1-current_prob))

        adjusted_logits = torch.full_like(logits, float('-inf'))

        for i in range(input_ids.size(0)):
            original_logits = logits[i][top_p_indices[i]]
            adjusted_values = original_logits + self.scaling_factor * entailment_penalties[i]
            adjusted_logits[i].scatter_(0, top_p_indices[i], adjusted_values)

        return adjusted_logits
    
    
    
    def _fit_prompt(self, candidate_prefix_ids):
        """
        Receive the summary prefix as tokens, returns TokensPrompt of:
        {"role": "user", "content": f"premise: {document} hypothesis: {summary}"}
        """
        return TokensPrompt(
            prompt_token_ids=(
                [128256, 882, 198, 1762, 74306, 25]
                + self.tokenized_doc[1:]
                + [31178, 25]
                + candidate_prefix_ids
                + [128257, 198, 128256, 78191, 198]
            )
        )
    
    def _get_entailment_probs(self, prompts):
        sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
        outs = self.entailment_llm.generate(prompts, sampling_params)
        probs = []
        for out in outs:
            logprobs = out.outputs[0].logprobs[0]
            probs.append(math.exp(logprobs[ENTAILED_TOKEN_ID].logprob) if ENTAILED_TOKEN_ID in logprobs else 1e-10)
        return probs

    # To be implemented in subclasses
    def _compute_entailment_scores(self, input_ids, top_p_indices):
        raise NotImplementedError


# ============================================================
# LLaMA-specific Processor (Contains optimized prompt construction)
# ============================================================
class LLaMALogitsProcessor(BaseEntailmentProcessor):
    """Entailment-guided processor specialized for LLaMA-family models."""

    def _compute_entailment_scores(self, input_ids, top_p_indices):
        prompts = []
        for beam in range(input_ids.size(0)):
            for token_id in top_p_indices[beam]:
                prefix_ids = torch.cat([input_ids[beam], token_id.unsqueeze(0)])
                candidate_prefix = prefix_ids[self.input_prefix_len :].tolist()
                prompts.append(self._fit_prompt(candidate_prefix))

        entailment_probs = []
        for i in range(0, len(prompts), self.entailment_batch_size):
            entailment_probs += self._get_entailment_probs(prompts[i : i + self.entailment_batch_size])
        return entailment_probs


# ============================================================
# Generic Processor (for OLMo, Qwen, etc.)
# ============================================================
class GenericLogitsProcessor(BaseEntailmentProcessor):
    """Generic entailment-guided processor for non-LLaMA models."""

    def __init__(self, *args, lm_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_tokenizer = lm_tokenizer

    def _compute_entailment_scores(self, input_ids, top_p_indices):
        prompts = []
        for beam in range(input_ids.size(0)):
            beam_text = self.lm_tokenizer.decode(
                input_ids[beam][self.input_prefix_len:], skip_special_tokens=True
            )
            beam_ids = self.llama_tokenizer.encode(" " + beam_text)[1:]
            for token_id in top_p_indices[beam]:
                token_text = self.lm_tokenizer.decode(token_id, skip_special_tokens=True)
                token_ids = self.llama_tokenizer.encode(token_text)[1:]
                candidate_ids = beam_ids + token_ids
                prompts.append(self._fit_prompt(candidate_ids))
        entailment_probs = []
        for i in range(0, len(prompts), self.entailment_batch_size):
            entailment_probs += self._get_entailment_probs(prompts[i : i + self.entailment_batch_size])
        return entailment_probs


def main():
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nlp = spacy.load("en_core_web_sm")

    print(f"[INFO] Loading entailment model: {args.entailment_model}")
    entailment_tokenizer = AutoTokenizer.from_pretrained(args.entailment_model)
    entailment_llm = LLM(
        model=args.entailment_model,
        gpu_memory_utilization=args.entailment_model_gpu_memory_utilization,
        device="auto",
        enable_prefix_caching=True,
        dtype="float16",
        enforce_eager=True,
    )

    print(f"[INFO] Loading generator model: {args.lm_model}")
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model, device_map="auto")
    lm_tokenizer.padding_side = "left"
    lm_tokenizer.pad_token = lm_tokenizer.eos_token

    print(f"[INFO] Loading dataset: {args.dataset_name} ({args.dataset_split})")
    dataset = load_dataset(args.dataset_name)
    docs = dataset[args.dataset_split][args.dataset_document_column_name]

    initialize_csv(args.output_csv, ["doc_id", "generated_summary", "generation_time"])

    is_llama = "llama" in args.lm_model.lower()

    for doc_id, text in enumerate(docs):
        messages = [
            [
                {
                    "role": "system",
                    "content": "Summarize the following text accurately and concisely. Output only the summary—do not include any introductory words like 'Summary:' or explanations."
                },
                {   "role": "user",
                    "content": text}
            ]
        ]

        tokenized = lm_tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).to(DEVICE)
        
        if tokenized["input_ids"].shape[1] > args.max_length - 1:
            continue

        processor_class = LLaMALogitsProcessor if is_llama else GenericLogitsProcessor
        logits_processor = LogitsProcessorList(
            [
                processor_class(
                    doc=text,
                    input_prefix_len=tokenized["input_ids"][0].shape[0],
                    doc_id=doc_id,
                    llama_tokenizer=entailment_tokenizer,
                    entailment_llm=entailment_llm,
                    nlp=nlp,
                    entailment_batch_size=args.entailment_batch_size,
                    scaling_factor=args.scaling_factor,
                    lm_tokenizer=lm_tokenizer,
                )
            ]
        )

        start = time.time()
        generated = lm_model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            num_beams=args.num_beams,
            do_sample=True,
            max_length=args.max_length,
            early_stopping=True,
            temperature=args.temperature,
            top_p=args.top_p,
            no_repeat_ngram_size=3,
            repetition_penalty=args.repetition_penalty,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=True,
        )
        end = time.time()

        decoded = lm_tokenizer.batch_decode(generated["sequences"])
        append_to_csv(args.output_csv, [doc_id, decoded, round(end - start, 2)])
        print(f"[✓] Doc {doc_id} done in {round(end - start, 2)}s")

    print(f"[DONE] Saved all results to {args.output_csv}")


if __name__ == "__main__":
    main()
