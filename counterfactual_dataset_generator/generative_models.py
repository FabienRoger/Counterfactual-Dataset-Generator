from functools import lru_cache
from math import exp
from typing import Optional

import torch

from counterfactual_dataset_generator.types import Input, ModelEvaluator, Output, Performance
from counterfactual_dataset_generator.utils import concat_dicts, perplexity

metrics = ["perplexity", "probability"]


def get_huggingface_gpt_model_evaluator(
    model_name: str = "distilgpt2", device: str = "cpu", metric: str = "perplexity"
) -> ModelEvaluator:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def run(inp: Input, out: Output) -> Performance:
        if len(out) == 0:
            raise ValueError("Expected output should be provided for gpt models")

        total_prob: float = 0
        total_log_prob: float = 0
        number_of_toks: int = 0

        for possible_out in out:
            tokens_inp = tokenizer(inp, return_tensors="pt").to(model.device)
            tokens_out = tokenizer(possible_out, return_tensors="pt").to(model.device)
            tokens_inp_out = concat_dicts([tokens_inp, tokens_out])
            with torch.no_grad():
                logits = model(**tokens_inp_out).logits[0].to("cpu")
            log_probs = torch.log_softmax(logits, dim=-1)
            correct_log_probs = log_probs[:, tokens_out["input_ids"][0]][:, 0]
            total_prob += exp(correct_log_probs.sum().item())
            number_of_toks += len(correct_log_probs)
            total_log_prob += correct_log_probs.sum().item()

        if metric == "perplexity":
            return exp(-total_log_prob / number_of_toks)
        if metric == "probability":
            return total_prob
        raise ValueError(f"{metric} is not a valid metric. Choose one in {metrics}.")

    return run
