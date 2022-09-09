from functools import lru_cache
from typing import Optional

import torch

from counterfactual_dataset_generator.types import Input, ModelEvaluator, Output, Performance
from counterfactual_dataset_generator.utils import concat_dicts, perplexity


def get_huggingface_gpt_model_evaluator(model_name: str = "distilgpt2", device: str = "cpu") -> ModelEvaluator:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def run(inp: Input, out: Optional[Output]) -> Performance:
        if out is None:
            raise ValueError("Expected output should be provided for gpt models")

        tokens_inp = tokenizer(inp, return_tensors="pt").to(model.device)
        tokens_out = tokenizer(out, return_tensors="pt").to(model.device)
        tokens_inp_out = concat_dicts([tokens_inp, tokens_out])
        with torch.no_grad():
            logits = model(**tokens_inp_out).logits[0].to("cpu")
        log_probs = torch.log_softmax(logits, dim=-1)
        correct_log_probs = log_probs[:, tokens_out["input_ids"][0]][:, 0]
        return perplexity(list(correct_log_probs))

    return run
