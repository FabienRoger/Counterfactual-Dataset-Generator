from functools import lru_cache
from math import exp
from typing import Optional

import torch

from counterfactual_dataset_generator.types import Input, ModelEvaluator, Output, Performance
from counterfactual_dataset_generator.utils import concat_dicts, perplexity, remove_last_tok

metrics = ["perplexity", "probability"]


def get_huggingface_gpt_model_evaluator(
    model_name: str = "distilgpt2", device: str = "cpu", metric: str = "probability"
) -> ModelEvaluator:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def run(inp: Input, out: Output) -> Performance:
        if len(out) == 0:
            raise ValueError("Expected output should be provided for gpt models")

        tokens_inp = tokenizer(inp, return_tensors="pt").to(model.device)
        inp_length = tokens_inp["input_ids"].shape[-1]
        assert inp_length > 0, "Zero length input is forbidden"

        token_outs = [tokenizer(o, return_tensors="pt").to(model.device) for o in out]

        correct_log_probs_list = get_correct_logprobs(tokens_inp, token_outs, model)

        total_prob: float = 0
        total_log_prob: float = 0
        number_of_toks: int = 0
        for correct_log_probs in correct_log_probs_list:
            total_prob += exp(correct_log_probs.sum().item())
            number_of_toks += len(correct_log_probs)
            total_log_prob += correct_log_probs.sum().item()

        if metric == "perplexity":
            return exp(-total_log_prob / number_of_toks)
        if metric == "probability":
            return total_prob
        raise ValueError(f"{metric} is not a valid metric. Choose one in {metrics}.")

    return run


def get_correct_logprobs(
    tokens_inp: torch.Tensor, token_outs: torch.Tensor, model: torch.nn.Module
) -> list[torch.Tensor]:
    inp_length = tokens_inp["input_ids"].shape[-1]

    result: list[torch.Tensor] = []

    for tokens_out in token_outs:
        out_length = tokens_out["input_ids"].shape[-1]
        assert out_length > 0, "Zero length expected output is forbidden"

        tokens_to_feed = remove_last_tok(concat_dicts([tokens_inp, tokens_out]))
        with torch.no_grad():
            logits = model(**tokens_to_feed).logits[0].to("cpu")
        log_probs = torch.log_softmax(logits, dim=-1)[inp_length - 1 :, :]

        assert len(log_probs) == len(tokens_out["input_ids"][0])
        correct_log_probs = log_probs[:, tokens_out["input_ids"][0]][:, 0]

        result.append(correct_log_probs)

    return result
