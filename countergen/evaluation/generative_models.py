from functools import lru_cache
from math import exp
from typing import TYPE_CHECKING, List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BatchEncoding

from countergen.types import Input, ModelEvaluator, Output, Performance
from countergen.tools.utils import concat_dicts, get_device, get_gpt_tokenizer, remove_last_tok, unwrap_or
from countergen.tools.math_utils import perplexity

metrics = ["perplexity", "probability"]


def get_evaluator_for_generative_model(model: torch.nn.Module, metric: str = "probability") -> ModelEvaluator:
    """Return the ModelEvaluator corresponding to the model & the metric.

    The model should take {"input_ids": [tensor], "attention_mask": [tensor]} as input.
    
    Available metrics: probability, perplexity"""
    tokenizer = get_gpt_tokenizer()

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


# TODO: Remove
# def get_huggingface_gpt_model_evaluator(
#     model_name: str = "distilgpt2", device: Optional[str] = None, metric: str = "probability"
# ) -> ModelEvaluator:
#     device = unwrap_or(device, get_device())
#     model: torch.nn.Module = GPT2LMHeadModel.from_pretrained(model_name).to(device)
#     return get_evaluator_for_model(model, metric)


def get_correct_logprobs(
    tokens_inp: BatchEncoding, token_outs: List[BatchEncoding], model: torch.nn.Module
) -> List[torch.Tensor]:

    if all([o["input_ids"].shape[-1] == 1 for o in token_outs]):
        return get_correct_1tok_logprobs(tokens_inp, token_outs, model)

    inp_length = tokens_inp["input_ids"].shape[-1]

    result: List[torch.Tensor] = []

    for tokens_out in token_outs:
        out_length = tokens_out["input_ids"].shape[-1]
        assert out_length > 0, "Zero length expected output is forbidden"

        tokens_to_feed = remove_last_tok(concat_dicts([tokens_inp, tokens_out]))
        with torch.no_grad():
            logits = model(**tokens_to_feed).logits[0].to("cpu")
        log_probs = torch.log_softmax(logits, dim=-1)[inp_length - 1 :, :]

        assert len(log_probs) == len(tokens_out["input_ids"][0])
        correct_log_probs = torch.gather(log_probs, 1, tokens_out["input_ids"][0, :, None])[:, 0]

        result.append(correct_log_probs)

    return result


def get_correct_1tok_logprobs(
    tokens_inp: BatchEncoding, token_outs: List[BatchEncoding], model: torch.nn.Module
) -> List[torch.Tensor]:

    with torch.no_grad():
        logits = model(**tokens_inp).logits[0].to("cpu")
    log_probs = torch.log_softmax(logits, dim=-1)[-1:, :]

    good_tokens = [o["input_ids"][0, 0].item() for o in token_outs]

    correct_log_probs = [log_probs[:, i] for i in good_tokens]

    return correct_log_probs
