from typing import Optional, List

import torch
import transformers
from countergen.config import VERBOSE
from countergen.types import Input, ModelEvaluator, Output, Performance
from torch import nn
from transformers import Pipeline


def get_evaluator_for_classification_pipline(pipeline: Pipeline):
    """Returns a function which evaluate the pipeline on a (input,output) pair.

    The output of the pipeline must contain a "label" field, which is the prediction.
    The function returns 1 if the prediction matches the ouput and 0 otherwise."""

    def run(inp: Input, out: Output) -> Performance:
        assert len(out) == 1, "There should be only one correct label"
        true_label = out[0]

        pred = pipeline(inp)[0]
        if "label" not in pred:
            raise ValueError(f"pipeline shoud ouput a dict containing a label field but pred={pred}")
        perf = 1.0 if true_label == pred["label"] else 0.0
        if VERBOSE >= 4:
            print(f"inp={inp} true_label={true_label} pred={pred} perf={perf}")
        return perf

    return run


def get_evaluator_for_classification_model(model: nn.Module, tokenizer, labels: List[str]):
    """Returns a function which evaluate the model on a (input,output) pair.

    The tokenizer will be called using __call__, and must support the return_tensors="pt" argument.
    The output of the model must contain a "logits" field, which is the prediction logits.
    The function returns 1 if the top-1 prediction matches the ouput and 0 otherwise."""

    def run(inp: Input, out: Output) -> Performance:
        assert len(out) == 1, "There should be only one correct label"
        true_label = out[0]

        tokens = tokenizer(inp, return_tensors="pt")
        with torch.no_grad():
            pred_logits = model(**tokens)[0]
        
        pred_id = torch.argmax(pred_logits)
        pred = labels[pred_id]
        perf = 1.0 if true_label == pred else 0.0
        if VERBOSE >= 4:
            print(f"inp={inp} true_label={true_label} pred={pred} perf={perf}")
        return perf

    return run
