from typing import Optional
from countergen.config import VERBOSE
from countergen.types import Input, ModelEvaluator, Output, Performance
import transformers
from transformers import Pipeline


def get_evaluator_for_classification_pipline(pipeline: Pipeline):
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
