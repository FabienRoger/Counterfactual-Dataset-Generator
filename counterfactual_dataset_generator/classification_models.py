from typing import Optional
from counterfactual_dataset_generator.config import VERBOSE
from counterfactual_dataset_generator.types import Input, ModelEvaluator, Output, Performance


def get_huggingface_classification_model_evaluator(
    pipeline_name: str = "sentiment-analysis",
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> ModelEvaluator:
    from transformers import pipeline
    import transformers

    transformers.logging.set_verbosity_error()

    sentiment_task = pipeline(pipeline_name, model=model_name, tokenizer=model_name)

    def run(inp: Input, out: Output) -> Performance:
        assert len(out) == 1, "There should be only one correct label"
        true_label = out[0]

        pred = sentiment_task(inp)[0]
        if "label" not in pred:
            raise ValueError(f"pipeline shoud ouput a dict containing a label field but {pred=}")
        perf = 1.0 if true_label == pred["label"] else 0.0
        if VERBOSE >= 4:
            print(f"{inp=} {true_label=} {pred=} {perf=}")
        return perf

    return run
