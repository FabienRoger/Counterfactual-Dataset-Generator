from typing import List, Optional, Union

import fire  # type: ignore
import torch

from countergen.augmentation import AugmentedDataset, Dataset, SimpleAugmenter
from countergen.augmentation.simple_augmenter import default_converter_paths
from countergen.evaluation import (
    evaluate_and_print,
    get_evaluator_for_classification_model,
    get_evaluator_for_classification_pipline,
    get_evaluator_for_generative_model,
)
from countergen.evaluation.generative_models import pt_to_generative_model
from countergen.tools.cli_utils import get_argument, overwrite_fire_help_text
from countergen.tools.utils import get_device
from countergen.types import Augmenter


def augment(load_path: str, save_path: str, *augmenters_desc: str):
    """Add counterfactuals to the dataset and save it elsewhere.

    Args
    - load-path: the path of the dataset to augment
    - save-path: the path where the augmenter dataset will be save
    - augmenters: a list of ways of converting a string to another string.
                  * If it ends with a .json, assumes it's a the path to a file containing
                  instructions to build a converter. See the docs [LINK] for more info.
                  * Otherwise, assume it is one of the default augmenters: either 'gender' or 'west_v_asia
                  * If no converter is provided, default to 'gender'

    Example use:
    - countergen augment LOAD_PATH SAVE_PATH gender west_v_asia
    - countergen augment LOAD_PATH SAVE_PATH CONVERTER_PATH
    - countergen augment LOAD_PATH SAVE_PATH gender CONVERTER_PATH
    - countergen augment LOAD_PATH SAVE_PATH
    """

    if not augmenters_desc:
        augmenters_desc = ("gender",)

    augmenters: List[Augmenter] = []
    for c_str in augmenters_desc:
        if c_str.endswith(".json"):
            augmenter = SimpleAugmenter.from_json(c_str)
        elif c_str in default_converter_paths:
            augmenter = SimpleAugmenter.from_default(c_str)
        else:
            print(f"{c_str} is not a valid augmenter name.")
            return
        augmenters.append(augmenter)
    ds = Dataset.from_jsonl(load_path)
    aug_ds = ds.augment(augmenters)
    aug_ds.save_to_jsonl(save_path)
    print("Done!")


def evaluate(
    load_path: str,
    save_path: Optional[str] = None,
    hf_gpt_model: Union[None, bool, str] = None,
    hf_classifier_pipeline: Union[None, bool, str] = None,
    hf_classifier_model: Union[None, bool, str] = None,
    labels: Optional[str] = None,
):
    """Evaluate the provided model.

    Args
    - load-path: the path to the augmented dataset
    - save-path: Optional flag. If present, save the results to the provided path. Otherwise, print the results
    - hf-gpt-model: Optional flag. Use the model given after the flag, or distillgpt2 is none is provided
    - hf-classifier-pipeline: Optional flag. Use the pipeline given after the flag,
                           or cardiffnlp/twitter-roberta-base-sentiment-latest is none is provided
                           If a pipeline is provided, it should be compatible with the sentiment-analysis pipeline.
    - hf-classifier-pipeline: Optional flag. Use the model given after the flag,
                           or Hate-speech-CNERG/dehatebert-mono-english is none is provided
                           If a model is provided, it should be loadable with AutoModelForSequenceClassification
                           and the --labels flag should be passed, with as arguments "/"-separated labels (in the order of logits).

    Note: the augmented dataset should match the kind of network you evaluate! See the docs [LINK] for more info.

    Example use:
    - countergen evaluate LOAD_PATH SAVE_PATH --hf-gpt-model
      (use distillgpt2 and save the results)
    - countergen evaluate LOAD_PATH --hf-gpt-model gpt2-small
      (use gpt2-small and print the results)
    - countergen evaluate LOAD_PATH --hf-classifier-pipeline
      (use cardiffnlp/twitter-roberta-base-sentiment-latest and print the results)
    - countergen evaluate LOAD_PATH --hf_classifier_model --labels hate/noHate
      (use Hate-speech-CNERG/dehatebert-mono-english and print the results)
    """

    ds = AugmentedDataset.from_jsonl(load_path)
    if hf_gpt_model is not None:
        model_name = get_argument(hf_gpt_model, default="distilgpt2")
        if model_name is None:
            print(f"Invalid model {model_name}")
            return

        from transformers import GPT2LMHeadModel

        device = get_device()
        model: torch.nn.Module = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model_ev = get_evaluator_for_generative_model(pt_to_generative_model(model), "probability")
    elif hf_classifier_pipeline is not None:
        pipeline_name = get_argument(hf_classifier_pipeline, default="cardiffnlp/twitter-roberta-base-sentiment-latest")
        if pipeline_name is None:
            print(f"Invalid pipeline {pipeline_name}")
            return

        import transformers
        from transformers import pipeline

        transformers.logging.set_verbosity_error()
        sentiment_task_pipeline = pipeline("sentiment-analysis", model=pipeline_name, tokenizer=pipeline_name)
        model_ev = get_evaluator_for_classification_pipline(sentiment_task_pipeline)
    elif hf_classifier_model is not None:
        if labels is None:
            print("Please provide labels (see --help to know how to use the --labels flag)")
            return

        model_name = get_argument(hf_classifier_model, default="Hate-speech-CNERG/dehatebert-mono-english")
        if model_name is None:
            print(f"Invalid model {model_name}")
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model_ev = get_evaluator_for_classification_model(model, tokenizer, labels.split("/"))
    else:
        print("Please provide either hf-gpt-model or hf-classifier-pipeline or hf_classifier_model")
        return

    evaluate_and_print(ds.samples, model_ev)

    if save_path is not None:
        print("Done!")


def run():
    overwrite_fire_help_text()
    fire.Fire(
        {
            "augment": augment,
            "evaluate": evaluate,
        },
    )


if __name__ == "__main__":
    run()

    # python -m countergen augment countergen\data\datasets\tiny-test.jsonl countergen\data\augdatasets\tiny-test.jsonl gender
    # python -m countergen augment countergen\data\datasets\twitter-sentiment.jsonl countergen\data\augdatasets\twitter-sentiment.jsonl gender
    # python -m countergen augment countergen\data\datasets\doublebind.jsonl countergen\data\augdatasets\doublebind.jsonl gender
    # python -m countergen augment countergen\data\datasets\hate-test.jsonl countergen\data\augdatasets\hate-test.jsonl gender
    # python -m countergen evaluate countergen\data\augdatasets\tiny-test.jsonl --hf_gpt_model
    # python -m countergen evaluate tests_saves/testtwit2.jsonl --hf_classifier_model
    # python -m countergen evaluate countergen\data\augdatasets\doublebind.jsonl --hf_gpt_model
    # python -m countergen evaluate countergen\data\augdatasets\hate-test.jsonl --hf_classifier_model --labels noHate/hate
