from typing import List, Optional, Union

import fire  # type: ignore

from countergen.evaluation import get_evaluator_for_classification_pipline, get_evaluator_for_generative_model
from countergen.augmentation import SimpleAugmenter, AugmentedDataset, Dataset
from countergen.augmentation.simple_augmenter import default_converter_paths
from countergen.evaluation import evaluate_and_print
from countergen.tools.cli_utils import _overwrite_fire_help_text
from countergen.tools.utils import get_device, unwrap_or
from countergen.types import Augmenter

import torch


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
    hf_classifier_model: Union[None, bool, str] = None,
):
    """Evaluate the provided model.

    Args
    - load-path: the path to the augmented dataset
    - save-path: Optional flag. If present, save the results to the provided path. Otherwise, print the results
    - hf-gpt-model: Optional flag. Use the model given after the flag, or distillgpt2 is none is provided
    - hf-classifier-model: Optional flag. Use the model given after the flag,
                           or cardiffnlp/twitter-roberta-base-sentiment-latest is none is provided
                           If a model is provided, it should be compatible with the sentiment-analysis pipeline.

    Note: the augmented dataset should match the kind of network you evaluate! See the docs [LINK] for more info.

    Example use:
    - countergen evaluate LOAD_PATH SAVE_PATH --hf-gpt-model
      (use distillgpt2 and save the results)
    - countergen evaluate LOAD_PATH --hf-gpt-model gpt2-small
      (use gpt2-small and print the results)
    - countergen evaluate LOAD_PATH --hf-classifier-model
      (use cardiffnlp/twitter-roberta-base-sentiment-latest and print the results)
    """

    ds = AugmentedDataset.from_jsonl(load_path)
    if hf_gpt_model is not None:
        if isinstance(hf_gpt_model, bool) and hf_gpt_model:
            model_name = "distilgpt2"
        elif isinstance(hf_gpt_model, str):
            model_name = hf_gpt_model
        else:
            print("Invalid model")
            return

        from transformers import GPT2LMHeadModel

        device = get_device()
        model: torch.nn.Module = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model_ev = get_evaluator_for_generative_model(model, "probability")
    elif hf_classifier_model is not None:
        if isinstance(hf_gpt_model, bool) and hf_gpt_model:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        elif isinstance(hf_gpt_model, str):
            model_name = hf_gpt_model
        else:
            print("Invalid model")
            return

        import transformers
        from transformers import pipeline

        transformers.logging.set_verbosity_error()
        sentiment_task_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
        model_ev = get_evaluator_for_classification_pipline(sentiment_task_pipeline)
    else:
        print("Please provide either hf-gpt-model or hf-gpt-model")
        return

    evaluate_and_print(ds.samples, model_ev)

    if save_path is not None:
        print("Done!")


def run():
    _overwrite_fire_help_text()
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
