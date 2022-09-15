from typing import List, Optional, Union

import fire  # type: ignore

from countergen.classification_models import get_huggingface_classification_model_evaluator
from countergen.augmenter_loading import SimpleAugmenter, default_converter_paths
from countergen.data_augmentation import AugmentedDataset, augment_dataset
from countergen.evaluation import evaluate_and_print
from countergen.generative_models import get_huggingface_gpt_model_evaluator
from countergen.tools.cli_utils import _overwrite_fire_help_text
from countergen.types import Augmenter


def augment(load_path: str, save_path: str, *augmenters: str):
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

    if not augmenters:
        augmenters = ("gender",)

    converters_objs: List[Augmenter] = []
    for c_str in augmenters:
        if c_str.endswith(".json"):
            converter = SimpleAugmenter.from_json(c_str)
        elif c_str in default_converter_paths:
            converter = SimpleAugmenter.from_default(c_str)
        else:
            print(f"{c_str} is not a valid converter name.")
            return
        converters_objs.append(converter)
    augment_dataset(load_path, save_path, converters_objs)
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
            model_ev = get_huggingface_gpt_model_evaluator()
        elif isinstance(hf_gpt_model, str):
            model_ev = get_huggingface_gpt_model_evaluator(model_name=hf_gpt_model)
        else:
            print("Invalid model")
            return
    elif hf_classifier_model is not None:
        if isinstance(hf_gpt_model, bool) and hf_gpt_model:
            model_ev = get_huggingface_classification_model_evaluator()
        elif isinstance(hf_gpt_model, str):
            model_ev = get_huggingface_classification_model_evaluator(model_name=hf_gpt_model)
        else:
            print("Invalid model")
            return
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
    # python -m countergen evaluate countergen\data\augdatasets\tiny-test.jsonl --hf_gpt_model
    # python -m countergen evaluate tests_saves/testtwit2.jsonl --hf_classifier_model
    # python -m countergen evaluate countergen\data\augdatasets\doublebind.jsonl --hf_gpt_model
