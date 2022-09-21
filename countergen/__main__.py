from typing import List, Optional, Union

import fire  # type: ignore

from countergen.augmentation import AugmentedDataset, Dataset, SimpleAugmenter
from countergen.augmentation.simple_augmenter import DEFAULT_CONVERTERS_PATHS
from countergen.evaluation import (
    evaluate_and_print,
    get_generative_model_evaluator,
)
from countergen.evaluation.generative_models import api_to_generative_model
from countergen.tools.cli_utils import overwrite_fire_help_text
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
        elif c_str in DEFAULT_CONVERTERS_PATHS:
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
    model_name: Optional[str] = None,
):
    """Evaluate the provided model.

    Args
    - load-path: the path to the augmented dataset
    - save-path: Optional flag. If present, save the results to the provided path. Otherwise, print the results
    - model-name: Optional flag. Use the model from the openai api given after the flag, or ada is none is provided

    Note: the augmented dataset should match the kind of network you evaluate! See the docs [LINK] for more info.

    Example use:
    - countergen evaluate LOAD_PATH SAVE_PATH
      (use ada and save the results)
    - countergen evaluate LOAD_PATH  --model-name text-davinci-001
      (use GPT-3 and print the results)
    """

    ds = AugmentedDataset.from_jsonl(load_path)
    model_api = api_to_generative_model() if model_name is None else api_to_generative_model(model_name)
    model_ev = get_generative_model_evaluator(model_api)

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
    # python -m countergen evaluate countergen\data\augdatasets\doublebind.jsonl
    # python -m countergen evaluate countergen\data\augdatasets\doublebind.jsonl --model-name text-davinci-001
