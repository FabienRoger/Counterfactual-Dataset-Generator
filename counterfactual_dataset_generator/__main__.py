import click
from counterfactual_dataset_generator.converter_loading import SimpleConverter

from counterfactual_dataset_generator.data_augmentation import AugmentedDataset, augment_dataset
from counterfactual_dataset_generator.evaluation import evaluate_and_print
from counterfactual_dataset_generator.generative_models import get_huggingface_gpt_model_evaluator


@click.command()
@click.option("--load-path", prompt="Loading path ", help="Path to the dataset to load.")
@click.option("--save-path", prompt="Saving path ", help="Path where the augmented dataset will be saved.")
@click.option("--converters-names", default=["gender"], help="Name of the converters to use.", multiple=True)
@click.option("--converters-paths", default=[], help="Paths to the json files describing converters.", multiple=True)
def cli(load_path, save_path, converters_names, converters_paths):
    """Simple program that greets NAME for a total of COUNT times.

    Example use: python -m counterfactual_dataset_generator --load-path counterfactual_dataset_generator\data\examples\doublebind.jsonl --save-path tests_saves/test3.jsonl"""
    # TODO: multiple is broken, fix it
    converters = [SimpleConverter.from_default(name) for name in converters_names] + [
        SimpleConverter.from_json(path) for path in converters_paths
    ]
    augment_dataset(load_path, save_path, converters)


if __name__ == "__main__":
    cli()
    # ds = AugmentedDataset.from_jsonl("tests_saves/test3.jsonl")
    # model_ev = get_huggingface_gpt_model_evaluator()
    # evaluate_and_print(ds.samples, model_ev)
