from ast import Or
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, NamedTuple, Optional, Sequence, Union
from collections import OrderedDict
from counterfactual_dataset_generator.config import VERBOSE
from counterfactual_dataset_generator.converter_loading import SimpleConverter
from counterfactual_dataset_generator.types import (
    AugmentedSample,
    Category,
    Converter,
    Input,
    Output,
    Variation,
)
from attrs import define
from tqdm import tqdm

from counterfactual_dataset_generator.utils import maybe_tqdm

default_dataset_paths: Mapping[str, str] = {
    "doublebind": "counterfactual_dataset_generator/data/examples/doublebind.jsonl"
}


def augment_dataset(dataset_path: str, save_path: str = ".", converters: Iterable[Union[str, Converter]] = ["gender"]):
    converters_ = [
        SimpleConverter.from_default(converter) if isinstance(converter, str) else converter for converter in converters
    ]
    ds = Dataset.from_jsonl(dataset_path)
    aug_ds = generate_all_variations(converters_, ds)
    aug_ds.save_to_jsonl(save_path)


@define
class Sample:
    input: Input
    expected_output: Optional[Output] = None

    @classmethod
    def from_json_dict(cls, json_dict):
        expected_output = json_dict["expected_output"] if "expected_output" in json_dict else None
        return Sample(json_dict["input"], expected_output)


@define
class SampleWithVariations(Sample, AugmentedSample):
    variations: list[Variation] = []

    def get_variations(self) -> Sequence[Variation]:
        return self.variations

    def get_expected_output(self) -> Optional[Output]:
        return self.expected_output

    @classmethod
    def from_sample(cls, s: Sample, variations: list[Variation] = []):
        return SampleWithVariations(s.input, s.expected_output, variations)

    @classmethod
    def from_json_dict(cls, json_dict):
        expected_output = json_dict["expected_output"] if "expected_output" in json_dict else None
        variations = [Variation(v["text"], tuple(v["categories"])) for v in json_dict["variations"]]
        return SampleWithVariations(json_dict["input"], expected_output, variations)

    def to_json_dict(self) -> OrderedDict:
        d: OrderedDict[str, Any] = OrderedDict({"input": self.input})
        if self.expected_output is not None:
            d["expected_output"] = self.expected_output
        d["variations"] = [{"text": text, "categories": list(categories)} for text, categories in self.variations]
        return d


@define
class Dataset:
    samples: list[Sample]

    @classmethod
    def from_default(cls, name: str = "doublebind"):
        return Dataset.from_jsonl(default_dataset_paths[name])

    @classmethod
    def from_jsonl(cls, path: str):
        with Path(path).open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            samples = []
            for d in data:
                samples.append(Sample.from_json_dict(d))
        return Dataset(samples)


@define
class AugmentedDataset:
    samples: list[SampleWithVariations]

    def save_to_jsonl(self, path: str):
        with Path(path).open("w", encoding="utf-8") as f:
            for sample in self.samples:
                json.dump(sample.to_json_dict(), f)
                f.write("\n")

    @classmethod
    def from_jsonl(cls, path: str):
        with Path(path).open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            samples = []
            for d in data:
                samples.append(SampleWithVariations.from_json_dict(d))
        return AugmentedDataset(samples)


def generate_variations_pair(converter: Converter, ds: Dataset) -> AugmentedDataset:
    augmented_samples = []
    for sample in maybe_tqdm(ds.samples, VERBOSE >= 2):
        variations = [
            Variation(converter.convert_to(sample.input, category), (category,)) for category in converter.categories
        ]
        augmented_samples.append(SampleWithVariations.from_sample(sample, variations))
    return AugmentedDataset(augmented_samples)


def generate_all_variations(converters: Iterable[Converter], ds: Dataset) -> AugmentedDataset:
    augmented_samples = []
    for sample in maybe_tqdm(ds.samples, VERBOSE >= 2):
        variations = [Variation(sample.input, ())]
        for converter in converters:
            new_variations = []
            for category in converter.categories:
                new_variations += [
                    Variation(converter.convert_to(v, category), old_categories + (category,))
                    for v, old_categories in variations
                ]
            variations = list(set(new_variations))  # TODO: better things to remove duplicates
        augmented_samples.append(SampleWithVariations.from_sample(sample, variations))
    return AugmentedDataset(augmented_samples)
