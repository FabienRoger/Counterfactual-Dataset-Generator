from ast import Or
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Union
from collections import OrderedDict
from counterfactual_dataset_generator.converter_loading import SimpleConverter
from counterfactual_dataset_generator.types import Converter, Input, Output
from attrs import define

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


@define
class SampleWithVariations(Sample):
    variations: list[Input] = []

    @classmethod
    def from_sample(cls, s: Sample, variations: list[Input] = []):
        return SampleWithVariations(s.input, s.expected_output, variations)

    def to_json_dict(self) -> OrderedDict:
        d: OrderedDict[str, Any] = OrderedDict({"input": self.input})
        if self.expected_output is not None:
            d["expected_output"] = self.expected_output
        d["variations"] = self.variations
        return d


@define
class Dataset:
    samples: list[Sample]

    @classmethod
    def from_default(cls, name: str = "doublebind"):
        return Dataset.from_jsonl(default_dataset_paths[name])

    @classmethod
    def from_jsonl(cls, path: str):
        with Path(path).open("r") as f:
            data = [json.loads(line) for line in f]
            samples = []
            for d in data:
                expected_output = d["expected_output"] if "expected_output" in d else None
                samples.append(Sample(d["input"], expected_output))
        return Dataset(samples)


@define
class AugmentedDataset:
    samples: list[SampleWithVariations]

    def save_to_jsonl(self, path: str):
        with Path(path).open("w") as f:
            for sample in self.samples:
                json.dump(sample.to_json_dict(), f)
                f.write("\n")


def generate_variations_pair(converter: Converter, ds: Dataset) -> AugmentedDataset:
    augmented_samples = []
    for sample in ds.samples:
        variations = [converter.convert_to(sample.input, category) for category in converter.categories]
        augmented_samples.append(SampleWithVariations.from_sample(sample, variations))
    return AugmentedDataset(augmented_samples)


def generate_all_variations(converters: Iterable[Converter], ds: Dataset) -> AugmentedDataset:
    augmented_samples = []
    for sample in ds.samples:
        variations = [sample.input]
        for converter in converters:
            new_variations = []
            for category in converter.categories:
                new_variations += [converter.convert_to(v, category) for v in variations]
            variations = list(set(new_variations))
        augmented_samples.append(SampleWithVariations.from_sample(sample, variations))
    return AugmentedDataset(augmented_samples)
