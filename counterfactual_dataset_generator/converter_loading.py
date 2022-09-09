from functools import lru_cache
import json
from collections import defaultdict
from pathlib import Path
from random import choice
from typing import Callable, DefaultDict, Iterable, Mapping, MutableMapping, OrderedDict, Sequence

import spacy
from attrs import define

from counterfactual_dataset_generator.types import Category, Converter, Input
from counterfactual_dataset_generator.utils import other

nlp = spacy.load("en_core_web_sm")

default_converter_paths: Mapping[str, str] = {
    "gender": "counterfactual_dataset_generator/data/converters/gender.json",
    "west_v_asia": "counterfactual_dataset_generator/data/converters/west_v_asia.json",
}


@define
class ConverterDataset:
    categories: tuple[Category, Category]
    correspondances: list[tuple[list[str], list[str]]]

    @classmethod
    def from_json(cls, json_dict: OrderedDict):
        categories = tuple(json_dict["categories"])
        correspondances_maps = json_dict["correspondances"]
        cat_0, cat_1 = categories
        correspondances = []
        for m in correspondances_maps:
            correspondances.append((m[cat_0], m[cat_1]))
        return ConverterDataset(categories, correspondances)  # type: ignore


Transformation = Callable[[str], str]
CorrespondanceDict = dict[Category, DefaultDict[str, list[str]]]
DEFAULT_TRANSFORMATIONS = [
    lambda s: s.lower(),
    lambda s: s.upper(),
    lambda s: s.capitalize(),
]


@define
class SimpleConverter(Converter):
    categories: tuple[Category, Category]
    correspondance_dict: CorrespondanceDict

    @classmethod
    def from_default(cls, name: str = "gender"):
        return SimpleConverter.from_json(default_converter_paths[name])

    @classmethod
    def from_json(cls, path: str, transformations: Iterable[Transformation] = DEFAULT_TRANSFORMATIONS):
        with Path(path).open("r", encoding="utf-8") as f:
            json_dict = json.loads(f.read())
            return SimpleConverter.from_ds(ConverterDataset.from_json(json_dict), transformations)

    @classmethod
    def from_ds(cls, ds: ConverterDataset, transformations: Iterable[Transformation] = DEFAULT_TRANSFORMATIONS):
        correspondance_dict: CorrespondanceDict = {}
        for c in ds.categories:
            correspondance_dict[c] = defaultdict(lambda: [])

        for correspondance in ds.correspondances:
            correspondance_t = {
                c: {t.__code__: list(map(t, l)) for t in transformations} for c, l in zip(ds.categories, correspondance)
            }

            for c, l in zip(ds.categories, correspondance):
                for word in l:
                    for t in transformations:
                        correspondance_dict[c][t(word)] += correspondance_t[other(ds.categories, c)][t.__code__]

        return SimpleConverter(ds.categories, correspondance_dict)

    def convert_to(self, inp: Input, to: Category) -> Input:
        from_category = other(self.categories, to)
        doc = nlp(inp)
        r = ""
        position_in_text = 0
        for t in doc:
            if self.correspondance_dict[from_category][t.text]:
                r += doc.text[position_in_text : t.idx] + choice(self.correspondance_dict[from_category][t.text])
                position_in_text = t.idx + len(t)
        return r + doc.text[position_in_text:]
