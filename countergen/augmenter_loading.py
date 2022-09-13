import json
from collections import defaultdict
from pathlib import Path
from random import choice
from typing import Callable, DefaultDict, Dict, Iterable, List, Mapping, OrderedDict, Sequence, Tuple

import spacy
from attrs import define

from countergen.types import Category, Augmenter, Input
from countergen.utils import MODULE_PATH, other

default_converter_paths: Mapping[str, str] = {
    "gender": f"{MODULE_PATH}/data/converters/gender.json",
    "west_v_asia": f"{MODULE_PATH}/data/converters/west_v_asia.json",
}


@define
class ConversionDataset:
    categories: Tuple[Category, Category]
    correspondances: List[Tuple[List[str], List[str]]]

    @classmethod
    def from_json(cls, json_dict: OrderedDict):
        categories = tuple(json_dict["categories"])
        correspondances_maps = json_dict["correspondances"]
        cat_0, cat_1 = categories
        correspondances = []
        for m in correspondances_maps:
            correspondances.append((m[cat_0], m[cat_1]))
        return ConversionDataset(categories, correspondances)  # type: ignore


Transformation = Callable[[str], str]
CorrespondanceDict = Dict[Category, DefaultDict[str, List[str]]]
DEFAULT_TRANSFORMATIONS = [
    lambda s: s.lower(),
    lambda s: s.upper(),
    lambda s: s.capitalize(),
]


@define
class SimpleAugmenter(Augmenter):
    categories: Tuple[Category, Category]
    correspondance_dict: CorrespondanceDict
    nlp: spacy.language.Language = spacy.load("en_core_web_sm")

    @classmethod
    def from_default(cls, name: str = "gender"):
        return SimpleAugmenter.from_json(default_converter_paths[name])

    @classmethod
    def from_json(
        cls,
        path: str,
        transformations: Iterable[Transformation] = DEFAULT_TRANSFORMATIONS,
    ):
        with Path(path).open("r", encoding="utf-8") as f:
            json_dict = json.loads(f.read())
            return SimpleAugmenter.from_ds(ConversionDataset.from_json(json_dict), transformations)

    @classmethod
    def from_ds(
        cls,
        ds: ConversionDataset,
        transformations: Iterable[Transformation] = DEFAULT_TRANSFORMATIONS,
    ):
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

        return SimpleAugmenter(ds.categories, correspondance_dict)

    def convert_to(self, inp: Input, to: Category) -> Input:
        from_category = other(self.categories, to)
        doc = self.nlp(inp)
        r = ""
        position_in_text = 0
        for t in doc:
            if self.correspondance_dict[from_category][t.text]:
                r += doc.text[position_in_text : t.idx] + choice(self.correspondance_dict[from_category][t.text])
                position_in_text = t.idx + len(t)
        return r + doc.text[position_in_text:]
