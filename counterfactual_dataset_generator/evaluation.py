from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, TypeVar
from attrs import define
from counterfactual_dataset_generator.agregators import AveragePerformancePerCategory

from counterfactual_dataset_generator.types import (
    AugmentedSample,
    Category,
    ModelEvaluator,
    Performance,
    Results,
    StatsAgregator,
)
from counterfactual_dataset_generator.utils import mean


T = TypeVar("T")


def compute_performances(samples: Iterable[AugmentedSample], model: ModelEvaluator) -> Results:
    performances = []
    for sample in samples:
        performance = [
            (model(variation.text, sample.expected_output), variation.categories)
            for variation in sample.get_variations()
        ]
        performances.append(performance)
    return performances


def evaluate(
    samples: Iterable[AugmentedSample],
    model: ModelEvaluator,
    agregator: StatsAgregator[T] = AveragePerformancePerCategory(),
) -> T:
    return agregator(compute_performances(samples, model))


def evaluate_and_print(
    samples: Iterable[AugmentedSample],
    model: ModelEvaluator,
    agregator: StatsAgregator[T] = AveragePerformancePerCategory(),
):
    agregator.save_agregation(compute_performances(samples, model))


def evaluate_and_save(
    samples: Iterable[AugmentedSample],
    model: ModelEvaluator,
    path: str,
    agregator: StatsAgregator[T] = AveragePerformancePerCategory(),
):
    with Path(path).open("w") as f:
        agregator.save_agregation(compute_performances(samples, model), file=f)
