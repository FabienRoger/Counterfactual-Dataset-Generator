from pathlib import Path
from typing import Iterable, Mapping, TypeVar
from countergen.agregators import AveragePerformancePerCategory
from countergen.config import VERBOSE
from countergen.data_augmentation import AugmentedDataset
from countergen.generative_models import get_huggingface_gpt_model_evaluator

from countergen.types import (
    AugmentedSample,
    Category,
    ModelEvaluator,
    Performance,
    Results,
    StatsAgregator,
)
from countergen.utils import maybe_tqdm, mean


T = TypeVar("T")


def compute_performances(samples: Iterable[AugmentedSample], model: ModelEvaluator) -> Results:
    performances = []
    for sample in maybe_tqdm(samples, VERBOSE >= 2):
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
    with Path(path).open("w", encoding="utf-8") as f:
        agregator.save_agregation(compute_performances(samples, model), file=f)
