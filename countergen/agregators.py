from collections import defaultdict
from statistics import geometric_mean
from typing import Any, Iterable, Mapping, Optional, TextIO, TypeVar
from attrs import define

from countergen.types import (
    AugmentedSample,
    Category,
    ModelEvaluator,
    Performance,
    Results,
    StatsAgregator,
)
from countergen.utils import mean


@define
class AveragePerformancePerCategory(StatsAgregator):
    use_geometric_mean: bool = False

    def __call__(self, performances: Results) -> Mapping[Category, float]:
        performances_per_category: defaultdict[Category, list[float]] = defaultdict(lambda: [])
        for sample_perfs in performances:
            for perf, categories in sample_perfs:
                for c in categories:
                    performances_per_category[c].append(perf)

        mean_ = geometric_mean if self.use_geometric_mean else mean
        avg_performances_per_category = {c: mean_(perfs) for c, perfs in performances_per_category.items()}
        return avg_performances_per_category

    def save_agregation(self, performances: Results, file: Optional[TextIO] = None):
        r = self(performances)

        print("Average performance per category:", file=file)
        for c, perf in r.items():
            print(f"{c}: {perf:.6f}", file=file)


@define
class AverageDifference(StatsAgregator):
    positive_category: Category
    negative_category: Category
    relative_difference: bool = False

    def __call__(self, performances: Results) -> float:
        differences = []
        for sample_perfs in performances:
            positive_perfs = [perf for perf, categories in sample_perfs if self.positive_category in categories]
            negative_perfs = [perf for perf, categories in sample_perfs if self.negative_category in categories]
            if positive_perfs and negative_perfs:
                diff = mean(positive_perfs) - mean(negative_perfs)
                if self.relative_difference:
                    diff /= mean(positive_perfs)
                differences.append(diff)
        return mean(differences)

    def save_agregation(self, performances: Results, file: Optional[TextIO] = None):
        r = self(performances)

        relative = "relative " if self.relative_difference else ""
        print(
            f"The {relative}performance between category {self.positive_category} and category {self.negative_category}:",
            file=file,
        )
        print(f"{r:.6f}", file=file)


@define
class MultipleStatsAgregator(StatsAgregator):
    agregators: list[StatsAgregator]

    def __call__(self, performances: Results) -> list[Any]:
        return [ag(performances) for ag in self.agregators]

    def save_agregation(self, performances: Results, file: Optional[TextIO] = None):
        for ag in self.agregators:
            ag.save_agregation(performances, file)
