import abc
from typing import (
    Callable,
    Generic,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TextIO,
    TypeVar,
    List,
    Tuple,
)
from attrs import define

Input = str  # The input to an NLP mode
Output = List[str]  # The different acceptable outputs of the NLP, string label or number, but in string format

Performance = float  # usually between zero & one (one is better)
ModelEvaluator = Callable[[Input, Output], Performance]

Category = str  # The different kinds of data produced by converters


class Converter(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def categories(self) -> Tuple[Category, Category]:
        ...

    @abc.abstractmethod
    def convert_to(self, inp: Input, to: Category) -> Input:
        ...


class Variation(NamedTuple):
    text: Input
    categories: Tuple[Category, ...]


class AugmentedSample(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def input(self) -> Input:
        ...

    @abc.abstractproperty
    def expected_output(self) -> Output:
        ...

    @abc.abstractmethod
    def get_variations(self) -> Sequence[Variation]:
        ...


SampleResults = Iterable[Tuple[Performance, Tuple[Category, ...]]]
Results = Iterable[SampleResults]

Aggregate = TypeVar("Aggregate")


class StatsAggregator(Generic[Aggregate], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, performances: Results) -> Aggregate:
        ...

    @abc.abstractmethod
    def save_aggregation(self, aggregate: Aggregate, file: Optional[TextIO] = None):
        ...

    @abc.abstractmethod
    def load_aggregation(self, file: TextIO) -> Aggregate:
        ...

    @abc.abstractmethod
    def display(self, aggregates: Mapping[str, Aggregate]):
        ...
