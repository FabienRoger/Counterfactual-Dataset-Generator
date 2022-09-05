import abc
from typing import Any, Callable, Generic, Iterable, NamedTuple, Optional, Sequence, TextIO, TypeVar
from attrs import define

Input = str
Output = Optional[str]
Category = str


class Variation(NamedTuple):
    text: Input
    categories: tuple[Category, ...]


class AugmentedSample(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def input(self) -> Input:
        ...

    @abc.abstractproperty
    def expected_output(self) -> Optional[Output]:
        ...

    @abc.abstractmethod
    def get_variations(self) -> Sequence[Variation]:
        ...


class Converter(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def categories(self) -> tuple[Category, Category]:
        ...

    @abc.abstractmethod
    def convert_to(self, inp: Input, to: Category) -> Input:
        ...


Performance = float  # between zero & one (one is better)
ModelEvaluator = Callable[[Input, Optional[Output]], Performance]
SampleResults = Iterable[tuple[Performance, tuple[Category, ...]]]
Results = Iterable[SampleResults]
T = TypeVar("T")


class StatsAgregator(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, performances: Results) -> T:
        ...

    @abc.abstractmethod
    def save_agregation(self, performances: Results, file: Optional[TextIO] = None):
        ...
