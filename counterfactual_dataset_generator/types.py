import abc
from typing import Any, Callable, Iterable, NamedTuple, Optional, Sequence
from attrs import define

Input = str
Output = Optional[str]
Category = str


class Variation(NamedTuple):
    text: Input
    categories: tuple[Category, ...]


class AugmentedInput(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def input(self) -> Input:
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
