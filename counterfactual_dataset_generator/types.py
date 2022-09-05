import abc
from typing import Any, Callable, Optional
from attrs import define

Input = str
Output = Optional[str]


Performance = float  # between zero & one (one is better)
ModelEvaluator = Callable[[Input, Optional[Output]], Performance]

Category = str


class Converter(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def categories(self) -> tuple[Category, Category]:
        ...

    @abc.abstractmethod
    def convert_to(self, inp: Input, to: Category) -> Input:
        ...
