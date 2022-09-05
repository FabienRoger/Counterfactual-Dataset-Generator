from typing import Any, Sequence, TypeVar

T = TypeVar("T")


def other(t: tuple[T, T], x: T) -> T:
    if x == t[0]:
        if x == t[1]:
            raise ValueError(f"{t} contains two copies of {x}")
        return t[1]
    if x != t[1]:
        raise ValueError(f"{t} does not contain {x}")
    return t[0]


def mean(l: Sequence[float]) -> float:
    return sum(l) / len(l)
