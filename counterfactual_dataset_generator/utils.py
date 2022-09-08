from math import exp, log2
import torch
from typing import Any, Callable, Sequence, TypeVar

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


def geometric_mean(l: Sequence[float]) -> float:
    return 2 ** (mean(list(map(log2, l))))


def perplexity(log_probs: Sequence[float]):
    """Take in natural log probs, returns (average) perplexity"""
    return exp(mean(log_probs))

def concat_dicts(dicts: Sequence[dict[Any, torch.Tensor]]) -> dict[Any, torch.Tensor]:
    if not dicts:
        raise ValueError("dicts is empty")
    keys = dicts[0].keys()
    for d in dicts:
        if d.keys() != keys:
            raise ValueError("dicts must have the same keys")
    return {k: torch.cat([d[k] for d in dicts], dim=-1) for k in keys}
