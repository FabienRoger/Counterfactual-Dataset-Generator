import os
from math import exp, log2
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, TypeVar, Tuple

import torch
from tqdm import tqdm  # type: ignore

MODULE_PATH = str(Path(__file__).parent)

T = TypeVar("T")


def other(t: Tuple[T, T], x: T) -> T:
    if x == t[0]:
        if x == t[1]:
            raise ValueError(f"{t} contains two copies of {x}")
        return t[1]
    if x != t[1]:
        raise ValueError(f"{t} does not contain {x}")
    return t[0]


def unwrap_or(maybe: Optional[T], default: T) -> T:
    return default if maybe is None else maybe


def mean(l: Sequence[float]) -> float:
    return sum(l) / len(l)


def geometric_mean(l: Sequence[float]) -> float:
    return 2 ** (mean(list(map(log2, l))))


def perplexity(log_probs: Sequence[float]):
    """Take in natural log probs, returns (average) perplexity"""
    return exp(mean(log_probs))


def concat_dicts(dicts: Sequence[Mapping[Any, torch.Tensor]]) -> Dict[Any, torch.Tensor]:
    if not dicts:
        raise ValueError("dicts is empty")
    keys = dicts[0].keys()
    for d in dicts:
        if d.keys() != keys:
            raise ValueError("dicts must have the same keys")
    return {k: torch.cat([d[k] for d in dicts], dim=-1) for k in keys}


def remove_last_tok(d: Mapping[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
    return {k: t[:, :-1] for k, t in d.items()}


def maybe_tqdm(it: Iterable[T], do_tqdm: bool = False, **kwargs) -> Iterable[T]:
    if do_tqdm:
        return tqdm(it, **kwargs)
    else:
        return it

def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"