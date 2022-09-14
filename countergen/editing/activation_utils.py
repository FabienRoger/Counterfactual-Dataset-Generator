from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional

import torch
from torch import nn
from transformers import BatchEncoding, GPT2LMHeadModel
from countergen.types import AugmentedSample, Category

from countergen.utils import get_gpt_tokenizer, unwrap_or


def get_mlp_layers(model: GPT2LMHeadModel, layer_numbers: Optional[List[int]]):
    model_transformer: nn.ModuleList = model.transformer.h  # type: ignore
    layer_numbers = unwrap_or(layer_numbers, list(range(len(model_transformer))))
    return [l.mlp for i, l in enumerate(model_transformer) if i in layer_numbers]


def get_res_layers(model: GPT2LMHeadModel, layer_numbers: Optional[List[int]]):
    model_transformer: nn.ModuleList = model.transformer.h  # type: ignore
    layer_numbers = unwrap_or(layer_numbers, list(range(len(model_transformer))))
    return [l for i, l in enumerate(model_transformer) if i in layer_numbers]


def get_corresponding_activations(
    samples: Iterable[AugmentedSample], model: nn.Module, layers: Iterable[nn.Module]
) -> Mapping[Category, List[Dict[nn.Module, torch.Tensor]]]:
    """For each category, returns a list of activations obtained by running the variations corresponding to this category."""

    tokenizer = get_gpt_tokenizer()

    activations_by_cat = defaultdict(lambda: [])
    for sample in samples:
        for inp, categories in sample.get_variations():
            acts = get_activations(tokenizer(inp), model, layers)
            for cat in categories:
                activations_by_cat[cat].append(acts)
    return activations_by_cat


Operation = Callable[[torch.Tensor], torch.Tensor]


def get_activations(
    tokens: BatchEncoding, model: nn.Module, layers: Iterable[nn.Module], operation: Operation = lambda x: x
) -> Dict[nn.Module, torch.Tensor]:
    handles = []
    activations: Dict[nn.Module, torch.Tensor] = {}

    def hook_fn(module, inp, out):
        activations[module] = operation(out[0].detach())

    for layer in layers:
        handles.append(layer.register_forward_hook(hook_fn))
    try:
        model(**tokens.to(model.device))
    except Exception as e:
        raise e
    finally:
        for handle in handles:
            handle.remove()
    return activations


# (module, input, output) -> output
ModificationFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


def run_and_modify(
    tokens: BatchEncoding, model: nn.Module, modification_fns: Dict[nn.Module, ModificationFn] = {}
) -> BatchEncoding:
    handles = []
    for layer, f in modification_fns.items():
        handles.append(layer.register_forward_hook(f))  # type: ignore
    try:
        out = model(**tokens.to(model.device))
        return out
    except Exception as e:
        raise e
    finally:
        for handle in handles:
            handle.remove()
