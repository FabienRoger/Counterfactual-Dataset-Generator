from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch import nn
from transformers import BatchEncoding, GPT2LMHeadModel

from countergen.utils import unwrap_or


def get_mlp_layers(model: GPT2LMHeadModel, layer_numbers: Optional[List[int]]):
    model_transformer: nn.ModuleList = model.transformer.h  # type: ignore
    layer_numbers = unwrap_or(layer_numbers, list(range(len(model_transformer))))
    return [l.mlp for i, l in enumerate(model_transformer) if i in layer_numbers]


def get_res_layers(model: GPT2LMHeadModel, layer_numbers: Optional[List[int]]):
    model_transformer: nn.ModuleList = model.transformer.h  # type: ignore
    layer_numbers = unwrap_or(layer_numbers, list(range(len(model_transformer))))
    return [l for i, l in enumerate(model_transformer) if i in layer_numbers]


# def get_all_activations(ds: VariationDataset, model, layers, mode: str = "val"):
#     prompts = ds.get_tokens_by_category(mode)
#     activations = {}
#     for category, l in prompts.items():
#         activations[category] = {}
#         for i, inps in enumerate(l):
#             acts = get_activations(inps, model, layers)
#             for layer, act in acts.items():
#                 if i == 0:
#                     activations[category][layer] = []
#                 activations[category][layer].append(act)
#     return activations


# def get_corresponding_activations(datasets, model, layers, mode: str = "val"):
#     """datasets is a dict where keys are categories & values are StringDatasets."""
#     activations = {}
#     for category, ds in datasets.items():
#         ds: StringsDataset = ds
#         prompts = ds.get_all_tokens(mode)
#         activations[category] = {}
#         for i, inps in enumerate(prompts):
#             acts = get_activations(inps, model, layers)
#             for layer, act in acts.items():
#                 if i == 0:
#                     activations[category][layer] = []
#                 activations[category][layer].append(act)
#     return activations


Operation = Callable[[torch.Tensor], torch.Tensor]


def get_activations(
    tokens: BatchEncoding, model: nn.Module, layers: Iterable[nn.Module], operation: Operation = lambda x: x
) -> Dict[nn.Module, BatchEncoding]:
    handles = []
    activations: Dict[nn.Module, BatchEncoding] = {}

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


# (module, input, output) => output
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
