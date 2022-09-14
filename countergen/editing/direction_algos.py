from typing import List
import torch
from torch import nn
from torchmetrics import HingeLoss

from countergen.config import VERBOSE
from countergen.editing.activation_ds import ActivationsDataset
from countergen.editing.models import fit_model, get_bottlenecked_linear, get_bottlenecked_mlp
from countergen.utils import maybe_tqdm, orthonormalize


def inlp(ds: ActivationsDataset, n_dim: int = 8, n_training_iters: int = 400) -> List[torch.Tensor]:
    working_ds = ds
    dirs = []

    tot_n_dims = ds.x_data.shape[-1]
    output_dims = torch.max(ds.y_data).item() + 1

    g = maybe_tqdm(range(n_dim), VERBOSE >= 1)
    for i in g:
        model = get_bottlenecked_linear(tot_n_dims, output_dims)
        last_epoch_perf = fit_model(model, ds, n_training_iters, loss_fn=HingeLoss())

        dir = model[0].weight.detach()[0]
        dir = orthonormalize(dir, dirs)

        if i == 0:
            working_ds = working_ds.project(dir)
        else:
            working_ds.project_(dir)

        dirs.append(dir)

        if VERBOSE >= 1:
            g.set_postfix(**last_epoch_perf)
    return dirs


def bottlenecked_mlp_span(ds: ActivationsDataset, n_dim: int = 8, n_training_iters: int = 400) -> List[torch.Tensor]:
    tot_n_dims = ds.x_data.shape[-1]
    output_dims = torch.max(ds.y_data).item() + 1
    model = get_bottlenecked_mlp(tot_n_dims, output_dims, bottleneck_dim=n_dim)
    last_epoch_perf = fit_model(model, ds, n_training_iters, loss_fn=HingeLoss())
    if VERBOSE >= 2:
        print(str(last_epoch_perf))

    return [model[0].weight.detach()[i] for i in range(n_dim)]


# TODO: Add the adversarial algorithm
