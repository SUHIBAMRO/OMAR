"""
Training script for the B2 benchmark (quarter ring, R_in=1, R_out=2,
internal pressure), for the three materials (Neo-Hookean, Mooney-Rivlin,
Arruda-Boyce). Structured exactly like VINO_Hyperelasticity.py: same FNO
architecture, same training loop / optimizer (utils.fno_utils.train_fno,
jax.example_libraries.optimizers.adam), same collate_fn / count_params /
model_evaluation utilities -- nothing there is changed.

Uses B2VinoLoss (omar/losses.py): per-element Gauss quadrature instead of
DemHyperelasticityLoss's whole-domain trapezoidal quadrature -- the paper's
own exact closed-form element integration is not possible on B2's curved
(polar) geometry (see B2VinoLoss's docstring for why), so Gauss quadrature
is the mathematically honest adaptation for a curved domain.
"""
import sys
sys.path.insert(0, '/content/OMAR/Practical_Examples')

import time

import jax
import torch
import numpy as np
from jax import random
from jax.example_libraries import optimizers
from torch.utils.data import DataLoader, random_split

from omar.config import B2_NH, B2_MR, B2_AB
from omar.datasets import B2Dataset
from omar.losses import FNO2d_B2, B2VinoLoss
from utils.fno_utils import count_params, train_fno, model_evaluation, collate_fn

jax.config.update("jax_enable_x64", True)
torch.manual_seed(42)
np.random.seed(42)


def run(model_data, seed=0):
    model_data["nrg"] = random.PRNGKey(seed)

    assert model_data["fno"]["mode1"] <= model_data["beam"]["numPtsU"] // 2 + 1
    assert model_data["fno"]["mode2"] <= model_data["beam"]["numPtsV"] // 2 + 1

    dataset = B2Dataset(model_data)
    # explicit generator (not the global torch RNG) so the split is reproducible
    # from (model_data, seed) alone -- needed to rebuild the same test set later
    # for plotting, independent of call order or which other cases were skipped
    # via checkpoint.
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [model_data["fno"]["n_train"], model_data["fno"]["n_test"]],
                                               generator=split_generator)
    normalizers = [dataset.normalizer_x, dataset.normalizer_y] if model_data["normalized"] else None

    train_loader = DataLoader(train_dataset, batch_size=model_data["fno"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=model_data["fno"]["batch_size"],
                             shuffle=True, collate_fn=collate_fn)

    model = FNO2d_B2(
        modes1=model_data["fno"]["mode1"],
        modes2=model_data["fno"]["mode2"],
        width=model_data["fno"]["width"],
        depth=model_data["fno"]["depth"],
        channels_last_proj=model_data["fno"]["channels_last_proj"],
        padding=model_data["fno"]["padding"],
        out_channels=2,
        model_data=model_data["beam"]
    )
    _x, _ = train_dataset[0]
    _x = np.expand_dims(_x, axis=0)
    _, model_params = model.init_with_output(model_data["nrg"], _x)

    print(f'\nOur model has {count_params(model_params)} parameters.')

    init_fun, update_fun, get_params = optimizers.adam(model_data["fno"]["learning_rate"])
    opt_state = init_fun(model_params)

    loss_fn = B2VinoLoss(model, model_data, normalizers, d=1, p=1, size_average=False)

    print("Training (ADAM)...")
    t0 = time.time()
    model_params, train_losses, test_losses = train_fno(model, train_loader, test_loader, loss_fn,
                                                        get_params, update_fun, opt_state)
    print("Training (ADAM) time: ", time.time() - t0)

    print("Evaluating Test Data")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model_evaluation(model, model_data, model_params, loss_fn, test_loader, normalizers)

    return {'params': model_params, 'train_losses': train_losses, 'test_losses': test_losses}


if __name__ == "__main__":
    for name, cfg, seed in [("NH", B2_NH, 0), ("MR", B2_MR, 1), ("AB", B2_AB, 2)]:
        print(f"\n===== B2 x {name} =====")
        run(cfg, seed=seed)
