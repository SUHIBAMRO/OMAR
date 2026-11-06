"""Independently measure B3 inference latency for a model of the EXACT
architecture train_B3_data_driven.py builds (confirmed: it imports and
calls the same train_B3.build_model with the same default hyperparameters
as PI training) -- timing is a property of the architecture's forward
pass, not the learned weights, so a freshly-initialized model of this
architecture gives the same real, measured number any trained DD
checkpoint would, without needing to fetch one from Drive.
"""
import argparse
import time
import types

import torch

from omar_pfem.train_B3 import build_fixed_geometry, build_model
from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION, sample_material_and_load
from omar_pfem.evaluate_B3 import benchmark_inference_latency_B3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(f"device: {device}")

# Exact defaults shared by train_B3.py (PI) and train_B3_data_driven.py (DD)
args = types.SimpleNamespace(
    model="Transolver_Irregular_Mesh",
    n_hidden=256, n_layers=4, n_heads=8, mlp_ratio=2,
    dropout=0.1, unified_pos=0, ref=16, slice_num=128,
)

Ntheta, Nr, Nz = DEFAULT_RESOLUTION
geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
model = build_model(args, device).to(dtype)

E_node, nu_node, phi, _, _ = sample_material_and_load(Ntheta, Nr, Nz, seed=0)
E_node = torch.tensor(E_node, dtype=dtype, device=device).unsqueeze(0)
nu_node = torch.tensor(nu_node, dtype=dtype, device=device).unsqueeze(0)
phi = torch.tensor([phi], dtype=dtype, device=device)

latency = benchmark_inference_latency_B3(model, geom, E_node, nu_node, phi, device, dtype)
print(latency)
