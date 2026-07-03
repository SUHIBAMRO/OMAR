"""
Dataset classes for the B1 and B2 hyperelasticity benchmarks, written in the
same style as utils/database_makers.py (Hyperelasticity_dataset, FGM_dataset)
from the original VINO repo. Reuses GaussianNormalizer, so the normalization
convention matches the rest of the codebase exactly.

Expected .npz files (same 'traction' / 'disp2D' convention as the original
Hyperelasticity_dataset):
    B1: traction [N, numPtsU]  -- GRF traction on the top edge (varies with x)
        disp2D   [N, numPtsV, numPtsU, 2]
    B2: traction [N, numPtsV]  -- GRF internal pressure (varies with theta)
        disp2D   [N, numPtsV, numPtsU, 2]  (in mapped (r, theta) coordinates)
"""
import os

import numpy as np
from torch.utils.data import Dataset

from utils.database_makers import GaussianNormalizer


class B1Dataset(Dataset):
    def __init__(self, model_data):
        n = model_data["fno"]["n_data"]
        n_u = model_data["beam"]["numPtsU"]
        n_v = model_data["beam"]["numPtsV"]
        if not os.path.exists(model_data["dir"]):
            os.makedirs(model_data["dir"])

        if os.path.isfile(model_data["path"]):
            loaded_data = np.load(model_data["path"])
            traction = loaded_data['traction']
            disp2d = loaded_data['disp2D']
            print("Found saved dataset at", model_data["path"])
        else:
            print("NOT Found saved dataset at", model_data["path"])
            raise ValueError("Please run the FEM data generation script for B1 first.")

        # fix dimensions: traction varies along the top edge (x-direction)
        traction = traction.reshape(n, 1, n_u, 1)

        # repeat inputs across the y-direction
        traction = np.tile(traction, (1, n_v, 1, 1))

        self.x = traction
        self.y = disp2d

        if model_data["normalized"]:
            self.normalizer_x = GaussianNormalizer(self.x)
            self.normalizer_y = GaussianNormalizer(self.y)
            self.x = self.normalizer_x.encode(self.x)
            self.y = self.normalizer_y.encode(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class B2Dataset(Dataset):
    def __init__(self, model_data):
        n = model_data["fno"]["n_data"]
        n_u = model_data["beam"]["numPtsU"]
        n_v = model_data["beam"]["numPtsV"]
        if not os.path.exists(model_data["dir"]):
            os.makedirs(model_data["dir"])

        if os.path.isfile(model_data["path"]):
            loaded_data = np.load(model_data["path"])
            traction = loaded_data['traction']
            disp2d = loaded_data['disp2D']
            print("Found saved dataset at", model_data["path"])
        else:
            print("NOT Found saved dataset at", model_data["path"])
            raise ValueError("Please run the FEM data generation script for B2 first.")

        # fix dimensions: pressure varies with theta (rows), a GRF profile along
        # the inner boundary, same convention as B1's traction varying along x
        pressure = traction.reshape(n, n_v, 1, 1)

        # repeat inputs across the r-direction (columns)
        pressure = np.tile(pressure, (1, 1, n_u, 1))

        self.x = pressure
        self.y = disp2d

        if model_data["normalized"]:
            self.normalizer_x = GaussianNormalizer(self.x)
            self.normalizer_y = GaussianNormalizer(self.y)
            self.x = self.normalizer_x.encode(self.x)
            self.y = self.normalizer_y.encode(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
