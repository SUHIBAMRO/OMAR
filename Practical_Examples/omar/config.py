"""
Configurations for the B1 and B2 hyperelasticity benchmarks, written in the same
style as utils/config.py (HyperElasticity dict) from the original VINO repo.

B1: unit square (1 x 1) under vertical top-edge compression, fixed bottom edge.
B2: quarter ring (R_in=1, R_out=2) under internal pressure, mapped to a
    [0, 1] x [0, 1] computational domain via (r, theta).

Materials: 'neohookean', 'mooneyrivlin', 'arrudaboyce'
"""


def _make_config(plate_length, plate_width, num_pts_x, num_pts_y, energy_type,
                  r_inner=None, r_outer=None, tag='B'):
    cfg = {}

    # define the geometry
    cfg["plate_length"] = plate_length
    cfg["plate_width"] = plate_width

    # define network parameters
    cfg["num_pts_x"] = num_pts_x
    cfg["num_pts_y"] = num_pts_y
    cfg["num_epoch"] = 1000
    cfg["num_epoch_LBFGS"] = 0
    cfg["print_epoch"] = 100
    cfg["data_type"] = 'float64'
    cfg["normalized"] = True

    # define model parameters
    cfg["beam"] = dict()
    cfg["beam"]["E"] = 1e3
    cfg["beam"]["nu"] = 0.25
    cfg["beam"]["state"] = "plane stress"
    cfg["beam"]["param_c1"] = 630
    cfg["beam"]["param_c2"] = -1.2
    cfg["beam"]["param_c"] = 100
    # Arruda-Boyce parameters
    cfg["beam"]["param_mu_ab"] = 630
    cfg["beam"]["param_N_ab"] = 5.0
    cfg["beam"]["param_kappa_ab"] = 100
    cfg["beam"]["pressure"] = -10.0
    cfg["beam"]["traction_type"] = 'GRF'
    cfg["beam"]["traction_mean"] = -10.0
    cfg["beam"]["traction_var"] = 0.1
    cfg["beam"]["traction_scale"] = 0.08
    cfg["beam"]["numPtsU"] = cfg["num_pts_x"]
    cfg["beam"]["numPtsV"] = cfg["num_pts_y"]
    cfg["beam"]['length'] = cfg["plate_length"]
    cfg["beam"]['width'] = cfg["plate_width"]
    cfg["beam"]["R_inner"] = r_inner
    cfg["beam"]["R_outer"] = r_outer

    cfg["fno"] = dict()
    cfg["fno"]["mode1"] = 8
    cfg["fno"]["mode2"] = 8
    cfg["fno"]["width"] = 32
    cfg["fno"]["depth"] = 4
    cfg["fno"]["channels_last_proj"] = 128
    cfg["fno"]["padding"] = 0
    cfg["fno"]["n_train"] = 500
    cfg["fno"]["n_test"] = 50
    cfg["fno"]["n_data"] = cfg["fno"]["n_train"] + cfg["fno"]["n_test"]
    cfg["fno"]["batch_size"] = 50
    cfg["fno"]["learning_rate"] = 0.001
    cfg["fno"]["weight_decay"] = 1e-4

    # define FEM parameters
    cfg["FEM_data"] = dict()
    cfg["FEM_data"]["num_pts_x"] = cfg["num_pts_x"]
    cfg["FEM_data"]["num_pts_y"] = cfg["num_pts_y"]
    cfg["FEM_data"]["energy_type"] = energy_type
    cfg["FEM_data"]["dimension"] = 2

    cfg["dir"] = './data/'
    cfg["path"] = (cfg["dir"] + tag + '_n' + str(cfg["fno"]["n_data"]) +
                   '_' + energy_type +
                   '_' + str(cfg["num_pts_x"]) + 'X' + str(cfg["num_pts_y"]) + '.npz')

    return cfg


# B1 -- unit square (1 x 1), 64 x 64 grid
B1_NH = _make_config(1., 1., 64, 64, 'neohookean', tag='B1')
B1_MR = _make_config(1., 1., 64, 64, 'mooneyrivlin', tag='B1')
B1_AB = _make_config(1., 1., 64, 64, 'arrudaboyce', tag='B1')

# B2 -- quarter ring (R_in=1, R_out=2), mapped to [0, 1] x [0, 1], 64 x 64 grid
B2_NH = _make_config(1., 1., 64, 64, 'neohookean', r_inner=1.0, r_outer=2.0, tag='B2')
B2_MR = _make_config(1., 1., 64, 64, 'mooneyrivlin', r_inner=1.0, r_outer=2.0, tag='B2')
B2_AB = _make_config(1., 1., 64, 64, 'arrudaboyce', r_inner=1.0, r_outer=2.0, tag='B2')
