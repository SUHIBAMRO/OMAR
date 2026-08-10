"""
Converts the B2 HDF5 dataset (data_generate_B2.py) into the Q4 NPZ format
expected by the training script -- mirrors convert_B1_quad.py's structure,
adapted for B2's curved geometry and mixed symmetry BCs:
  - B1 has one fixed edge (bottom_nodes) and one loaded edge (top_edges);
    B2 instead has one PRESSURE-loaded edge (inner_edges, the r=R_in arc)
    and TWO symmetry edges, each constraining only one displacement
    component (theta0_nodes: u_y=0: thetahalfpi_nodes: u_x=0).
  - B1's per-node boundary force is (0, ty_value) -- a fixed direction
    times a raw traction value, which is a valid approximation on B1's
    FLAT loaded edge. B2's loaded edge is CURVED, so the corresponding
    "raw pressure value x exact radial direction" approximation (no
    edge-length/quadrature weighting) is NOT force-consistent with the
    FEM solve that produced the ground-truth displacements --
    data_generate_B2.py's own solver instead loads the inner arc via
    assemble_traction_inner_curved (integral(N_a * p * normal) ds,
    properly quadrature-weighted). check_boundary_force_consistency.py
    confirmed the raw approximation is off from that consistent force by
    a factor of ~4.4x in magnitude (direction agrees, cosine~0.9975) --
    exactly the kind of boundary-localized, direction-specific,
    load-magnitude-independent bias accuracy_diagnostics.py's error
    breakdown found (radial >> circumferential, inner boundary > interior
    > outer). So this now reads the FEM-consistent nodal force
    data_generate_B2.py saves directly (inner_force_consistent) instead
    of recomputing the inconsistent approximation.
"""
import numpy as np
import h5py
import os
import argparse


def convert_h5_to_training_format_q4(h5_path, mesh_info_path, output_path, num_samples=None):
    print(f"Converting {h5_path} to training format (Q4 mesh, B2 ring)...")

    with h5py.File(h5_path, 'r') as f:
        num_samples_total = f.attrs['num_samples']
        n_nodes = f.attrs['n_nodes']
        R_in = f.attrs['R_in']
        R_out = f.attrs['R_out']

        mesh_info = np.load(mesh_info_path)
        nodes = mesh_info['nodes']       # (N, 2)
        elements = mesh_info['elements']  # (M, 4) Q4 elements

        if num_samples is None:
            num_samples = num_samples_total

        valid_mask = ~np.isnan(f['displacements'][:, 0, 0])
        valid_indices = np.where(valid_mask)[0]
        num_valid = len(valid_indices)

        if num_valid < num_samples:
            print(f"Warning: Only {num_valid} valid samples available, requested {num_samples}")
            num_samples = num_valid

        sample_indices = valid_indices[:num_samples]

        print(f"Total samples: {num_samples_total}")
        print(f"Valid samples: {num_valid}")
        print(f"Using samples: {num_samples}")
        print(f"Nodes: {n_nodes}")
        print(f"Domain: quarter ring R_in={R_in}, R_out={R_out}")
        print(f"Elements: {len(elements)} Q4 elements")

        tol = 1e-9
        r_all = np.linalg.norm(nodes, axis=1)
        theta0_nodes_original = np.where(np.abs(nodes[:, 1]) < tol)[0]
        thetahalfpi_nodes_original = np.where(np.abs(nodes[:, 0]) < tol)[0]
        inner_nodes_original = np.where(np.abs(r_all - R_in) < tol)[0]

        # Inner-arc line segments: local edge (n4, n1), i.e. xi=-1 side of a
        # Q4 (see data_generate_B2.py -- r is the fast local index, so the
        # arc at fixed innermost r runs along n4->n1, not n1->n2)
        inner_edges = []
        for elem in elements:
            n1, n2, n3, n4 = elem
            if abs(r_all[n1] - R_in) < tol and abs(r_all[n4] - R_in) < tol:
                inner_edges.append([n4, n1])
        inner_edges = np.array(inner_edges, dtype=np.int64)

        print(f"theta=0 symmetry nodes (u_y=0): {len(theta0_nodes_original)}")
        print(f"theta=pi/2 symmetry nodes (u_x=0): {len(thetahalfpi_nodes_original)}")
        print(f"Inner boundary nodes: {len(inner_nodes_original)}")
        print(f"Inner boundary edges: {len(inner_edges)}")

        if "inner_force_consistent" not in f:
            raise KeyError(
                "This H5 dataset was generated before data_generate_B2.py started saving "
                "'inner_force_consistent' (the FEM-consistent nodal force the ground-truth "
                "displacements were actually solved with -- see this module's docstring and "
                "check_boundary_force_consistency.py). Regenerate the dataset with the current "
                "data_generate_B2.py before converting; the old raw-pressure approximation this "
                "converter used to fall back to was confirmed off by ~4.4x in magnitude."
            )

        coords_list, quads_list, disp2Ds_list = [], [], []
        inner_edges_list, theta0_nodes_list, thetahalfpi_nodes_list = [], [], []
        E_node_list, nu_node_list, boundary_info_list = [], [], []

        for idx, sample_idx in enumerate(sample_indices):
            if idx % 10 == 0:
                print(f"Processing sample {idx+1}/{num_samples} (original index {sample_idx})")

            E_node = f['E_nodes'][sample_idx]
            nu_node = f['nu_nodes'][sample_idx]
            displacement = f['displacements'][sample_idx]

            boundary_forces = f['inner_force_consistent'][sample_idx].astype(np.float32)

            boundary_info = {
                'node_indices': inner_nodes_original.astype(np.int64),
                'force_vectors': boundary_forces,
                'coordinates': nodes[inner_nodes_original].astype(np.float32)
            }

            coords_list.append(nodes.astype(np.float32))
            quads_list.append(elements.astype(np.int64))
            disp2Ds_list.append(displacement.astype(np.float32))
            inner_edges_list.append(inner_edges.astype(np.int64))
            theta0_nodes_list.append(theta0_nodes_original.astype(np.int64))
            thetahalfpi_nodes_list.append(thetahalfpi_nodes_original.astype(np.int64))
            E_node_list.append(E_node.astype(np.float32))
            nu_node_list.append(nu_node.astype(np.float32))
            boundary_info_list.append(boundary_info)

        coords_array = np.array(coords_list, dtype=object)
        quads_array = np.array(quads_list, dtype=object)
        disp2Ds_array = np.array(disp2Ds_list, dtype=object)
        inner_edges_array = np.array(inner_edges_list, dtype=object)
        theta0_nodes_array = np.array(theta0_nodes_list, dtype=object)
        thetahalfpi_nodes_array = np.array(thetahalfpi_nodes_list, dtype=object)
        E_node_array = np.array(E_node_list, dtype=object)
        nu_node_array = np.array(nu_node_list, dtype=object)
        boundary_info_array = np.array(boundary_info_list, dtype=object)

        print(f"\nSaving to {output_path}...")
        np.savez(
            output_path,
            coord=coords_array,
            quad=quads_array,
            disp2D=disp2Ds_array,
            inner_edges=inner_edges_array,
            theta0_nodes=theta0_nodes_array,
            thetahalfpi_nodes=thetahalfpi_nodes_array,
            E_node=E_node_array,
            nu_node=nu_node_array,
            boundary_info=boundary_info_array,
            R_in=np.float32(R_in),
            R_out=np.float32(R_out),
        )
        print(f"Successfully saved {num_samples} samples to {output_path}")
        return coords_array, quads_array, disp2Ds_array


def main():
    parser = argparse.ArgumentParser("Convert B2 HDF5 dataset to Q4 training NPZ format")
    parser.add_argument("--h5_dir", type=str, default="physics_training_data_B2_1",
                        help="Directory produced by data_generate_B2.py")
    parser.add_argument("--out_dir", type=str, default="./training_data_B2_q4")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    h5_path = os.path.join(args.h5_dir, "hyperelastic_dataset_physics.h5")
    mesh_info_path = os.path.join(args.h5_dir, "mesh_info.npz")
    os.makedirs(args.out_dir, exist_ok=True)
    output_npz = os.path.join(args.out_dir, "hyperelastic_training_data_q4.npz")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found at {h5_path}. Run data_generate_B2.py first.")
    if not os.path.exists(mesh_info_path):
        raise FileNotFoundError(f"Mesh info file not found at {mesh_info_path}.")

    convert_h5_to_training_format_q4(
        h5_path=h5_path,
        mesh_info_path=mesh_info_path,
        output_path=output_npz,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
