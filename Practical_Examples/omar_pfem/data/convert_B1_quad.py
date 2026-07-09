"""
Converts the B1 HDF5 dataset (data_generate_B1.py) into the Q4 NPZ format
expected by the training script -- adapted from PFEM-main's
data/hyper/data_generate_beam_convert_quad.py, with the Dirichlet/traction
edges swapped from left/right to bottom/top to match B1's geometry.
"""
import numpy as np
import h5py
import os
import argparse


def convert_h5_to_training_format_q4(h5_path, mesh_info_path, output_path, num_samples=None):
    print(f"Converting {h5_path} to training format (Q4 mesh)...")

    with h5py.File(h5_path, 'r') as f:
        num_samples_total = f.attrs['num_samples']
        n_nodes = f.attrs['n_nodes']
        Lx = f.attrs['Lx']
        Ly = f.attrs['Ly']

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
        print(f"Domain: {Lx}x{Ly}")
        print(f"Elements: {len(elements)} Q4 elements")

        tol = 1e-12
        bottom_nodes_original = np.where(nodes[:, 1] < tol)[0]
        top_nodes_original = np.where(np.abs(nodes[:, 1] - Ly) < tol)[0]

        # Top-edge line segments (local nodes n3, n4 -- eta=+1 side of a Q4)
        top_edges = []
        for elem in elements:
            n1, n2, n3, n4 = elem
            if nodes[n3, 1] > Ly - tol and nodes[n4, 1] > Ly - tol:
                top_edges.append([n3, n4])
        top_edges = np.array(top_edges, dtype=np.int64)

        print(f"Bottom boundary nodes: {len(bottom_nodes_original)}")
        print(f"Top boundary nodes: {len(top_nodes_original)}")
        print(f"Top boundary edges: {len(top_edges)}")

        coords_list, quads_list, disp2Ds_list = [], [], []
        top_edges_list, bottom_nodes_list = [], []
        E_node_list, nu_node_list, boundary_info_list = [], [], []

        for idx, sample_idx in enumerate(sample_indices):
            if idx % 10 == 0:
                print(f"Processing sample {idx+1}/{num_samples} (original index {sample_idx})")

            E_node = f['E_nodes'][sample_idx]
            nu_node = f['nu_nodes'][sample_idx]
            displacement = f['displacements'][sample_idx]
            top_traction = f['top_traction'][sample_idx]

            boundary_forces = np.zeros((len(top_nodes_original), 2), dtype=np.float32)
            boundary_forces[:, 1] = top_traction  # y-direction traction on the top edge

            boundary_info = {
                'node_indices': top_nodes_original.astype(np.int64),
                'force_vectors': boundary_forces.astype(np.float32),
                'coordinates': nodes[top_nodes_original].astype(np.float32)
            }

            coords_list.append(nodes.astype(np.float32))
            quads_list.append(elements.astype(np.int64))
            disp2Ds_list.append(displacement.astype(np.float32))
            top_edges_list.append(top_edges.astype(np.int64))
            bottom_nodes_list.append(bottom_nodes_original.astype(np.int64))
            E_node_list.append(E_node.astype(np.float32))
            nu_node_list.append(nu_node.astype(np.float32))
            boundary_info_list.append(boundary_info)

        coords_array = np.array(coords_list, dtype=object)
        quads_array = np.array(quads_list, dtype=object)
        disp2Ds_array = np.array(disp2Ds_list, dtype=object)
        top_edges_array = np.array(top_edges_list, dtype=object)
        bottom_nodes_array = np.array(bottom_nodes_list, dtype=object)
        E_node_array = np.array(E_node_list, dtype=object)
        nu_node_array = np.array(nu_node_list, dtype=object)
        boundary_info_array = np.array(boundary_info_list, dtype=object)

        print(f"\nSaving to {output_path}...")
        np.savez(
            output_path,
            coord=coords_array,
            quad=quads_array,
            disp2D=disp2Ds_array,
            top_edges=top_edges_array,
            bottom_nodes=bottom_nodes_array,
            E_node=E_node_array,
            nu_node=nu_node_array,
            boundary_info=boundary_info_array
        )
        print(f"Successfully saved {num_samples} samples to {output_path}")
        return coords_array, quads_array, disp2Ds_array


def main():
    parser = argparse.ArgumentParser("Convert B1 HDF5 dataset to Q4 training NPZ format")
    parser.add_argument("--h5_dir", type=str, default="physics_training_data_B1_1",
                        help="Directory produced by data_generate_B1.py")
    parser.add_argument("--out_dir", type=str, default="./training_data_B1_q4")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    h5_path = os.path.join(args.h5_dir, "hyperelastic_dataset_physics.h5")
    mesh_info_path = os.path.join(args.h5_dir, "mesh_info.npz")
    os.makedirs(args.out_dir, exist_ok=True)
    output_npz = os.path.join(args.out_dir, "hyperelastic_training_data_q4.npz")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found at {h5_path}. Run data_generate_B1.py first.")
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
