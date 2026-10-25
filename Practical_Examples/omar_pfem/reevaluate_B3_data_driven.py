"""Re-evaluate an EXISTING B3 data-driven checkpoint (model_best.pt from
train_B3_data_driven.py) against the held-out set, WITHOUT retraining --
added 2026-10-26 specifically to add the H1 semi-norm metric (Professor
Rabczuk's 2026-10-20 email named it explicitly; train_B3_data_driven.py's
own evaluate() did not compute it when the rel_l2/sobolev checkpoints
were originally trained). No training dataset is needed here at all,
unlike train_B3_data_driven.py's own main() -- only the checkpoint and
the held-out set.

Usage:
  python -m omar_pfem.reevaluate_B3_data_driven \
      --checkpoint /content/drive/MyDrive/pfem_run/b3_data_driven_rel_l2/model_best.pt \
      --test_dataset /content/drive/MyDrive/pfem_run/b3_dataset_clean_holdout/dataset.h5 \
      --out_json /content/drive/MyDrive/pfem_run/b3_data_driven_rel_l2/qois_h1_reeval.json
"""
import argparse
import json

import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.train_B1 import install_input_norm_for_checkpoint, set_input_norm
from omar_pfem.train_B3 import build_fixed_geometry, build_model
from omar_pfem.train_B3_data_driven import evaluate, load_b3_h5


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_dataset", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resolution", type=int, nargs=3, default=None, metavar=("NTHETA", "NR", "NZ"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64

    Ntheta, Nr, Nz = tuple(args.resolution) if args.resolution else DEFAULT_RESOLUTION
    geom = build_fixed_geometry(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"Geometry: {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    print(f"Loading checkpoint: {args.checkpoint}")
    install_input_norm_for_checkpoint(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    normalize_inputs = int(getattr(ckpt_args, "normalize_inputs", 0))
    if not normalize_inputs:
        set_input_norm(None)
    model = build_model(ckpt_args, device).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loading held-out test set: {args.test_dataset}")
    test = load_b3_h5(args.test_dataset)
    print(f"  {test['E_node'].shape[0]} samples")

    metrics = evaluate(test, model, geom, device, dtype, normalize_inputs)
    print("\n" + "=" * 70)
    print(f"RE-EVALUATION of {args.checkpoint} (loss={getattr(ckpt_args, 'loss', '?')}, "
          f"step={ckpt.get('step', '?')})")
    print(f"  mean rel. L2            : {metrics['mean_rel_L2']:.4f}")
    print(f"  median rel. L2          : {metrics['median_rel_L2']:.4f}")
    print(f"  mean H1 semi-norm rel.  : {metrics['mean_H1_seminorm_rel']:.4f}")
    print(f"  median H1 semi-norm rel.: {metrics['median_H1_seminorm_rel']:.4f}")
    print("=" * 70)

    with open(args.out_json, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "checkpoint_step": ckpt.get("step"),
                   "loss_variant": getattr(ckpt_args, "loss", None),
                   "test_dataset": args.test_dataset, "metrics": metrics}, f, indent=2)
    print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
