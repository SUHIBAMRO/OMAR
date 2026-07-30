"""
Standalone inference-latency measurement for an ALREADY-TRAINED checkpoint
-- no retraining, no risk to the case's real results.

Why this exists: benchmark_inference_latency_Q4 is normally called at the
very end of train_hyperelastic_Q4() (train_B1.py/train_B2.py), right after
training finishes. But that function returns EARLY -- before ever reaching
the benchmark call -- whenever the case is already fully trained
(model_final.pt or EARLY_STOPPED already present). That early return is
exactly the resume/skip behavior needed to avoid wasting GPU time
re-training a finished case -- but its side effect is that for any case
that was ALREADY fully trained before benchmark_inference_latency_Q4
existed in the codebase, simply re-running train_B{1,2}.py never reaches
the new benchmark call, so inference_latency.json is silently never
produced for that case. (The same early-return also skips the
device-level GPU memory instrumentation, which genuinely does need a real
training pass to measure -- see the Colab notebook for the short,
throwaway "memory profiling" run that covers that one instead.)

This script re-derives exactly the same inference-latency measurement,
standalone, from an existing checkpoint. Writes inference_latency.json to
the SAME path train_hyperelastic_Q4() would have written it to, so
write_case_manifest() picks it up identically either way.

Usage:
  python -m omar_pfem.measure_inference_latency \
      --geometry B1 --material neo_hookean \
      --checkpoint /path/to/B1_neo_hookean/model_best.pt \
      --dataset /path/to/B1_neo_hookean/hyperelastic_training_data_q4.npz \
      --ntrain 800 --ntest 200 \
      --out_json /path/to/B1_neo_hookean/inference_latency.json
"""
import argparse
import json
import torch

from omar_pfem.model_dict import get_model


def build_model(args, device):
    model = get_model(args).Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=args.mlp_ratio,
        fun_dim=args.fun_dim,
        out_dim=2,
        slice_num=args.slice_num,
        ref=args.ref,
        unified_pos=args.unified_pos,
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser(
        "Standalone inference-latency benchmark for an existing checkpoint (no retraining)."
    )
    parser.add_argument("--geometry", type=str, required=True, choices=["B1", "B2"])
    parser.add_argument("--material", type=str, required=True,
                         choices=["neo_hookean", "mooney_rivlin", "arruda_boyce"])
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to model_best.pt or model_final.pt from a normal training run")
    parser.add_argument("--dataset", type=str, required=True,
                         help="The case's converted NPZ (same one it was trained on)")
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--ntest", type=int, default=200)

    # Architecture -- must match the checkpoint's training run exactly (Table 1 defaults).
    parser.add_argument("--model", type=str, default="Transolver_Irregular_Mesh")
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unified_pos", type=int, default=0)
    parser.add_argument("--ref", type=int, default=16)
    parser.add_argument("--slice_num", type=int, default=128)
    parser.add_argument("--fun_dim", type=int, default=4)

    # Physics/BC settings -- must match training (defaults match train_B1.py/train_B2.py's own).
    parser.add_argument("--use_soft_dirichlet", type=int, default=1)
    parser.add_argument("--Lx", type=float, default=1.0)
    parser.add_argument("--Ly", type=float, default=1.0)
    parser.add_argument("--R_out", type=float, default=2.0)

    parser.add_argument("--n_repeats", type=int, default=200)
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32

    if args.geometry == "B1":
        from omar_pfem.train_B1 import (
            benchmark_inference_latency_Q4, load_fem_dataset_Q4_with_materials_and_random_force,
        )
    else:
        from omar_pfem.train_B2 import (
            benchmark_inference_latency_Q4, load_fem_dataset_Q4_with_materials_and_random_force,
        )

    model = build_model(args, device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    _, Test_samples = load_fem_dataset_Q4_with_materials_and_random_force(
        args.dataset, args.ntrain, args.ntest
    )
    print(f"Test samples: {len(Test_samples)}")

    inference_stats = benchmark_inference_latency_Q4(
        Test_samples, model, args, device, dtype,
        n_repeats=args.n_repeats, n_warmup=args.n_warmup,
    )
    print(f"Inference latency: {inference_stats['inference_ms_per_sample']:.4f} ms/sample "
          f"(batch_size=1, {args.n_repeats} repeats, device={inference_stats['inference_device']})")

    with open(args.out_json, "w") as f:
        json.dump(inference_stats, f, indent=2)
    print(f"Written to {args.out_json}")


if __name__ == "__main__":
    main()
