#!/usr/bin/env python3
"""Score a trained baseline checkpoint on a shard corpus it never trained on.

The trainer already knows how to build the model, run a forward pass and compute
open-loop metrics; this only points those at a different corpus and a saved
checkpoint, with split="all" so the whole corpus is scored rather than a holdout
slice of it.

    python3 evaluation/eval_checkpoint.py \
        --checkpoint runs/edgar_autonomous/best.pt \
        --shard-dir /path/to/shards
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_parsing.pre_extracted import make_pre_extracted_loader
from model_components.auto_e2e import AutoE2E
from training.train_baseline import evaluate, policy_from_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", help="write metrics JSON here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    manifest = json.loads((Path(args.shard_dir) / "manifest.json").read_text())
    num_views = int(manifest["num_views"])
    if num_views != int(ckpt["num_views"]):
        raise SystemExit(f"checkpoint has V={ckpt['num_views']}, corpus has V={num_views}")

    # Rebuild with the checkpoint's own geometry, not the corpus's: the BEV grid
    # is baked into the weight shapes. is_pretrained=False because the saved
    # weights are about to overwrite everything anyway.
    model = AutoE2E(backbone=ckpt.get("backbone", "swin_v2_tiny"),
                    num_views=num_views, embed_dim=256, is_pretrained=False,
                    view_fusion_kwargs=ckpt.get("view_fusion_kwargs") or {},
                    map_context_channels=int(manifest.get("map_context_channels", 3)),
                    route_channels=int(manifest.get("route_channels", 2))).to(device)
    model.load_state_dict(ckpt["model"])

    policy = policy_from_manifest(manifest)
    loader = make_pre_extracted_loader(
        args.shard_dir, split="all", shuffle=0, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    metrics = evaluate(model, loader, device,
                       getattr(loader, "projection", None),
                       manifest.get("geometry_type", "pseudo"))

    z = metrics["reference_zero_prediction"]
    print(f"checkpoint : {args.checkpoint} (epoch {ckpt.get('epoch')})")
    print(f"corpus     : {args.shard_dir}  {metrics['samples']} samples, "
          f"groups {manifest['split_groups']}")
    print(f"policy     : signal_scales={policy.signal_scales}")
    for k in ("ADE@1s", "ADE@2s", "ADE@3s", "FDE@3s", "accel_mae", "curvature_mae"):
        if k in metrics:
            print(f"  {k:14s} {metrics[k]:.3f}   zero-pred {z[k]:.3f}   "
                  f"{100 * metrics[k] / z[k]:.0f}% of reference")
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
