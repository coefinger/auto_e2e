#!/usr/bin/env python3
"""Single-GPU trajectory-imitation baseline on EDGAR rosbag shards.

Pairs with ``data_parsing/edgar/extract_shards.py``. Run from ``Model/`` with
the torch venv (needs torch, timm, webdataset):

    python -m training.train_baseline --shard-dir /data/edgar_shards --epochs 10

Imitation loss only — no map, route, reasoning or world-model branch, so the
number it reports is the camera+egomotion baseline the other configurations get
compared against. Validation is a whole-bag holdout (never adjacent frames),
and a zero-prediction reference is scored on the same split so the model's
ADE/FDE can be read against "do nothing".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from data_parsing.pre_extracted import make_pre_extracted_loader
from evaluation.metrics import compute_open_loop_metrics
from model_components.auto_e2e import AutoE2E
from model_components.losses import TrajectoryImitationLoss

from training.dataset_policy import DatasetTrainingPolicy

TIMESTEPS = 64
SIGNALS = 2


def holdout_groups(groups: list[str], fraction: float) -> list[str]:
    """Deterministic whole-bag validation holdout, independent of bag order."""
    from data_processing.dataset_snapshot import split_bucket

    ranked = sorted(set(groups), key=lambda g: (split_bucket(g, 1000), g))
    count = max(1, min(len(ranked) - 1, round(fraction * len(ranked))))
    return sorted(ranked[:count])


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int,
                       floor: float) -> float:
    """Multiplier on the base lr: linear warmup, then cosine decay to `floor`.

    `step` past `total_steps` clamps at the floor rather than climbing back up,
    which matters because the trailing partial accumulation window adds an extra
    optimizer step to some epochs.
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return floor + (1.0 - floor) * cosine


def train_steps_per_epoch(manifest: dict, val_groups, batch_size: int,
                          accum_steps: int) -> int:
    """Optimizer steps in one training epoch, from the manifest's own counts."""
    sources = manifest.get("source_bags") or []
    held = set(val_groups)
    if sources and all("split_group_uid" in s and "samples" in s for s in sources):
        train_samples = sum(int(s["samples"]) for s in sources
                            if s["split_group_uid"] not in held)
    else:
        # Older manifests predate source_bags; fall back to the whole corpus,
        # which overestimates and so anneals slightly too slowly. Never zero.
        train_samples = int(manifest.get("total_samples", 0))
    return max(1, math.ceil(math.ceil(max(1, train_samples) / batch_size) / accum_steps))


def policy_from_manifest(manifest: dict) -> DatasetTrainingPolicy:
    measured = manifest.get("measured_signal_scales") or [0.79, 0.12]
    return DatasetTrainingPolicy(
        dataset_name=str(manifest.get("dataset", "TUM-EDGAR/rosbags")),
        temporal_decay=0.95,
        # Corpus-measured RMS, floored so a near-constant signal cannot blow the
        # loss up. Never inherit L2D's scales silently.
        signal_scales=(max(float(measured[0]), 1e-3), max(float(measured[1]), 1e-3)),
    )


def forward(model, batch, device, projection, geometry_type, target=None):
    visual = batch["visual_tiles"].to(device, non_blocking=True)
    out = model(
        visual,
        batch["map_context"].to(device, non_blocking=True),
        batch["visual_history"].to(device, non_blocking=True),
        batch["egomotion_history"].to(device, non_blocking=True),
        route_mask=batch["route_mask"].to(device, non_blocking=True),
        map_valid=batch["map_valid"].to(device, non_blocking=True),
        route_valid=batch["route_valid"].to(device, non_blocking=True),
        projection=projection,
        geometry_type=geometry_type,
        mode="train" if target is not None else "infer",
        trajectory_target=target,
    )
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def evaluate(model, loader, device, projection, geometry_type):
    """Open-loop ADE/FDE on the holdout, plus the zero-prediction reference."""
    model.eval()
    preds, targets, speeds = [], [], []
    for batch in loader:
        pred = forward(model, batch, device, projection, geometry_type)
        preds.append(pred.float().cpu().numpy())
        targets.append(batch["trajectory_target"].numpy())
        history = batch["egomotion_history"].numpy().reshape(-1, TIMESTEPS, 4)
        speeds.append(history[:, -1, 0])  # speed at the newest history step
    if not preds:
        raise SystemExit("validation split is empty — check the holdout groups")
    pred = np.concatenate(preds).reshape(-1, TIMESTEPS, SIGNALS)
    gt = np.concatenate(targets).reshape(-1, TIMESTEPS, SIGNALS)
    v0 = np.concatenate(speeds)
    metrics = compute_open_loop_metrics(
        pred[..., 0], pred[..., 1], gt[..., 0], gt[..., 1], v0)
    zeros = np.zeros_like(gt[..., 0])
    metrics["reference_zero_prediction"] = compute_open_loop_metrics(
        zeros, zeros, gt[..., 0], gt[..., 1], v0)
    metrics["samples"] = int(gt.shape[0])
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", default="runs/edgar_baseline")
    ap.add_argument("--backbone", default="swin_v2_tiny")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="samples per GPU step; ~7 GiB each with 6 views at 256px")
    ap.add_argument("--accum-steps", type=int, default=4,
                    help="optimizer steps every N batches (effective batch = "
                         "batch-size * accum-steps)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", choices=("cosine", "none"), default="cosine",
                    help="cosine anneals lr to --lr-min over the run, with a "
                         "one-epoch linear warmup. 'none' is the old fixed lr, "
                         "kept to reproduce earlier runs.")
    ap.add_argument("--lr-min", type=float, default=1e-6,
                    help="floor for the cosine schedule")
    ap.add_argument("--warmup-epochs", type=float, default=1.0,
                    help="linear warmup length; 0 disables it")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--val-groups", nargs="+",
                    help="split_group_uids to hold out, instead of the "
                         "hash-ranked --val-fraction pick. Use when the corpus "
                         "has few bags and which one validates is a judgement "
                         "call rather than a coin flip.")
    ap.add_argument("--bev-size", type=int, default=8,
                    help="BEV fusion grid (bev_h=bev_w). BEVViewFusion defaults "
                         "to 450x300 = 135k queries, whose grid_sample backward "
                         "is ~25x the rest of the step; 8 matches the grid the "
                         "repo's own speed benchmark reports.")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    if args.val_groups:
        known = set(manifest["split_groups"])
        unknown = sorted(set(args.val_groups) - known)
        if unknown:
            raise SystemExit(f"--val-groups not in the corpus: {unknown}\n"
                             f"available: {sorted(known)}")
        if set(args.val_groups) >= known:
            raise SystemExit("--val-groups holds out every bag; nothing to train on")
        val_groups = sorted(args.val_groups)
    else:
        val_groups = holdout_groups(manifest["split_groups"], args.val_fraction)
    policy = policy_from_manifest(manifest)
    num_views = int(manifest["num_views"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] {manifest['total_samples']} samples, V={num_views}, "
          f"device={device}, amp={amp}")
    print(f"[baseline] signal_scales={policy.signal_scales}")
    print(f"[baseline] bev grid {args.bev_size}x{args.bev_size}, "
          f"batch {args.batch_size} x accum {args.accum_steps}")
    print(f"[baseline] holdout bags ({len(val_groups)}/"
          f"{len(manifest['split_groups'])}): {val_groups}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_fraction": args.val_fraction,
        "validation_group_uids": val_groups,
        "pin_memory": device.type == "cuda",
    }
    train_loader = make_pre_extracted_loader(
        str(shard_dir), split="train", shuffle=1000, **loader_kwargs)
    val_loader = make_pre_extracted_loader(
        str(shard_dir), split="val", shuffle=0, **loader_kwargs)
    projection = getattr(train_loader, "projection", None)
    geometry_type = getattr(train_loader, "geometry_type", "pseudo")
    if projection is not None:
        projection = projection.to(device)

    bev_kwargs = {"bev_h": args.bev_size, "bev_w": args.bev_size}
    model = AutoE2E(backbone=args.backbone, num_views=num_views, embed_dim=256,
                    is_pretrained=not args.no_pretrained,
                    view_fusion_kwargs=bev_kwargs,
                    map_context_channels=int(manifest.get("map_context_channels", 3)),
                    route_channels=int(manifest.get("route_channels", 2))).to(device)
    loss_fn = TrajectoryImitationLoss(
        loss_type="smooth_l1",
        temporal_decay=policy.temporal_decay,
        temporal_weight_normalization=policy.temporal_weight_normalization,
        signal_scales=policy.signal_scales,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Per-optimizer-step schedule, not per-epoch: with ~1400 batches an epoch
    # the cosine is smooth, and warmup only means anything at step granularity.
    # The count comes from the manifest, not len(train_loader) -- a WebLoader has
    # no __len__, and guessing 1 would collapse the cosine inside one epoch.
    steps_per_epoch = train_steps_per_epoch(
        manifest, val_groups, args.batch_size, args.accum_steps)
    scheduler = None
    if args.lr_schedule == "cosine":
        total_steps = max(1, steps_per_epoch * args.epochs)
        warmup_steps = min(int(steps_per_epoch * args.warmup_epochs), total_steps - 1)

        def lr_at(step: int) -> float:
            return cosine_with_warmup(step, warmup_steps, total_steps,
                                      args.lr_min / args.lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
        print(f"[baseline] lr {args.lr:g} -> {args.lr_min:g} cosine over "
              f"{total_steps} steps ({steps_per_epoch}/epoch), "
              f"{warmup_steps} warmup", flush=True)
    else:
        print(f"[baseline] lr fixed at {args.lr:g}", flush=True)

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            target = batch["trajectory_target"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred = forward(model, batch, device, projection, geometry_type,
                               target=target)
                loss = loss_fn(pred.float(), target)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch} step {steps}")
            # Scale so the accumulated gradient is the mean over the effective
            # batch, not its sum.
            (loss / args.accum_steps).backward()
            total += float(loss.item())
            steps += 1
            if steps % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if steps % 100 == 0:
                print(f"  epoch {epoch} step {steps} loss {total / steps:.4f}", flush=True)
        # Flush a trailing partial accumulation window.
        if steps % args.accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss = total / max(steps, 1)
        metrics = evaluate(model, val_loader, device, projection, geometry_type)
        record = {"epoch": epoch, "train_loss": train_loss, "train_steps": steps,
                  **{k: v for k, v in metrics.items() if k != "reference_zero_prediction"}}
        history.append(record)
        print(f"[baseline] epoch {epoch}: train_loss={train_loss:.4f} "
              f"ADE@3s={metrics['ADE@3s']:.3f} FDE@3s={metrics['FDE@3s']:.3f} "
              f"accel_mae={metrics['accel_mae']:.3f} "
              f"curv_mae={metrics['curvature_mae']:.4f}", flush=True)

        if metrics["ADE@3s"] < best:
            best = metrics["ADE@3s"]
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics,
                        "num_views": num_views,
                        "backbone": args.backbone,
                        "view_fusion_kwargs": bev_kwargs,
                        "trajectory_training_policy": policy.metadata()},
                       out_dir / "best.pt")
        (out_dir / "metrics.json").write_text(json.dumps(
            {"manifest": manifest, "config": vars(args),
             "holdout_groups": val_groups,
             "policy": policy.metadata(),
             "zero_prediction_reference": metrics["reference_zero_prediction"],
             "epochs": history, "best_ADE@3s": best}, indent=2))

    ref = history and metrics["reference_zero_prediction"]
    print(f"\n[baseline] best ADE@3s={best:.3f} "
          f"(zero-prediction reference ADE@3s={ref['ADE@3s']:.3f})")
    print(f"[baseline] wrote {out_dir / 'metrics.json'} and {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
