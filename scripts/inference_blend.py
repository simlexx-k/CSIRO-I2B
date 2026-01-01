#!/usr/bin/env python3
"""Blend precomputed pillar submissions into a single submission.csv.

Each input CSV must contain the Kaggle-required columns: `sample_id`, `target`.
By default the script blends SigLIP/DINO/MVP/Dinov2 outputs with weights
0.60/0.20/0.10/0.10, but any four positive weights are accepted (they will be
normalized automatically). Optionally clip the blended targets to a min/max.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REQUIRED_COLS = ("sample_id", "target")


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required column(s): {missing}")
    return df[REQUIRED_COLS].copy()


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (4,):
        raise ValueError("Exactly four weights are required.")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    total = weights.sum()
    if total == 0:
        raise ValueError("Weights sum to zero; provide at least one positive weight.")
    return weights / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend pillar submissions into a final submission.csv"
    )
    parser.add_argument("--siglip", type=Path, required=True, help="SigLIP submission CSV path")
    parser.add_argument("--dino", type=Path, required=True, help="DINO submission CSV path")
    parser.add_argument("--mvp", type=Path, required=True, help="MVP submission CSV path")
    parser.add_argument("--dinov2", type=Path, required=True, help="Dinov2 submission CSV path")
    parser.add_argument(
        "--weights",
        type=float,
        nargs=4,
        default=(0.60, 0.20, 0.10, 0.10),
        metavar=("W_SIGLIP", "W_DINO", "W_MVP", "W_DINOV2"),
        help="Blend weights (will be normalized); default 0.60/0.20/0.10/0.10",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Optional lower bound applied to blended targets",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Optional upper bound applied to blended targets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Output submission path (default submission.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = normalize_weights(args.weights)

    siglip_df = load_submission(args.siglip)
    dino_df = load_submission(args.dino)
    mvp_df = load_submission(args.mvp)
    dinov2_df = load_submission(args.dinov2)

    # Ensure ordering matches across pillars.
    id_column = siglip_df["sample_id"]
    for name, pillar_df in (
        ("DINO", dino_df),
        ("MVP", mvp_df),
        ("Dinov2", dinov2_df),
    ):
        if not pillar_df["sample_id"].equals(id_column):
            raise ValueError(f"{name} submission sample_id ordering does not match SigLIP.")

    stacked = np.stack(
        [
            siglip_df["target"].to_numpy(dtype=np.float64),
            dino_df["target"].to_numpy(dtype=np.float64),
            mvp_df["target"].to_numpy(dtype=np.float64),
            dinov2_df["target"].to_numpy(dtype=np.float64),
        ],
        axis=0,
    )
    blended = np.tensordot(weights, stacked, axes=(0, 0))

    if args.clip_min is not None or args.clip_max is not None:
        blended = np.clip(blended, args.clip_min, args.clip_max)

    out_df = pd.DataFrame({"sample_id": id_column, "target": blended})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(
        f"Saved blended submission to {args.output} with weights {weights.round(4).tolist()} "
        f"and {len(out_df):,} rows."
    )


if __name__ == "__main__":
    main()
