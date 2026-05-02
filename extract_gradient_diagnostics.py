"""
extract_gradient_diagnostics.py — Gradient analysis across all seeds
=====================================================================
Reads epoch_diag_fold_*.csv from each seed's phase3 folder and aggregates
gradient cosine (aff vs spec) across seeds, folds, and feature types.

USAGE:
    python extract_gradient_diagnostics.py

OUTPUT:
    outputs/multiseed/gradient_cosine.csv            (aggregated per epoch)
    outputs/multiseed/gradient_cosine_raw.csv         (all folds, all seeds)
    outputs/multiseed/gradient_cosine_summary.csv     (per-feature summary stats)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> int:
    output_dir = Path("outputs").resolve()
    seeds_root = output_dir / "seeds"

    seeds = sorted([int(p.name.split("_")[1]) for p in seeds_root.iterdir()
                    if p.is_dir() and p.name.startswith("seed_")])

    print(f"Extracting gradient diagnostics from {len(seeds)} seeds: {seeds}")

    all_rows = []

    for seed in seeds:
        phase3_dir = seeds_root / f"seed_{seed}" / "phase3"
        if not phase3_dir.exists():
            print(f"  [seed {seed}] phase3 not found, skipping")
            continue

        csvs = sorted(phase3_dir.glob("epoch_diag_fold_*.csv"))
        if not csvs:
            print(f"  [seed {seed}] no epoch_diag CSVs found, skipping")
            continue

        for csv_path in csvs:
            # Parse filename: epoch_diag_fold_1_OneHot.csv
            name = csv_path.stem  # epoch_diag_fold_1_OneHot
            parts = name.replace("epoch_diag_fold_", "").split("_", 1)
            fold = int(parts[0])
            feature = parts[1] if len(parts) > 1 else "unknown"

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"  [seed {seed}] error reading {csv_path.name}: {e}")
                continue

            for _, row in df.iterrows():
                entry = {
                    'seed': seed,
                    'fold': fold,
                    'feature': feature,
                    'epoch': int(row.get('epoch', 0)),
                }
                # Grab all numeric columns
                for col in df.columns:
                    if col != 'epoch':
                        val = row.get(col)
                        if pd.notna(val):
                            entry[col] = float(val)
                all_rows.append(entry)

        print(f"  [seed {seed}] {len(csvs)} CSVs, "
              f"{len([r for r in all_rows if r['seed'] == seed])} epoch entries")

    if not all_rows:
        print("No data extracted!")
        return 1

    df = pd.DataFrame(all_rows)

    out_dir = output_dir / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw
    raw_path = out_dir / "gradient_cosine_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path} ({len(df)} rows)")

    # Aggregate: mean gradient cosine per epoch per feature (across folds and seeds)
    if 'grad_cosine_aff_spec' in df.columns:
        agg = df.groupby(['feature', 'epoch']).agg(
            grad_cos_mean=('grad_cosine_aff_spec', 'mean'),
            grad_cos_std=('grad_cosine_aff_spec', 'std'),
            grad_cos_min=('grad_cosine_aff_spec', 'min'),
            grad_cos_max=('grad_cosine_aff_spec', 'max'),
            n=('grad_cosine_aff_spec', 'count'),
            acc_aff_mean=('acc_aff', 'mean'),
            acc_spec_mean=('acc_spec', 'mean'),
            aff_gap_mean=('aff_gap', 'mean'),
            spec_gap_mean=('spec_gap', 'mean'),
            loss_total_mean=('loss_total', 'mean'),
        ).reset_index()

        agg_path = out_dir / "gradient_cosine.csv"
        agg.to_csv(agg_path, index=False)
        print(f"  Saved: {agg_path} ({len(agg)} rows)")

        # Summary per feature
        summary_rows = []
        for feat in df['feature'].unique():
            sub = df[df['feature'] == feat]
            gc = sub['grad_cosine_aff_spec']

            # Overall stats
            overall_mean = gc.mean()
            overall_std = gc.std()
            pct_positive = (gc > 0).mean() * 100
            pct_negative = (gc < 0).mean() * 100

            # Early epochs (1-5) vs late epochs (21-25)
            early = sub[sub['epoch'] <= 5]['grad_cosine_aff_spec']
            late = sub[sub['epoch'] >= 21]['grad_cosine_aff_spec']

            summary_rows.append({
                'feature': feat,
                'overall_mean': overall_mean,
                'overall_std': overall_std,
                'pct_positive': pct_positive,
                'pct_negative': pct_negative,
                'early_mean_ep1_5': early.mean() if len(early) > 0 else None,
                'late_mean_ep21_25': late.mean() if len(late) > 0 else None,
                'early_pct_positive': (early > 0).mean() * 100 if len(early) > 0 else None,
                'late_pct_positive': (late > 0).mean() * 100 if len(late) > 0 else None,
                'n_epochs': sub['epoch'].nunique(),
                'n_folds': sub['fold'].nunique(),
                'n_seeds': sub['seed'].nunique(),
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "gradient_cosine_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Saved: {summary_path}")

        # Print report
        print(f"\n{'='*85}")
        print(f"  GRADIENT COSINE (Aff vs Spec) — Summary across {len(seeds)} seeds")
        print(f"{'='*85}")

        for _, r in summary_df.iterrows():
            print(f"\n  {r['feature']}  ({int(r['n_seeds'])} seeds × {int(r['n_folds'])} folds × {int(r['n_epochs'])} epochs)")
            print(f"    Overall mean cosine:   {r['overall_mean']:+.4f} ± {r['overall_std']:.4f}")
            print(f"    %% positive (aligned):  {r['pct_positive']:.1f}%")
            print(f"    %% negative (conflict): {r['pct_negative']:.1f}%")
            if r['early_mean_ep1_5'] is not None:
                print(f"    Early (ep 1-5):        {r['early_mean_ep1_5']:+.4f}  ({r['early_pct_positive']:.1f}% positive)")
            if r['late_mean_ep21_25'] is not None:
                print(f"    Late  (ep 21-25):      {r['late_mean_ep21_25']:+.4f}  ({r['late_pct_positive']:.1f}% positive)")

            # Interpretation
            if r['overall_mean'] > 0.05:
                print(f"    → Tasks are ALIGNED: gradients mostly cooperate")
            elif r['overall_mean'] < -0.05:
                print(f"    → Tasks are in CONFLICT: gradients mostly oppose")
            else:
                print(f"    → Tasks are WEAKLY ALIGNED: near-orthogonal gradients")

        # Epoch-by-epoch trajectory (averaged across folds/seeds)
        print(f"\n{'='*85}")
        print(f"  GRADIENT COSINE TRAJECTORY — Per epoch (mean across all folds/seeds)")
        print(f"{'='*85}")

        for feat in sorted(df['feature'].unique()):
            sub_agg = agg[agg['feature'] == feat].sort_values('epoch')
            print(f"\n  {feat}:")
            print(f"  {'Epoch':>6} {'Cosine':>10} {'±Std':>8} {'Aff Acc':>10} {'Spec Acc':>10} {'Aff Gap':>10} {'Spec Gap':>10}")
            print(f"  {'-'*68}")
            for _, r in sub_agg.iterrows():
                print(f"  {int(r['epoch']):>6} {r['grad_cos_mean']:>+10.4f} {r['grad_cos_std']:>8.4f}"
                      f" {r['acc_aff_mean']:>10.3f} {r['acc_spec_mean']:>10.3f}"
                      f" {r['aff_gap_mean']:>+10.2f} {r['spec_gap_mean']:>+10.2f}")

    else:
        print("  WARNING: 'grad_cosine_aff_spec' column not found in CSVs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
