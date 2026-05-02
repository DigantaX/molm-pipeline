"""
extract_phase2_results.py - Phase 2 baseline results across all seeds
======================================================================
Extracts LDA and NN CV results from all seed pickles and saves as CSV.

USAGE:
    python extract_phase2_results.py

OUTPUT:
    outputs/multiseed/phase2_baselines.csv       (mean +/- std across seeds)
    outputs/multiseed/phase2_baselines_raw.csv    (per-seed raw values)
"""

from __future__ import annotations
import argparse, pickle, io, sys
from pathlib import Path
import numpy as np
import pandas as pd


FEAT_LABELS = {'onehot': 'OneHot', 'esm2': 'ESM2', 'fusion_esm2': 'Fusion-ESM2'}


def load_pickle(path: Path):
    try:
        import torch
        original_loader = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = (
            lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        )
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        finally:
            torch.storage._load_from_bytes = original_loader
    except ImportError:
        with open(path, "rb") as f:
            return pickle.load(f)


def extract_baselines(data: dict) -> list[dict]:
    rows = []

    # LDA: data['lda'] = {'LDA (OneHot)': {'affinity_cv_mean': ..., 'affinity_mcc': ..., ...}, ...}
    for model_name, res in data.get('lda', {}).items():
        if not isinstance(res, dict):
            continue
        for key, val in res.items():
            if key in ('lda_aff', 'lda_spec'):  # skip model objects
                continue
            if isinstance(val, (int, float, np.floating, np.integer)):
                rows.append({'model': model_name, 'metric': key, 'value': float(val)})

    # NN: data['nn'] = {'onehot': {'affinity': {'cv_mean': ..., 'mcc_mean': ...}, 'specificity': {...}}, ...}
    for feat_type, res in data.get('nn', {}).items():
        if not isinstance(res, dict):
            continue
        fl = FEAT_LABELS.get(feat_type, feat_type)
        model_name = f"NN ({fl})"

        for task in ['affinity', 'specificity']:
            task_data = res.get(task, {})
            if not isinstance(task_data, dict):
                continue
            for key, val in task_data.items():
                if key in ('model', 'cv_scores', 'auc_scores'):  # skip objects/lists
                    continue
                if isinstance(val, (int, float, np.floating, np.integer)):
                    rows.append({
                        'model': model_name,
                        'metric': f"{task}_{key}",
                        'value': float(val)
                    })

        # Also capture top-level summary keys
        for key in ['affinity_cv_mean', 'affinity_cv_std', 'specificity_cv_mean', 'specificity_cv_std']:
            val = res.get(key)
            if val is not None and isinstance(val, (int, float, np.floating, np.integer)):
                rows.append({'model': model_name, 'metric': key, 'value': float(val)})

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    seeds_root = output_dir / "seeds"

    if args.seeds is None:
        seeds = sorted([int(p.name.split("_")[1]) for p in seeds_root.iterdir()
                        if p.is_dir() and p.name.startswith("seed_")])
    else:
        seeds = sorted(args.seeds)

    print(f"Extracting Phase 2 baselines from {len(seeds)} seeds: {seeds}")

    all_rows = []
    for seed in seeds:
        pkl_path = seeds_root / f"seed_{seed}" / "phase2" / "baselines.pkl"
        if not pkl_path.exists():
            print(f"  [seed {seed}] baselines.pkl not found, skipping")
            continue

        data = load_pickle(pkl_path)
        seed_rows = extract_baselines(data)
        for r in seed_rows:
            r['seed'] = seed
        all_rows.extend(seed_rows)
        print(f"  [seed {seed}] extracted {len(seed_rows)} metrics")

    if not all_rows:
        print("No data extracted!")
        return 1

    df = pd.DataFrame(all_rows)

    # Save raw
    out_dir = output_dir / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "phase2_baselines_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path} ({len(df)} rows)")

    # Aggregate across seeds
    agg = df.groupby(['model', 'metric']).agg(
        n_seeds=('value', 'count'),
        mean=('value', 'mean'),
        std=('value', lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
    ).reset_index()

    agg_path = out_dir / "phase2_baselines.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  Saved: {agg_path} ({len(agg)} rows)")

    # Print summary
    # Focus on key metrics: cv_mean, mcc, auc
    key_metrics = ['affinity_cv_mean', 'affinity_mcc', 'affinity_auc',
                   'specificity_cv_mean', 'specificity_mcc', 'specificity_auc']

    # Also check NN naming — might be affinity_cv_mean or affinity_mcc_mean
    nn_key_metrics = ['affinity_cv_mean', 'affinity_mcc_mean', 'affinity_auc_mean',
                      'specificity_cv_mean', 'specificity_mcc_mean', 'specificity_auc_mean']

    print(f"\n{'='*95}")
    print(f"  PHASE 2 BASELINES — CV Results (mean +/- std, {len(seeds)} seeds)")
    print(f"{'='*95}")

    models = agg['model'].unique()
    model_order = ['LDA (OneHot)', 'LDA (ESM2)', 'LDA (Fusion-ESM2)',
                   'NN (OneHot)', 'NN (ESM2)', 'NN (Fusion-ESM2)']
    models_sorted = [m for m in model_order if m in models]
    models_sorted += [m for m in models if m not in models_sorted]

    print(f"\n  {'Model':<25} {'Aff Acc':>14} {'Aff MCC':>14} {'Aff AUC':>14} {'Spec Acc':>14} {'Spec MCC':>14} {'Spec AUC':>14}")
    print(f"  {'-'*97}")

    for model in models_sorted:
        sub = agg[agg['model'] == model]

        def get_val(metric_names):
            """Try multiple metric name variants."""
            if isinstance(metric_names, str):
                metric_names = [metric_names]
            for mn in metric_names:
                row = sub[sub['metric'] == mn]
                if not row.empty:
                    r = row.iloc[0]
                    if r['std'] > 0:
                        return f"{r['mean']:.4f}+/-{r['std']:.4f}"
                    else:
                        return f"{r['mean']:.4f}"
            return "    —     "

        aff_acc = get_val(['affinity_cv_mean'])
        aff_mcc = get_val(['affinity_mcc', 'affinity_mcc_mean'])
        aff_auc = get_val(['affinity_auc', 'affinity_auc_mean'])
        spec_acc = get_val(['specificity_cv_mean'])
        spec_mcc = get_val(['specificity_mcc', 'specificity_mcc_mean'])
        spec_auc = get_val(['specificity_auc', 'specificity_auc_mean'])

        print(f"  {model:<25} {aff_acc:>14} {aff_mcc:>14} {aff_auc:>14} {spec_acc:>14} {spec_mcc:>14} {spec_auc:>14}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
