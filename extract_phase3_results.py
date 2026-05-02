"""
extract_phase3_results.py - Phase 3 MOLM + MOLM-ST CV results across all seeds
================================================================================
Extracts MOLM and MOLM-ST cross-validation results from all seed pickles.

USAGE:
    python extract_phase3_results.py

OUTPUT:
    outputs/multiseed/phase3_molm_cv.csv       (mean +/- std across seeds)
    outputs/multiseed/phase3_molm_cv_raw.csv    (per-seed raw values)
"""

from __future__ import annotations
import argparse, pickle, io, sys
from pathlib import Path
import numpy as np
import pandas as pd


FEAT_LABELS = {'onehot': 'OneHot', 'esm2': 'ESM2', 'fusion_esm2': 'Fusion-ESM2'}

KEY_METRICS = [
    'affinity_mean', 'affinity_std',
    'specificity_mean', 'specificity_std',
    'aff_mccs_mean', 'aff_mccs_std',
    'spec_mccs_mean', 'spec_mccs_std',
    'aff_aucs_mean', 'aff_aucs_std',
    'spec_aucs_mean', 'spec_aucs_std',
]


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


def extract_cv(data: dict) -> list[dict]:
    rows = []

    for prefix, cv_key in [('MOLM', 'molm_cv'), ('MOLM-ST', 'molm_st_cv')]:
        cv_data = data.get(cv_key, {})
        if not isinstance(cv_data, dict):
            continue

        for feat_type, res in cv_data.items():
            if not isinstance(res, dict):
                continue
            fl = FEAT_LABELS.get(feat_type, feat_type)
            model_name = f"{prefix} ({fl})"

            for key, val in res.items():
                # Skip list/array/model objects, keep only scalar metrics
                if isinstance(val, (int, float, np.floating, np.integer)):
                    rows.append({
                        'model': model_name,
                        'metric': key,
                        'value': float(val),
                    })

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

    print(f"Extracting Phase 3 MOLM CV from {len(seeds)} seeds: {seeds}")

    all_rows = []
    for seed in seeds:
        pkl_path = seeds_root / f"seed_{seed}" / "phase3" / "molm_cv.pkl"
        if not pkl_path.exists():
            print(f"  [seed {seed}] molm_cv.pkl not found, skipping")
            continue

        data = load_pickle(pkl_path)
        seed_rows = extract_cv(data)
        for r in seed_rows:
            r['seed'] = seed
        all_rows.extend(seed_rows)
        print(f"  [seed {seed}] extracted {len(seed_rows)} metrics")

    if not all_rows:
        print("No data extracted!")
        return 1

    df = pd.DataFrame(all_rows)

    out_dir = output_dir / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw
    raw_path = out_dir / "phase3_molm_cv_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path} ({len(df)} rows)")

    # Aggregate across seeds
    agg = df.groupby(['model', 'metric']).agg(
        n_seeds=('value', 'count'),
        mean=('value', 'mean'),
        std=('value', lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
    ).reset_index()

    agg_path = out_dir / "phase3_molm_cv.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  Saved: {agg_path} ({len(agg)} rows)")

    # Print summary
    print(f"\n{'='*100}")
    print(f"  PHASE 3 MOLM CV — Results (mean +/- std across {len(seeds)} seeds)")
    print(f"{'='*100}")

    model_order = [
        'MOLM (OneHot)', 'MOLM (ESM2)', 'MOLM (Fusion-ESM2)',
        'MOLM-ST (OneHot)', 'MOLM-ST (ESM2)', 'MOLM-ST (Fusion-ESM2)',
    ]
    models = agg['model'].unique()
    models_sorted = [m for m in model_order if m in models]
    models_sorted += [m for m in models if m not in models_sorted]

    print(f"\n  {'Model':<25} {'Aff Acc':>14} {'Aff MCC':>14} {'Aff AUC':>14} {'Spec Acc':>14} {'Spec MCC':>14} {'Spec AUC':>14}")
    print(f"  {'-'*100}")

    for model in models_sorted:
        sub = agg[agg['model'] == model]

        def get_val(metric_name):
            row = sub[sub['metric'] == metric_name]
            if row.empty:
                return "    —     "
            r = row.iloc[0]
            if r['std'] > 0:
                return f"{r['mean']:.4f}+/-{r['std']:.4f}"
            return f"{r['mean']:.4f}"

        aff_acc = get_val('affinity_mean')
        aff_mcc = get_val('aff_mccs_mean')
        aff_auc = get_val('aff_aucs_mean')
        spec_acc = get_val('specificity_mean')
        spec_mcc = get_val('spec_mccs_mean')
        spec_auc = get_val('spec_aucs_mean')

        print(f"  {model:<25} {aff_acc:>14} {aff_mcc:>14} {aff_auc:>14} {spec_acc:>14} {spec_mcc:>14} {spec_auc:>14}")

    # Compare MOLM vs MOLM-ST directly
    print(f"\n{'='*100}")
    print(f"  MOLM vs MOLM-ST (joint vs separate training)")
    print(f"{'='*100}")

    for feat in ['OneHot', 'ESM2', 'Fusion-ESM2']:
        molm_sub = agg[agg['model'] == f'MOLM ({feat})']
        st_sub = agg[agg['model'] == f'MOLM-ST ({feat})']

        if molm_sub.empty or st_sub.empty:
            continue

        print(f"\n  {feat}:")
        for metric in ['affinity_mean', 'specificity_mean', 'aff_mccs_mean', 'spec_mccs_mean']:
            m_row = molm_sub[molm_sub['metric'] == metric]
            s_row = st_sub[st_sub['metric'] == metric]
            if m_row.empty or s_row.empty:
                continue
            m_val = m_row.iloc[0]['mean']
            s_val = s_row.iloc[0]['mean']
            gap = m_val - s_val
            winner = "MOLM" if gap > 0 else "MOLM-ST"
            print(f"    {metric:<22} MOLM={m_val:.4f}  MOLM-ST={s_val:.4f}  gap={gap:+.4f}  {winner}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
