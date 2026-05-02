"""
extract_persite_holdout.py — Per-site holdout results across seeds
====================================================================
Reads holdout.pkl from each seed, extracts per-site accuracy and MCC
for every model, and produces a paper-ready CSV.

USAGE:
    python extract_persite_holdout.py
    python extract_persite_holdout.py --output-dir outputs

OUTPUT:
    outputs/multiseed/persite_holdout.csv
"""

from __future__ import annotations
import argparse, pickle, io, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


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


def extract_persite(holdout: dict, mode_key: str) -> list[dict]:
    """Extract per-site results from a single seed's holdout dict."""
    res = holdout.get(mode_key, {})
    site_names = res.get('site_names', [])
    train_sizes = res.get('train_sizes', [])
    test_sizes = res.get('test_sizes', [])
    n_sites = len(site_names)
    
    if n_sites == 0:
        return []
    
    rows = []
    # Find all per-site metric lists (they have exactly n_sites elements)
    for key, vals in res.items():
        if not isinstance(vals, list) or len(vals) != n_sites:
            continue
        if key in ['sites', 'site_names', 'train_sizes', 'test_sizes']:
            continue
        # Skip non-numeric lists (e.g. McNemar p-value lists in nested dicts)
        if not all(isinstance(v, (int, float, np.floating, np.integer)) for v in vals):
            continue
        
        for si in range(n_sites):
            rows.append({
                'site': site_names[si],
                'train_n': train_sizes[si],
                'test_n': test_sizes[si],
                'metric_key': key,
                'value': float(vals[si]),
            })
    
    return rows


def parse_metric_key(key: str) -> tuple[str, str, str]:
    """Parse e.g. 'molm_fusion_esm2_affinity_accs' -> ('MOLM', 'Fusion-ESM2', 'affinity_acc')"""
    # Model prefixes
    if key.startswith('molm_st_'):
        model = 'MOLM-ST'
        rest = key[len('molm_st_'):]
    elif key.startswith('molm_'):
        model = 'MOLM'
        rest = key[len('molm_'):]
    elif key.startswith('nn_'):
        model = 'NN'
        rest = key[len('nn_'):]
    elif key.startswith('lda_'):
        model = 'LDA'
        rest = key[len('lda_'):]
    else:
        return key, '', key
    
    # Feature type
    feat_map = {
        'fusion_esm2_': 'Fusion-ESM2',
        'onehot_': 'OneHot',
        'esm2_': 'ESM2',
    }
    feature = ''
    for prefix, label in feat_map.items():
        if rest.startswith(prefix):
            feature = label
            rest = rest[len(prefix):]
            break
    
    # Metric name: strip trailing 's' from 'accs' -> 'acc', 'mccs' -> 'mcc'
    metric = rest
    if metric.endswith('_accs'):
        metric = metric[:-1]  # accs -> acc  (but keep as 'affinity_accs' -> 'affinity_acc')
    elif metric.endswith('_mccs'):
        metric = metric[:-1]
    
    return model, feature, metric


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
    
    print(f"Extracting per-site holdout from {len(seeds)} seeds: {seeds}")
    
    all_rows = []
    
    for seed in seeds:
        pkl_path = seeds_root / f"seed_{seed}" / "phase4" / "holdout.pkl"
        if not pkl_path.exists():
            print(f"  [seed {seed}] holdout.pkl not found, skipping")
            continue
        
        holdout = load_pickle(pkl_path)
        
        for mode_key, mode_label in [('top', 'TOP'), ('wt', 'WILDTYPE')]:
            site_rows = extract_persite(holdout, mode_key)
            for r in site_rows:
                model, feature, metric = parse_metric_key(r['metric_key'])
                r['seed'] = seed
                r['mode'] = mode_label
                r['model'] = model
                r['feature'] = feature
                r['metric'] = metric
                all_rows.append(r)
        
        print(f"  [seed {seed}] extracted {len([r for r in all_rows if r['seed'] == seed])} site-metric entries")
    
    if not all_rows:
        print("No data extracted!")
        return 1
    
    df = pd.DataFrame(all_rows)
    
    # Aggregate across seeds: mean ± std per (mode, site, model, feature, metric)
    agg = df.groupby(['mode', 'site', 'model', 'feature', 'metric']).agg(
        n_seeds=('value', 'count'),
        mean=('value', 'mean'),
        std=('value', lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
        min=('value', 'min'),
        max=('value', 'max'),
        values=('value', lambda x: ';'.join(f'{v:.4f}' for v in x)),
        train_n=('train_n', 'first'),
        test_n=('test_n', 'first'),
    ).reset_index()
    
    # Sort nicely
    site_order = ['Kabat 33', 'Kabat 50', 'Kabat 54', 'Kabat 55', 'Kabat 56', 'Kabat 95', 'Kabat 97', 'Kabat 102']
    model_order = ['MOLM', 'MOLM-ST', 'NN', 'LDA']
    agg['site_rank'] = agg['site'].map({s: i for i, s in enumerate(site_order)})
    agg['model_rank'] = agg['model'].map({m: i for i, m in enumerate(model_order)})
    agg = agg.sort_values(['mode', 'site_rank', 'model_rank', 'feature', 'metric'])
    agg = agg.drop(columns=['site_rank', 'model_rank'])
    
    # Save full CSV
    out_dir = output_dir / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    full_path = out_dir / "persite_holdout.csv"
    agg.to_csv(full_path, index=False)
    print(f"\n  ✓ Saved: {full_path} ({len(agg)} rows)")
    
    # Print a nice summary — focus on accuracy and MCC
    for mode in ['TOP', 'WILDTYPE']:
        print(f"\n{'='*90}")
        print(f"  {mode} HOLDOUT — Per-Site Accuracy (mean ± std, {len(seeds)} seeds)")
        print(f"{'='*90}")
        
        for site in site_order:
            sub = agg[(agg['mode'] == mode) & (agg['site'] == site)]
            if sub.empty:
                continue
            
            test_n = sub['test_n'].iloc[0]
            train_n = sub['train_n'].iloc[0]
            print(f"\n  {site} (train={train_n}, test={test_n})")
            print(f"  {'Model':<30} {'Aff Acc':>14} {'Aff MCC':>14} {'Spec Acc':>14} {'Spec MCC':>14}")
            print(f"  {'-'*88}")
            
            # Get unique model+feature combos
            combos = sub[['model', 'feature']].drop_duplicates()
            
            for _, combo in combos.iterrows():
                m, f = combo['model'], combo['feature']
                label = f"{m} ({f})" if f else m
                
                mask = (sub['model'] == m) & (sub['feature'] == f)
                
                def get_val(metric_contains):
                    row = sub[mask & sub['metric'].str.contains(metric_contains, na=False)]
                    if row.empty:
                        return "    —     "
                    r = row.iloc[0]
                    if r['std'] > 0:
                        return f"{r['mean']:.3f}±{r['std']:.3f}"
                    else:
                        return f"{r['mean']:.3f}      "
                
                aff_acc = get_val('affinity_acc')
                aff_mcc = get_val('aff_mcc')
                spec_acc = get_val('specificity_acc')
                spec_mcc = get_val('spec_mcc')
                
                if aff_acc.strip() == '—' and spec_acc.strip() == '—':
                    continue
                
                print(f"  {label:<30} {aff_acc:>14} {aff_mcc:>14} {spec_acc:>14} {spec_mcc:>14}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
