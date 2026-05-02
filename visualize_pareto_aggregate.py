"""
visualize_pareto_aggregate.py — Paper-ready Pareto figures across seeds
========================================================================
Creates aggregated bar charts and scatter plots for Pareto front recovery
across all 5 seeds.

USAGE:
    python visualize_pareto_aggregate.py

OUTPUT:
    outputs/multiseed/figures/
        pareto_recall_barplot.png       — Recall bar chart, all models
        pareto_precision_recall.png     — Precision vs recall scatter
        pareto_per_seed_dotplot.png     — Per-seed recall dots (consistency)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> int:
    pareto_path = Path("outputs/multiseed/phase5_pareto_raw.csv")
    if not pareto_path.exists():
        print(f"Not found: {pareto_path}")
        return 1

    df = pd.read_csv(pareto_path)
    df = df[df['model'] != '_TRUE_PARETO_']
    print(f"Loaded {len(df)} Pareto entries")

    fig_dir = Path("outputs/multiseed/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = df['seed'].nunique()
    seeds = sorted(df['seed'].unique())

    # ====================================================================
    # FIGURE 1: ISO Pareto Recall Bar Chart (paper main figure)
    # ====================================================================
    iso = df[df['dataset'] == 'ISO']

    # Aggregate
    iso_agg = iso.groupby(['model', 'score_type']).agg(
        recall_mean=('recall', 'mean'),
        recall_std=('recall', lambda x: x.std(ddof=1) if len(x) > 1 else 0),
        precision_mean=('precision', 'mean'),
        precision_std=('precision', lambda x: x.std(ddof=1) if len(x) > 1 else 0),
    ).reset_index()
    iso_agg = iso_agg.sort_values('recall_mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'MOLM_latent_pca': '#e74c3c',
        'MOLM-ST_latent_pca': '#c0392b',
        'MOLM': '#3498db',
        'MOLM_probs': '#2980b9',
        'MOLM-ST': '#1abc9c',
        'MOLM-ST_probs': '#16a085',
        'NN (OneHot)': '#f39c12',
        'NN (Fusion-ESM2)': '#e67e22',
        'LDA': '#95a5a6',
    }

    labels = []
    for idx, (_, r) in enumerate(iso_agg.iterrows()):
        label = r['model']
        if r['score_type'] not in ['logits', 'ground_truth']:
            label = f"{r['model']}"
        labels.append(label)

        color = colors.get(r['model'], '#7f8c8d')
        is_molm = 'MOLM' in r['model'] and 'ST' not in r['model']
        edge = 'black' if is_molm else 'none'
        lw = 2 if is_molm else 0

        ax.barh(idx, r['recall_mean'], xerr=r['recall_std'],
                color=color, edgecolor=edge, linewidth=lw,
                capsize=4, alpha=0.85, height=0.7)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Pareto Recall (overlap / true Pareto)', fontsize=12)
    ax.set_title(f'ISO Pareto Front Recovery\n(15 true Pareto-optimal, mean ± std, {n_seeds} seeds)',
                 fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlim(-0.02, 0.45)
    ax.grid(True, axis='x', alpha=0.3)

    # Add recall values on bars
    for idx, (_, r) in enumerate(iso_agg.iterrows()):
        ax.text(r['recall_mean'] + r['recall_std'] + 0.01, idx,
                f"R={r['recall_mean']:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    path1 = fig_dir / 'pareto_recall_barplot.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {path1}")

    # ====================================================================
    # FIGURE 2: Per-seed dot plot (shows consistency)
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ds_idx, (ds_name, ds_df) in enumerate([('ISO', iso), ('IgG', df[df['dataset'] == 'IgG'])]):
        ax = axes[ds_idx]

        # Get unique model-score combos, sorted by mean recall
        combos = ds_df.groupby(['model', 'score_type'])['recall'].mean().reset_index()
        combos = combos.sort_values('recall', ascending=True)
        combo_labels = [f"{r['model']}" for _, r in combos.iterrows()]

        for idx, (_, combo) in enumerate(combos.iterrows()):
            sub = ds_df[(ds_df['model'] == combo['model']) &
                       (ds_df['score_type'] == combo['score_type'])]

            is_molm = 'MOLM' in combo['model'] and 'ST' not in combo['model']
            color = '#e74c3c' if 'latent_pca' in combo['model'] else (
                    '#3498db' if is_molm else '#7f8c8d')
            marker = 'D' if 'latent_pca' in combo['model'] else 'o'

            # Individual seeds
            for _, sr in sub.iterrows():
                ax.scatter(sr['recall'], idx, c=color, marker=marker,
                          s=60, alpha=0.5, zorder=3)

            # Mean
            mean_r = sub['recall'].mean()
            ax.scatter(mean_r, idx, c=color, marker=marker,
                      s=150, edgecolors='black', linewidth=1.5, zorder=4)

        true_count = 15 if ds_name == 'ISO' else 5
        ax.set_yticks(range(len(combo_labels)))
        ax.set_yticklabels(combo_labels, fontsize=9)
        ax.set_xlabel('Pareto Recall', fontsize=11)
        ax.set_title(f'{ds_name} Pareto Recovery ({true_count} true Pareto)\n'
                     f'Small dots = individual seeds, Large = mean',
                     fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(-0.05, 0.55)

    plt.tight_layout()
    path2 = fig_dir / 'pareto_per_seed_dotplot.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {path2}")

    # ====================================================================
    # FIGURE 3: Precision vs Recall scatter (trade-off)
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ds_idx, (ds_name, ds_df) in enumerate([('ISO', iso), ('IgG', df[df['dataset'] == 'IgG'])]):
        ax = axes[ds_idx]

        for model in ds_df['model'].unique():
            sub = ds_df[ds_df['model'] == model]
            is_molm_lat = 'latent_pca' in model
            is_molm = 'MOLM' in model and 'ST' not in model
            color = '#e74c3c' if is_molm_lat else ('#3498db' if is_molm else '#7f8c8d')
            marker = 'D' if is_molm_lat else ('s' if 'ST' in model else 'o')
            size = 120 if is_molm_lat else 80

            # Plot each seed
            ax.scatter(sub['recall'], sub['precision'], c=color, marker=marker,
                      s=size, alpha=0.4, zorder=3)

            # Plot mean with label
            mean_r = sub['recall'].mean()
            mean_p = sub['precision'].mean()
            ax.scatter(mean_r, mean_p, c=color, marker=marker,
                      s=200, edgecolors='black', linewidth=1.5, zorder=4)
            ax.annotate(model.replace('MOLM_', 'MOLM\n').replace('MOLM-ST_', 'ST\n'),
                       (mean_r, mean_p), fontsize=7, ha='left', va='bottom',
                       xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        true_count = 15 if ds_name == 'ISO' else 5
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{ds_name} Pareto — Precision vs Recall\n({true_count} true Pareto, {n_seeds} seeds)',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 0.55)
        ax.set_ylim(-0.05, 0.75)
        ax.grid(True, alpha=0.3)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.3, label='P=R line')

    plt.tight_layout()
    path3 = fig_dir / 'pareto_precision_recall.png'
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {path3}")

    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION FOR PAPER")
    print(f"{'='*60}")
    print(f"  Main text figure: pareto_recall_barplot.png")
    print(f"  Supplementary:    pareto_per_seed_dotplot.png")
    print(f"  Supplementary:    pareto_precision_recall.png")
    print(f"  Plus: one representative seed's diagnostic scatter")

    return 0


if __name__ == "__main__":
    sys.exit(main())
