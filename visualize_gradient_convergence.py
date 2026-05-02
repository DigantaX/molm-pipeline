"""
visualize_gradient_convergence.py — MOLM dual-objective convergence analysis
==============================================================================
Generates publication-quality figures showing whether MOLM reaches a shared
minimum for both affinity and specificity objectives.

USAGE:
    python visualize_gradient_convergence.py

OUTPUT:
    outputs/multiseed/figures/
        convergence_dual_loss.png         — Both losses decreasing simultaneously
        convergence_loss_landscape.png    — 2D loss-loss trajectory (path to minimum)
        convergence_gradient_alignment.png — Gradient cosine + loss overlay
        convergence_score_gaps.png        — Score margins growing for both tasks
        convergence_generalization.png    — Transfer rho during training
        convergence_combined.png          — All panels in one figure
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection


def make_gradient_colored_line(ax, x, y, colors, cmap='coolwarm', lw=2.5):
    """Draw a line colored by a third variable."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin=min(colors), vmax=max(colors)))
    lc.set_array(np.array(colors[:-1]))
    lc.set_linewidth(lw)
    return ax.add_collection(lc)


def main() -> int:
    raw_path = Path("outputs/multiseed/gradient_cosine_raw.csv")
    if not raw_path.exists():
        print(f"Not found: {raw_path}")
        print("Run extract_gradient_diagnostics.py first")
        return 1

    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows from {raw_path}")

    fig_dir = Path("outputs/multiseed/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    features = sorted(df['feature'].unique())
    feat_colors = {'ESM2': '#e74c3c', 'Fusion-ESM2': '#3498db', 'OneHot': '#2ecc71'}
    n_seeds = df['seed'].nunique()

    # Aggregate: mean per epoch per feature (across folds and seeds)
    agg = df.groupby(['feature', 'epoch']).agg(
        loss_aff=('loss_cls_aff', 'mean'), loss_aff_std=('loss_cls_aff', 'std'),
        loss_spec=('loss_cls_spec', 'mean'), loss_spec_std=('loss_cls_spec', 'std'),
        loss_total=('loss_total', 'mean'), loss_total_std=('loss_total', 'std'),
        acc_aff=('acc_aff', 'mean'), acc_aff_std=('acc_aff', 'std'),
        acc_spec=('acc_spec', 'mean'), acc_spec_std=('acc_spec', 'std'),
        aff_gap=('aff_gap', 'mean'), aff_gap_std=('aff_gap', 'std'),
        spec_gap=('spec_gap', 'mean'), spec_gap_std=('spec_gap', 'std'),
        grad_cos=('grad_cosine_aff_spec', 'mean'), grad_cos_std=('grad_cosine_aff_spec', 'std'),
    ).reset_index()

    # Check for generalization columns
    has_gen = 'iso_aff_rho' in df.columns and df['iso_aff_rho'].notna().any()
    if has_gen:
        gen_agg = df.dropna(subset=['iso_aff_rho']).groupby(['feature', 'epoch']).agg(
            iso_aff=('iso_aff_rho', 'mean'), iso_spec=('iso_spec_rho', 'mean'),
            igg_aff=('igg_aff_rho', 'mean'), igg_spec=('igg_spec_rho', 'mean'),
        ).reset_index()

    # ====================================================================
    # FIGURE 1: Dual Loss Convergence
    # ====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Dual-Objective Loss Convergence (mean across {n_seeds} seeds x 5 folds)',
                 fontsize=13, fontweight='bold', y=1.02)

    for i, feat in enumerate(features):
        ax = axes[i]
        sub = agg[agg['feature'] == feat]
        ep = sub['epoch'].values

        ax.plot(ep, sub['loss_aff'], 'o-', color='#3498db', lw=2, ms=3, label='Affinity loss')
        ax.fill_between(ep, sub['loss_aff'] - sub['loss_aff_std'],
                        sub['loss_aff'] + sub['loss_aff_std'], alpha=0.15, color='#3498db')

        ax.plot(ep, sub['loss_spec'], 's-', color='#e74c3c', lw=2, ms=3, label='Specificity loss')
        ax.fill_between(ep, sub['loss_spec'] - sub['loss_spec_std'],
                        sub['loss_spec'] + sub['loss_spec_std'], alpha=0.15, color='#e74c3c')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss' if i == 0 else '', fontsize=11)
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 25)

    plt.tight_layout()
    path1 = fig_dir / 'convergence_dual_loss.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {path1}")

    # ====================================================================
    # FIGURE 2: 2D Loss-Loss Trajectory (path to shared minimum)
    # ====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Training Path in Loss Space (Aff vs Spec)',
                 fontsize=13, fontweight='bold', y=1.02)

    for i, feat in enumerate(features):
        ax = axes[i]
        sub = agg[agg['feature'] == feat]
        x = sub['loss_aff'].values
        y = sub['loss_spec'].values
        epochs = sub['epoch'].values
        cos = sub['grad_cos'].values

        # Color line by gradient cosine
        lc = make_gradient_colored_line(ax, x, y, cos, cmap='RdYlGn', lw=3)
        cb = plt.colorbar(lc, ax=ax, label='Grad cosine', shrink=0.8)

        # Mark start and end
        ax.scatter(x[0], y[0], s=150, c='red', marker='*', zorder=5, label=f'Epoch 1')
        ax.scatter(x[-1], y[-1], s=150, c='green', marker='*', zorder=5, label=f'Epoch 25')

        # Add epoch labels
        for ep_idx in [0, 4, 9, 14, 24]:
            if ep_idx < len(x):
                ax.annotate(f'ep{int(epochs[ep_idx])}', (x[ep_idx], y[ep_idx]),
                           fontsize=7, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Draw arrow showing direction
        if len(x) > 1:
            ax.annotate('', xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))

        ax.set_xlabel('Affinity Loss', fontsize=11)
        ax.set_ylabel('Specificity Loss' if i == 0 else '', fontsize=11)
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = fig_dir / 'convergence_loss_landscape.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {path2}")

    # ====================================================================
    # FIGURE 3: Gradient Alignment + Loss Overlay
    # ====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Gradient Cosine (Aff vs Spec) + Total Loss',
                 fontsize=13, fontweight='bold', y=1.02)

    for i, feat in enumerate(features):
        ax = axes[i]
        sub = agg[agg['feature'] == feat]
        ep = sub['epoch'].values

        # Gradient cosine
        ax.plot(ep, sub['grad_cos'], 'o-', color='#8e44ad', lw=2, ms=4, label='Grad cosine')
        ax.fill_between(ep, sub['grad_cos'] - sub['grad_cos_std'],
                        sub['grad_cos'] + sub['grad_cos_std'], alpha=0.15, color='#8e44ad')
        ax.axhline(0, ls='--', color='red', alpha=0.5, lw=1)
        ax.fill_between(ep, sub['grad_cos'], 0,
                        where=sub['grad_cos'] < 0, alpha=0.3, color='red', label='Conflict zone')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Gradient Cosine' if i == 0 else '', fontsize=11)
        ax.set_title(feat, fontsize=12, fontweight='bold')

        # Total loss on secondary axis
        ax2 = ax.twinx()
        ax2.plot(ep, sub['loss_total'], '--', color='#7f8c8d', lw=1.5, alpha=0.7, label='Total loss')
        if i == 2:
            ax2.set_ylabel('Total Loss', fontsize=11, color='#7f8c8d')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 25)

    plt.tight_layout()
    path3 = fig_dir / 'convergence_gradient_alignment.png'
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {path3}")

    # ====================================================================
    # FIGURE 4: Score Gap Growth (both tasks simultaneously)
    # ====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Score Margin Growth — Both Tasks Simultaneously',
                 fontsize=13, fontweight='bold', y=1.02)

    for i, feat in enumerate(features):
        ax = axes[i]
        sub = agg[agg['feature'] == feat]
        ep = sub['epoch'].values

        ax.plot(ep, sub['aff_gap'], 'o-', color='#3498db', lw=2, ms=3, label='Affinity gap')
        ax.fill_between(ep, sub['aff_gap'] - sub['aff_gap_std'],
                        sub['aff_gap'] + sub['aff_gap_std'], alpha=0.15, color='#3498db')

        ax.plot(ep, sub['spec_gap'], 's-', color='#e74c3c', lw=2, ms=3, label='Specificity gap')
        ax.fill_between(ep, sub['spec_gap'] - sub['spec_gap_std'],
                        sub['spec_gap'] + sub['spec_gap_std'], alpha=0.15, color='#e74c3c')

        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Score Gap (pos - neg)' if i == 0 else '', fontsize=11)
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 25)

    plt.tight_layout()
    path4 = fig_dir / 'convergence_score_gaps.png'
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {path4}")

    # ====================================================================
    # FIGURE 5: Dual Accuracy Convergence
    # ====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Dual-Objective Accuracy Convergence',
                 fontsize=13, fontweight='bold', y=1.02)

    for i, feat in enumerate(features):
        ax = axes[i]
        sub = agg[agg['feature'] == feat]
        ep = sub['epoch'].values

        ax.plot(ep, sub['acc_aff'], 'o-', color='#3498db', lw=2, ms=3, label='Affinity acc')
        ax.fill_between(ep, sub['acc_aff'] - sub['acc_aff_std'],
                        sub['acc_aff'] + sub['acc_aff_std'], alpha=0.15, color='#3498db')

        ax.plot(ep, sub['acc_spec'], 's-', color='#e74c3c', lw=2, ms=3, label='Specificity acc')
        ax.fill_between(ep, sub['acc_spec'] - sub['acc_spec_std'],
                        sub['acc_spec'] + sub['acc_spec_std'], alpha=0.15, color='#e74c3c')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy' if i == 0 else '', fontsize=11)
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 25)
        ax.set_ylim(0.45, 0.95)

    plt.tight_layout()
    path5 = fig_dir / 'convergence_dual_accuracy.png'
    plt.savefig(path5, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {path5}")

    # ====================================================================
    # FIGURE 6: COMBINED — Paper-ready 4-panel figure
    # ====================================================================
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(f'MOLM Dual-Objective Convergence Analysis\n(mean ± std, {n_seeds} seeds x 5 folds)',
                 fontsize=14, fontweight='bold', y=0.98)

    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    panel_idx = 0

    # Row 1: Dual loss convergence
    for i, feat in enumerate(features):
        ax = fig.add_subplot(gs[0, i])
        sub = agg[agg['feature'] == feat]
        ep = sub['epoch'].values
        ax.plot(ep, sub['loss_aff'], 'o-', color='#3498db', lw=2, ms=2, label='Aff loss')
        ax.fill_between(ep, sub['loss_aff']-sub['loss_aff_std'], sub['loss_aff']+sub['loss_aff_std'], alpha=0.15, color='#3498db')
        ax.plot(ep, sub['loss_spec'], 's-', color='#e74c3c', lw=2, ms=2, label='Spec loss')
        ax.fill_between(ep, sub['loss_spec']-sub['loss_spec_std'], sub['loss_spec']+sub['loss_spec_std'], alpha=0.15, color='#e74c3c')
        ax.set_title(f'({panel_labels[panel_idx]}) {feat} — Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss' if i==0 else '')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(1,25)
        panel_idx += 1

    # Row 2: Gradient cosine
    for i, feat in enumerate(features):
        ax = fig.add_subplot(gs[1, i])
        sub = agg[agg['feature'] == feat]
        ep = sub['epoch'].values
        ax.plot(ep, sub['grad_cos'], 'o-', color='#8e44ad', lw=2, ms=3)
        ax.fill_between(ep, sub['grad_cos']-sub['grad_cos_std'], sub['grad_cos']+sub['grad_cos_std'], alpha=0.15, color='#8e44ad')
        ax.axhline(0, ls='--', color='red', alpha=0.5, lw=1)
        ax.fill_between(ep, sub['grad_cos'], 0, where=sub['grad_cos']<0, alpha=0.3, color='red')
        seed_mean = df[df['feature']==feat].groupby('seed')['grad_cosine_aff_spec'].mean()
        ax.set_title(f'({panel_labels[panel_idx]}) {feat} — Grad Cosine\n'
                     f'(per-seed: {seed_mean.mean():+.3f} +/- {seed_mean.std():.3f})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Cosine(grad_aff, grad_spec)' if i==0 else '')
        ax.grid(True, alpha=0.3); ax.set_xlim(1,25)
        panel_idx += 1

    # Row 3: Score gaps
    for i, feat in enumerate(features):
        ax = fig.add_subplot(gs[2, i])
        sub = agg[agg['feature'] == feat]
        ep = sub['epoch'].values
        ax.plot(ep, sub['aff_gap'], 'o-', color='#3498db', lw=2, ms=2, label='Aff gap')
        ax.fill_between(ep, sub['aff_gap']-sub['aff_gap_std'], sub['aff_gap']+sub['aff_gap_std'], alpha=0.15, color='#3498db')
        ax.plot(ep, sub['spec_gap'], 's-', color='#e74c3c', lw=2, ms=2, label='Spec gap')
        ax.fill_between(ep, sub['spec_gap']-sub['spec_gap_std'], sub['spec_gap']+sub['spec_gap_std'], alpha=0.15, color='#e74c3c')
        ax.axhline(0, ls='--', color='black', alpha=0.3)
        ax.set_title(f'({panel_labels[panel_idx]}) {feat} — Score Gaps', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Gap (pos - neg)' if i==0 else '')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(1,25)
        panel_idx += 1

    path6 = fig_dir / 'convergence_combined.png'
    plt.savefig(path6, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved: {path6}")

    # ====================================================================
    # Print convergence verdict
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"  CONVERGENCE VERDICT")
    print(f"{'='*70}")

    for feat in features:
        sub = agg[agg['feature'] == feat]
        loss_aff_start = sub[sub['epoch']==1]['loss_aff'].values[0]
        loss_aff_end = sub[sub['epoch']==25]['loss_aff'].values[0]
        loss_spec_start = sub[sub['epoch']==1]['loss_spec'].values[0]
        loss_spec_end = sub[sub['epoch']==25]['loss_spec'].values[0]
        cos_start = sub[sub['epoch']==1]['grad_cos'].values[0]
        cos_end = sub[sub['epoch']==25]['grad_cos'].values[0]
        gap_aff_end = sub[sub['epoch']==25]['aff_gap'].values[0]
        gap_spec_end = sub[sub['epoch']==25]['spec_gap'].values[0]

        aff_reduction = (loss_aff_start - loss_aff_end) / loss_aff_start * 100
        spec_reduction = (loss_spec_start - loss_spec_end) / loss_spec_start * 100

        print(f"\n  {feat}:")
        print(f"    Aff loss:  {loss_aff_start:.4f} -> {loss_aff_end:.4f} ({aff_reduction:+.1f}% reduction)")
        print(f"    Spec loss: {loss_spec_start:.4f} -> {loss_spec_end:.4f} ({spec_reduction:+.1f}% reduction)")
        print(f"    Grad cos:  {cos_start:+.4f} -> {cos_end:+.4f}")
        print(f"    Final gaps: Aff={gap_aff_end:+.2f}, Spec={gap_spec_end:+.2f}")

        both_decrease = aff_reduction > 0 and spec_reduction > 0
        cos_positive = cos_end > 0
        gaps_positive = gap_aff_end > 0 and gap_spec_end > 0

        if both_decrease and cos_positive and gaps_positive:
            print(f"    -> SHARED MINIMUM REACHED: both losses decrease, gradients aligned, both gaps positive")
        elif both_decrease and gaps_positive:
            print(f"    -> CONVERGING: both losses decrease, both gaps positive")
        else:
            print(f"    -> PARTIAL: check individual metrics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
