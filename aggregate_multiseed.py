"""
aggregate_multiseed.py — Aggregate per-seed MOLM results
===========================================================
Reads per-seed pickles from <output_dir>/seeds/seed_<N>/phase{2,3,4,5}/*.pkl
and produces paper-ready summaries with mean ± std and paired-bootstrap
p-values for MOLM vs each baseline.

USAGE
-----
    # Default: reads ./outputs/seeds/, auto-detects all seed_<N> dirs
    python aggregate_multiseed.py

    # Custom output root
    python aggregate_multiseed.py --output-dir D:/research/molm_outputs

    # Specific seeds only
    python aggregate_multiseed.py --seeds 42 123 456

OUTPUTS
-------
    <output_dir>/multiseed/
    ├── multiseed_summary.csv      # flat: model, metric, n_seeds, mean, std, ci_lo, ci_hi
    ├── multiseed_summary.json     # full nested + raw per-seed values
    ├── multiseed_bootstrap.csv    # MOLM vs baselines paired bootstrap
    └── multiseed_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# I/O HELPERS
# ============================================================================
def load_pickle(path: Path):
    """Load a pickle, tolerant to torch tensors saved on GPU."""
    try:
        import torch  # noqa: F401
        # Re-route torch storage loading to CPU in case some pickles contain
        # GPU tensors (mirrors phase0_config._safe_pickle_load behaviour).
        import io
        original_loader = torch.storage._load_from_bytes  # type: ignore[attr-defined]
        torch.storage._load_from_bytes = (  # type: ignore[attr-defined]
            lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        )
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        finally:
            torch.storage._load_from_bytes = original_loader  # type: ignore[attr-defined]
    except ImportError:
        with open(path, "rb") as f:
            return pickle.load(f)


def safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def is_numeric(v):
    return isinstance(v, (int, float, np.floating, np.integer)) and not (
        isinstance(v, float) and np.isnan(v)
    )


# ============================================================================
# PER-PHASE EXTRACTORS
# Each returns a list of {"model": str, "metric": str, "value": float}
# ============================================================================
def extract_phase2(baselines: dict) -> list[dict]:
    """LDA + NN baselines (Phase 2)."""
    rows = []

    # LDA — model_key already includes the feature label, e.g. "LDA (OneHot)"
    for model_key, res in baselines.get("lda", {}).items():
        for metric in [
            "affinity_cv_mean", "affinity_mcc", "affinity_auc",
            "specificity_cv_mean", "specificity_mcc", "specificity_auc",
        ]:
            v = res.get(metric)
            if is_numeric(v):
                rows.append({"model": model_key, "metric": metric, "value": float(v)})

    # NN — keyed by feature type
    for feat_type, res in baselines.get("nn", {}).items():
        model = f"NN ({feat_type})"
        for top_metric in ["affinity_cv_mean", "specificity_cv_mean"]:
            v = res.get(top_metric)
            if is_numeric(v):
                rows.append({"model": model, "metric": top_metric, "value": float(v)})
        # Nested per-task metrics
        for task in ["affinity", "specificity"]:
            for sub in ["mcc_mean", "auc_mean", "cv_mean"]:
                v = safe_get(res, task, sub)
                if is_numeric(v):
                    rows.append({"model": model,
                                 "metric": f"{task}_{sub}",
                                 "value": float(v)})
    return rows


def extract_phase3(molm_data: dict) -> list[dict]:
    """MOLM + MOLM-ST 5-fold CV (Phase 3)."""
    rows = []
    for kind_key, kind_label in [("molm_cv", "MOLM"), ("molm_st_cv", "MOLM-ST")]:
        for feat_type, res in molm_data.get(kind_key, {}).items():
            model = f"{kind_label} ({feat_type})"
            for metric in [
                "affinity_mean", "specificity_mean",
                "aff_aucs_mean", "spec_aucs_mean",
                "aff_aps_mean", "spec_aps_mean",
                "aff_mccs_mean", "spec_mccs_mean",
                "affinity_accs_mean", "specificity_accs_mean",
            ]:
                v = res.get(metric)
                if is_numeric(v):
                    rows.append({"model": model, "metric": metric, "value": float(v)})
    return rows


def extract_phase4(holdout: dict) -> list[dict]:
    """Held-out site evaluation (Phase 4)."""
    rows = []
    for mode_key, mode_label in [("top", "TopHoldout"), ("wt", "WTHoldout")]:
        res = holdout.get(mode_key, {})
        if not isinstance(res, dict):
            continue
        for k, v in res.items():
            # Aggregated keys end in _mean (acc, mcc, auc, etc.)
            if not isinstance(k, str) or not k.endswith("_mean"):
                continue
            if not is_numeric(v):
                continue
            # Encode mode into model name so each (mode, model_metric) is unique
            base = k[:-len("_mean")]
            rows.append({
                "model": f"{mode_label}/{base}",
                "metric": k,
                "value": float(v),
            })
    return rows


def extract_phase5(gen_data: dict) -> list[dict]:
    """Generalization (ISO/IgG Spearman ρ) + Pareto (Phase 5)."""
    rows = []

    # Generalization: dict keyed by model name -> dict of metrics
    for model_name, res in gen_data.get("generalization", {}).items():
        if not isinstance(res, dict):
            continue
        for metric in [
            "iso_aff_rho", "iso_spec_rho",
            "igg_aff_rho", "igg_spec_rho",
            "iso_aff_pval", "iso_spec_pval",
            "igg_aff_pval", "igg_spec_pval",
        ]:
            v = res.get(metric)
            if is_numeric(v):
                rows.append({"model": model_name, "metric": metric, "value": float(v)})

    # Pareto: nested dict keyed by dataset (emi/iso/igg) -> per-model entries
    pareto = gen_data.get("pareto", {})
    for ds_key, ds_res in pareto.items():
        if not isinstance(ds_res, dict):
            continue
        for k, v in ds_res.items():
            if is_numeric(v):
                # k might be like 'MOLM_logits_recall' or 'true_pareto_count'
                rows.append({
                    "model": f"Pareto/{ds_key}",
                    "metric": str(k),
                    "value": float(v),
                })
    return rows


PHASE_LOADERS = {
    2: ("phase2/baselines.pkl", extract_phase2),
    3: ("phase3/molm_cv.pkl", extract_phase3),
    4: ("phase4/holdout.pkl", extract_phase4),
    5: ("phase5/generalization.pkl", extract_phase5),
}


# ============================================================================
# STATS
# ============================================================================
def paired_bootstrap_pvalue(values_a, values_b, n_resamples=10000, seed=12345) -> float:
    """Two-sided paired bootstrap on mean(a - b). NaN-safe via masking."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan")
    diff = a - b
    obs = float(diff.mean())
    if obs == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    centered = diff - obs
    # Resample under H0 (centered diffs)
    boot_means = rng.choice(centered, size=(n_resamples, len(centered)), replace=True).mean(axis=1)
    p = float((np.abs(boot_means) >= abs(obs)).mean())
    return p


# ============================================================================
# MAIN
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed MOLM results into mean±std + paired bootstrap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="outputs",
                        help="The shared outputs/ dir (must contain seeds/seed_<N>/).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to aggregate (default: auto-detect from seeds/).")
    parser.add_argument("--bootstrap-n", type=int, default=10000,
                        help="Bootstrap resamples for paired comparisons.")
    parser.add_argument("--bootstrap-seed", type=int, default=12345,
                        help="RNG seed for bootstrap reproducibility.")
    parser.add_argument("--baselines", nargs="+",
                        default=["LDA (OneHot)", "LDA (ESM2)", "LDA (Fusion-ESM2)",
                                 "NN (onehot)", "NN (esm2)", "NN (fusion_esm2)"],
                        help="Model names to compare MOLM against in paired bootstrap.")
    parser.add_argument("--quiet", action="store_true",
                        help="Don't print headline grids.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    seeds_root = output_dir / "seeds"
    if not seeds_root.exists():
        print(f"FATAL: {seeds_root} not found. Run run_multiseed.py first.")
        return 1

    # Discover seeds
    if args.seeds is None:
        seed_dirs = sorted([p for p in seeds_root.iterdir()
                            if p.is_dir() and p.name.startswith("seed_")])
        seeds = []
        for p in seed_dirs:
            try:
                seeds.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                continue
        seeds = sorted(seeds)
    else:
        seeds = sorted(args.seeds)

    if not seeds:
        print(f"FATAL: no seed dirs found under {seeds_root}")
        return 1

    print(f"Aggregating {len(seeds)} seeds: {seeds}")

    # ------------------------------------------------------------------
    # Collect per-seed values: raw[(model, metric)][seed] = value
    # ------------------------------------------------------------------
    raw: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    coverage: dict[int, dict[int, str]] = {s: {} for s in seeds}

    for seed in seeds:
        seed_dir = seeds_root / f"seed_{seed}"
        for phase, (rel_path, extractor) in PHASE_LOADERS.items():
            pkl_path = seed_dir / rel_path
            if not pkl_path.exists():
                coverage[seed][phase] = "missing"
                print(f"  [seed {seed}] phase{phase}: missing")
                continue
            try:
                data = load_pickle(pkl_path)
            except Exception as e:  # noqa: BLE001
                coverage[seed][phase] = f"load_error: {e}"
                print(f"  [seed {seed}] phase{phase}: load failed — {e}")
                continue
            try:
                rows = extractor(data)
            except Exception as e:  # noqa: BLE001
                coverage[seed][phase] = f"extract_error: {e}"
                print(f"  [seed {seed}] phase{phase}: extract failed — {e}")
                continue
            for row in rows:
                raw[(row["model"], row["metric"])][seed] = row["value"]
            coverage[seed][phase] = f"{len(rows)} metrics"
            print(f"  [seed {seed}] phase{phase}: {len(rows)} metrics")

    if not raw:
        print("FATAL: no metrics extracted. Check that per-seed pickles exist.")
        return 1

    # ------------------------------------------------------------------
    # Build summary table
    # ------------------------------------------------------------------
    summary_rows = []
    for (model, metric), seed_dict in sorted(raw.items()):
        used_seeds = [s for s in seeds if s in seed_dict]
        vals = np.array([seed_dict[s] for s in used_seeds], dtype=float)
        if len(vals) == 0:
            continue
        # ddof=1 (sample std) is the right reporting choice for n>=2
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append({
            "model": model,
            "metric": metric,
            "n_seeds": int(len(vals)),
            "mean": float(np.nanmean(vals)),
            "std": std,
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "ci_lo": float(np.nanpercentile(vals, 2.5)) if len(vals) >= 3 else float("nan"),
            "ci_hi": float(np.nanpercentile(vals, 97.5)) if len(vals) >= 3 else float("nan"),
            "seeds_used": ",".join(str(s) for s in used_seeds),
            "values": ";".join(f"{v:.6f}" for v in vals),
        })
    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Paired bootstrap: MOLM (each feat) vs each baseline, per metric
    # ------------------------------------------------------------------
    bootstrap_rows = []
    molm_models = sorted({m for (m, _) in raw if m.startswith("MOLM (")})
    all_metrics = sorted({mt for (_, mt) in raw})

    for molm_model in molm_models:
        molm_metrics = {mt for (m, mt) in raw if m == molm_model}
        for baseline in args.baselines:
            base_metrics = {mt for (m, mt) in raw if m == baseline}
            common_metrics = sorted(molm_metrics & base_metrics)
            for metric in common_metrics:
                molm_seeds = raw[(molm_model, metric)]
                base_seeds = raw[(baseline, metric)]
                common = sorted(set(molm_seeds) & set(base_seeds))
                if len(common) < 3:
                    continue
                a = np.array([molm_seeds[s] for s in common], dtype=float)
                b = np.array([base_seeds[s] for s in common], dtype=float)
                p = paired_bootstrap_pvalue(a, b,
                                            n_resamples=args.bootstrap_n,
                                            seed=args.bootstrap_seed)
                bootstrap_rows.append({
                    "metric": metric,
                    "model_a": molm_model,
                    "model_b": baseline,
                    "n_seeds": len(common),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "diff_mean": float(np.mean(a) - np.mean(b)),
                    "diff_std": float(np.std(a - b, ddof=1)) if len(a) > 1 else 0.0,
                    "p_bootstrap": p,
                    "significant_05": bool(p < 0.05) if not np.isnan(p) else False,
                    "seeds_used": ",".join(str(s) for s in common),
                })
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    out_dir = output_dir / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "multiseed_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  ✓ Saved: {summary_csv} ({len(summary_df)} rows)")

    summary_json = out_dir / "multiseed_summary.json"
    with open(summary_json, "w") as f:
        json.dump({
            "seeds": seeds,
            "coverage": {str(s): cov for s, cov in coverage.items()},
            "summary": summary_df.to_dict(orient="records"),
            "raw": {f"{m}|{mt}": {str(s): v for s, v in sd.items()}
                    for (m, mt), sd in raw.items()},
        }, f, indent=2, default=str)
    print(f"  ✓ Saved: {summary_json}")

    if not bootstrap_df.empty:
        bs_csv = out_dir / "multiseed_bootstrap.csv"
        bootstrap_df.to_csv(bs_csv, index=False)
        print(f"  ✓ Saved: {bs_csv} ({len(bootstrap_df)} comparisons)")

        bs_json = out_dir / "multiseed_bootstrap.json"
        with open(bs_json, "w") as f:
            json.dump(bootstrap_rows, f, indent=2, default=str)
        print(f"  ✓ Saved: {bs_json}")

    # ------------------------------------------------------------------
    # Headline grids
    # ------------------------------------------------------------------
    if not args.quiet:
        print(f"\n{'=' * 80}")
        print(f"HEADLINE GRID  (mean ± std across {len(seeds)} seeds)")
        print(f"{'=' * 80}")
        headline_metrics = [
            "affinity_mean", "specificity_mean",
            "aff_aucs_mean", "spec_aucs_mean",
            "aff_mccs_mean", "spec_mccs_mean",
            "iso_aff_rho", "iso_spec_rho",
            "igg_aff_rho", "igg_spec_rho",
        ]
        for mt in headline_metrics:
            sub = summary_df[summary_df["metric"] == mt].sort_values("mean", ascending=False)
            if sub.empty:
                continue
            print(f"\n  [{mt}]")
            for _, r in sub.iterrows():
                print(f"    {r['model']:<35} {r['mean']:.4f} ± {r['std']:.4f}  "
                      f"(n={int(r['n_seeds'])})")

        if not bootstrap_df.empty:
            print(f"\n{'=' * 80}")
            print(f"SIGNIFICANT MOLM vs BASELINE (p<0.05, paired bootstrap n={args.bootstrap_n})")
            print(f"{'=' * 80}")
            sig = bootstrap_df[bootstrap_df["significant_05"]].sort_values("p_bootstrap")
            if sig.empty:
                print("  (no comparisons reached p<0.05 — expected for n=5 paired bootstrap)")
            else:
                shown = sig.head(30)
                for _, r in shown.iterrows():
                    print(f"  {r['metric']:<22} {r['model_a']:<22} vs {r['model_b']:<22} "
                          f"Δ={r['diff_mean']:+.4f}  p={r['p_bootstrap']:.4f}")
                if len(sig) > 30:
                    print(f"  ... ({len(sig)-30} more — see {out_dir/'multiseed_bootstrap.csv'})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
