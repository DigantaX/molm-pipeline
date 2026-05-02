"""
extract_phase5_results.py — Phase 5 results across all seeds
===============================================================
Extracts generalization (Spearman rho) + Pareto (recall/precision)
from all seed pickles and saves as clean CSVs.

USAGE:
    python extract_phase5_results.py

OUTPUT:
    outputs/multiseed/phase5_generalization.csv   (ISO/IgG Spearman per model)
    outputs/multiseed/phase5_pareto.csv           (Pareto recall/precision per model)
"""

from __future__ import annotations
import argparse, pickle, io, sys
from pathlib import Path
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

    print(f"Extracting Phase 5 from {len(seeds)} seeds: {seeds}")

    # ---- Generalization ----
    gen_rows = []
    pareto_rows = []

    for seed in seeds:
        pkl_path = seeds_root / f"seed_{seed}" / "phase5" / "generalization.pkl"
        if not pkl_path.exists():
            print(f"  [seed {seed}] not found, skipping")
            continue

        data = load_pickle(pkl_path)

        # Generalization: Spearman rho
        for model_name, res in data.get("generalization", {}).items():
            if not isinstance(res, dict):
                continue
            row = {"seed": seed, "model": model_name}
            for metric in ["iso_aff_rho", "iso_spec_rho", "igg_aff_rho", "igg_spec_rho",
                           "iso_aff_pval", "iso_spec_pval", "igg_aff_pval", "igg_spec_pval",
                           "iso_aff_ci_lo", "iso_aff_ci_hi", "iso_spec_ci_lo", "iso_spec_ci_hi",
                           "igg_aff_ci_lo", "igg_aff_ci_hi", "igg_spec_ci_lo", "igg_spec_ci_hi"]:
                v = res.get(metric)
                if v is not None and isinstance(v, (int, float, np.floating, np.integer)):
                    row[metric] = float(v)
            gen_rows.append(row)

        # Pareto
        for ds_key, ds_res in data.get("pareto", {}).items():
            if not isinstance(ds_res, dict):
                continue
            for model_key, model_res in ds_res.items():
                if isinstance(model_res, dict):
                    row = {
                        "seed": seed,
                        "dataset": ds_key,
                        "model": model_key,
                        "predicted": model_res.get("predicted"),
                        "overlap": model_res.get("overlap"),
                        "precision": model_res.get("precision"),
                        "recall": model_res.get("recall"),
                        "score_type": model_res.get("score_type", "logits"),
                    }
                    pareto_rows.append(row)
                elif model_key == "true_pareto_count":
                    # Store as metadata
                    pareto_rows.append({
                        "seed": seed, "dataset": ds_key, "model": "_TRUE_PARETO_",
                        "predicted": int(model_res), "overlap": int(model_res),
                        "precision": 1.0, "recall": 1.0, "score_type": "ground_truth",
                    })

        print(f"  [seed {seed}] gen: {len([r for r in gen_rows if r['seed']==seed])} models, "
              f"pareto: {len([r for r in pareto_rows if r['seed']==seed])} entries")

    # ---- Build generalization CSV ----
    gen_df = pd.DataFrame(gen_rows)
    if not gen_df.empty:
        # Aggregate across seeds
        rho_cols = [c for c in gen_df.columns if c.endswith("_rho")]
        gen_agg = gen_df.groupby("model")[rho_cols].agg(["mean", "std", "count"]).reset_index()
        # Flatten multi-level columns
        gen_agg.columns = [f"{a}_{b}" if b else a for a, b in gen_agg.columns]

        # Also save raw per-seed values
        out_dir = output_dir / "multiseed"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_gen_path = out_dir / "phase5_generalization_raw.csv"
        gen_df.to_csv(raw_gen_path, index=False)

        agg_gen_path = out_dir / "phase5_generalization.csv"
        gen_agg.to_csv(agg_gen_path, index=False)

        print(f"\n  Saved: {raw_gen_path} ({len(gen_df)} rows)")
        print(f"  Saved: {agg_gen_path} ({len(gen_agg)} models)")

        # Print summary
        print(f"\n{'='*85}")
        print(f"  GENERALIZATION — Spearman rho (mean +/- std, {len(seeds)} seeds)")
        print(f"{'='*85}")
        print(f"  {'Model':<30} {'ISO Aff':>12} {'ISO Spec':>12} {'IgG Aff':>12} {'IgG Spec':>12}")
        print(f"  {'-'*82}")

        for _, r in gen_df.groupby("model").agg(
            iso_aff=("iso_aff_rho", "mean"), iso_aff_s=("iso_aff_rho", "std"),
            iso_spec=("iso_spec_rho", "mean"), iso_spec_s=("iso_spec_rho", "std"),
            igg_aff=("igg_aff_rho", "mean"), igg_aff_s=("igg_aff_rho", "std"),
            igg_spec=("igg_spec_rho", "mean"), igg_spec_s=("igg_spec_rho", "std"),
        ).sort_values("iso_aff", ascending=False).iterrows():
            def fmt(m, s):
                if pd.isna(m): return "    —     "
                if pd.isna(s) or s == 0: return f"{m:.3f}      "
                return f"{m:.3f}+/-{s:.3f}"
            print(f"  {_:<30} {fmt(r['iso_aff'],r['iso_aff_s']):>12} "
                  f"{fmt(r['iso_spec'],r['iso_spec_s']):>12} "
                  f"{fmt(r['igg_aff'],r['igg_aff_s']):>12} "
                  f"{fmt(r['igg_spec'],r['igg_spec_s']):>12}")

    # ---- Build Pareto CSV ----
    pareto_df = pd.DataFrame(pareto_rows)
    if not pareto_df.empty:
        out_dir = output_dir / "multiseed"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_pareto_path = out_dir / "phase5_pareto_raw.csv"
        pareto_df.to_csv(raw_pareto_path, index=False)

        # Aggregate across seeds (exclude ground truth rows)
        pareto_models = pareto_df[pareto_df["model"] != "_TRUE_PARETO_"]
        if not pareto_models.empty:
            pareto_agg = pareto_models.groupby(["dataset", "model", "score_type"]).agg(
                n_seeds=("recall", "count"),
                recall_mean=("recall", "mean"),
                recall_std=("recall", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
                precision_mean=("precision", "mean"),
                precision_std=("precision", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
                overlap_mean=("overlap", "mean"),
                predicted_mean=("predicted", "mean"),
            ).reset_index().sort_values(["dataset", "recall_mean"], ascending=[True, False])

            agg_pareto_path = out_dir / "phase5_pareto.csv"
            pareto_agg.to_csv(agg_pareto_path, index=False)

            print(f"\n  Saved: {raw_pareto_path} ({len(pareto_df)} rows)")
            print(f"  Saved: {agg_pareto_path} ({len(pareto_agg)} model-score combos)")

            # Print summary
            for ds in ["ISO", "IgG"]:
                true_count = pareto_df[(pareto_df["dataset"] == ds) & 
                                       (pareto_df["model"] == "_TRUE_PARETO_")]["predicted"].iloc[0] if \
                             len(pareto_df[(pareto_df["dataset"] == ds) & (pareto_df["model"] == "_TRUE_PARETO_")]) > 0 else "?"
                sub = pareto_agg[pareto_agg["dataset"] == ds]
                print(f"\n{'='*85}")
                print(f"  PARETO — {ds} ({true_count} true Pareto, mean +/- std, {len(seeds)} seeds)")
                print(f"{'='*85}")
                print(f"  {'Model':<25} {'Score':>12} {'Recall':>14} {'Precision':>14} {'Overlap':>10} {'Predicted':>10}")
                print(f"  {'-'*87}")
                for _, r in sub.iterrows():
                    rec = f"{r['recall_mean']:.3f}+/-{r['recall_std']:.3f}" if r['recall_std'] > 0 else f"{r['recall_mean']:.3f}"
                    prec = f"{r['precision_mean']:.3f}+/-{r['precision_std']:.3f}" if r['precision_std'] > 0 else f"{r['precision_mean']:.3f}"
                    print(f"  {r['model']:<25} {r['score_type']:>12} {rec:>14} {prec:>14} {r['overlap_mean']:>10.1f} {r['predicted_mean']:>10.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
