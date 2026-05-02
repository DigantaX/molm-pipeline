"""
extract_pareto_on_results.py — Compare Pareto Loss ON vs OFF across 5 seeds
=============================================================================
Reads Phase 5 pickles from both seeds/ (OFF) and seeds_pareto_on/ (ON),
and produces a side-by-side comparison CSV + formatted report.

USAGE:
    python extract_pareto_on_results.py

OUTPUT:
    outputs/multiseed/pareto_on_vs_off.csv
    outputs/multiseed/pareto_on_vs_off_generalization.csv
"""

from __future__ import annotations
import pickle, io, sys
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


def extract_from_pkl(pkl_path: Path) -> tuple[list[dict], list[dict]]:
    data = load_pickle(pkl_path)
    gen_rows = []
    pareto_rows = []

    for model_name, res in data.get("generalization", {}).items():
        if not isinstance(res, dict):
            continue
        row = {"model": model_name}
        for metric in ["iso_aff_rho", "iso_spec_rho", "igg_aff_rho", "igg_spec_rho"]:
            v = res.get(metric)
            if v is not None and isinstance(v, (int, float, np.floating)):
                row[metric] = float(v)
        gen_rows.append(row)

    for ds_key, ds_res in data.get("pareto", {}).items():
        if not isinstance(ds_res, dict):
            continue
        for model_key, model_res in ds_res.items():
            if isinstance(model_res, dict):
                pareto_rows.append({
                    "dataset": ds_key, "model": model_key,
                    "predicted": model_res.get("predicted"),
                    "overlap": model_res.get("overlap"),
                    "precision": model_res.get("precision"),
                    "recall": model_res.get("recall"),
                    "score_type": model_res.get("score_type", "logits"),
                })

    return gen_rows, pareto_rows


def main() -> int:
    output_dir = Path("outputs").resolve()
    out_dir = output_dir / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456, 789, 2024]

    all_gen = []
    all_pareto = []

    for condition, subdir in [("OFF", "seeds"), ("ON", "seeds_pareto_on")]:
        for seed in seeds:
            pkl_path = output_dir / subdir / f"seed_{seed}" / "phase5" / "generalization.pkl"
            if not pkl_path.exists():
                print(f"  [{condition}] seed {seed}: not found ({pkl_path})")
                continue

            gen_rows, pareto_rows = extract_from_pkl(pkl_path)
            for r in gen_rows:
                r["seed"] = seed
                r["pareto_loss"] = condition
            for r in pareto_rows:
                r["seed"] = seed
                r["pareto_loss"] = condition
            all_gen.extend(gen_rows)
            all_pareto.extend(pareto_rows)
            print(f"  [{condition}] seed {seed}: {len(gen_rows)} gen, {len(pareto_rows)} pareto")

    if not all_gen and not all_pareto:
        print("No data found!")
        return 1

    # Generalization comparison
    if all_gen:
        gen_df = pd.DataFrame(all_gen)
        gen_df.to_csv(out_dir / "pareto_on_vs_off_generalization_raw.csv", index=False)

        rho_cols = [c for c in gen_df.columns if c.endswith("_rho")]
        gen_agg = gen_df.groupby(["pareto_loss", "model"])[rho_cols].agg(["mean", "std"]).reset_index()
        gen_agg.columns = ["_".join(c).rstrip("_") for c in gen_agg.columns]
        gen_agg.to_csv(out_dir / "pareto_on_vs_off_generalization.csv", index=False)

        print(f"\n{'='*90}")
        print(f"  GENERALIZATION — Pareto Loss ON vs OFF (5 seeds)")
        print(f"{'='*90}")

        # Show MOLM models only (baselines unaffected)
        for model in sorted(gen_df["model"].unique()):
            if "MOLM" not in model:
                continue
            print(f"\n  {model}:")
            for cond in ["OFF", "ON"]:
                sub = gen_df[(gen_df["model"] == model) & (gen_df["pareto_loss"] == cond)]
                if sub.empty:
                    continue
                iso_a = f"{sub['iso_aff_rho'].mean():.3f}+/-{sub['iso_aff_rho'].std():.3f}" if "iso_aff_rho" in sub else "—"
                iso_s = f"{sub['iso_spec_rho'].mean():.3f}+/-{sub['iso_spec_rho'].std():.3f}" if "iso_spec_rho" in sub else "—"
                igg_a = f"{sub['igg_aff_rho'].mean():.3f}+/-{sub['igg_aff_rho'].std():.3f}" if "igg_aff_rho" in sub else "—"
                igg_s = f"{sub['igg_spec_rho'].mean():.3f}+/-{sub['igg_spec_rho'].std():.3f}" if "igg_spec_rho" in sub else "—"
                print(f"    {cond:>4}: ISO Aff={iso_a}  ISO Spec={iso_s}  IgG Aff={igg_a}  IgG Spec={igg_s}")

    # Pareto comparison
    if all_pareto:
        pareto_df = pd.DataFrame(all_pareto)
        pareto_df.to_csv(out_dir / "pareto_on_vs_off_raw.csv", index=False)

        pareto_agg = pareto_df.groupby(["pareto_loss", "dataset", "model", "score_type"]).agg(
            recall_mean=("recall", "mean"),
            recall_std=("recall", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
            precision_mean=("precision", "mean"),
            overlap_mean=("overlap", "mean"),
            predicted_mean=("predicted", "mean"),
        ).reset_index()
        pareto_agg.to_csv(out_dir / "pareto_on_vs_off.csv", index=False)

        print(f"\n{'='*90}")
        print(f"  PARETO — ON vs OFF (5 seeds)")
        print(f"{'='*90}")

        for ds in ["ISO", "IgG"]:
            sub = pareto_agg[pareto_agg["dataset"] == ds]
            if sub.empty:
                continue
            true_n = 15 if ds == "ISO" else 5
            print(f"\n  {ds} ({true_n} true Pareto):")
            print(f"  {'Condition':>6} {'Model':<25} {'Score':>12} {'Recall':>14} {'Precision':>10}")
            print(f"  {'-'*72}")

            # Show ON and OFF side by side for MOLM models
            for model in sorted(sub["model"].unique()):
                if "MOLM" not in model:
                    continue
                for cond in ["OFF", "ON"]:
                    rows = sub[(sub["model"] == model) & (sub["pareto_loss"] == cond)]
                    for _, r in rows.iterrows():
                        rec = f"{r['recall_mean']:.3f}+/-{r['recall_std']:.3f}" if r["recall_std"] > 0 else f"{r['recall_mean']:.3f}"
                        print(f"  {cond:>6} {r['model']:<25} {r['score_type']:>12} {rec:>14} {r['precision_mean']:>10.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
