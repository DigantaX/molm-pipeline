"""
extract_phase3_pareto_on.py — Extract Phase 3 CV from Pareto ON runs
"""
import pickle, io, sys
from pathlib import Path
import numpy as np
import pandas as pd

FEAT_LABELS = {'onehot': 'OneHot', 'esm2': 'ESM2', 'fusion_esm2': 'Fusion-ESM2'}

def load_pickle(path):
    try:
        import torch
        orig = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        try:
            with open(path, "rb") as f: return pickle.load(f)
        finally:
            torch.storage._load_from_bytes = orig
    except ImportError:
        with open(path, "rb") as f: return pickle.load(f)

seeds = [42, 123, 456, 789, 2024]
output_dir = Path("outputs").resolve()
out_dir = output_dir / "multiseed"

all_rows = []
for condition, subdir in [("OFF", "seeds"), ("ON", "seeds_pareto_on")]:
    for seed in seeds:
        pkl = output_dir / subdir / f"seed_{seed}" / "phase3" / "molm_cv.pkl"
        if not pkl.exists():
            print(f"  [{condition}] seed {seed}: not found")
            continue
        data = load_pickle(pkl)
        for prefix, cv_key in [("MOLM", "molm_cv"), ("MOLM-ST", "molm_st_cv")]:
            cv_data = data.get(cv_key, {})
            if not isinstance(cv_data, dict): continue
            for feat_type, res in cv_data.items():
                if not isinstance(res, dict): continue
                fl = FEAT_LABELS.get(feat_type, feat_type)
                for key, val in res.items():
                    if isinstance(val, (int, float, np.floating, np.integer)):
                        all_rows.append({
                            "seed": seed, "pareto_loss": condition,
                            "model": f"{prefix} ({fl})", "metric": key, "value": float(val)
                        })
        print(f"  [{condition}] seed {seed}: OK")

df = pd.DataFrame(all_rows)
df.to_csv(out_dir / "phase3_pareto_on_vs_off_raw.csv", index=False)

# Print summary for key metrics
print(f"\n{'='*90}")
print(f"  PHASE 3 CV — Pareto Loss ON vs OFF (5 seeds)")
print(f"{'='*90}")

for model in sorted(df["model"].unique()):
    if "MOLM" not in model: continue
    print(f"\n  {model}:")
    for cond in ["OFF", "ON"]:
        sub = df[(df["model"] == model) & (df["pareto_loss"] == cond)]
        aff = sub[sub["metric"] == "affinity_mean"]["value"]
        spec = sub[sub["metric"] == "specificity_mean"]["value"]
        aff_mcc = sub[sub["metric"] == "aff_mccs_mean"]["value"]
        spec_mcc = sub[sub["metric"] == "spec_mccs_mean"]["value"]
        
        def fmt(s):
            if len(s) == 0: return "—"
            return f"{s.mean():.4f}+/-{s.std():.4f}"
        
        print(f"    {cond:>4}: Aff Acc={fmt(aff)}  Spec Acc={fmt(spec)}  Aff MCC={fmt(aff_mcc)}  Spec MCC={fmt(spec_mcc)}")

print(f"\n  Saved: {out_dir / 'phase3_pareto_on_vs_off_raw.csv'}")
