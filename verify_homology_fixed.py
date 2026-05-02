"""
verify_homology_fixed.py — Check ISO/IgG vs EMI overlap (fixed dataset IDs)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import random

random.seed(42)
data_dir = Path(__file__).parent.resolve() / "data"

# Load datasets explicitly
emi = pd.read_csv(data_dir / "emi_binding.csv")["VH Sequence"].dropna().tolist()
iso = pd.read_csv(data_dir / "iso_binding.csv")["VH Sequence"].dropna().tolist()
igg = pd.read_csv(data_dir / "igg_binding.csv")["VH Sequence"].dropna().tolist()

print(f"EMI: {len(emi)} sequences")
print(f"ISO: {len(iso)} sequences")
print(f"IgG: {len(igg)} sequences")

emi_set = set(emi)
iso_set = set(iso)
igg_set = set(igg)

# ISO vs EMI
iso_in_emi = iso_set & emi_set
iso_not_emi = iso_set - emi_set
print(f"\n{'='*70}")
print(f"  ISO vs EMI")
print(f"{'='*70}")
print(f"  Exact matches: {len(iso_in_emi)}/{len(iso_set)} ({len(iso_in_emi)/len(iso_set)*100:.1f}%)")
print(f"  Unique to ISO: {len(iso_not_emi)}")

if iso_not_emi:
    ids = []
    for s1 in iso_not_emi:
        best = max(sum(a==b for a,b in zip(s1,s2))/max(len(s1),len(s2)) for s2 in random.sample(list(emi_set), min(500, len(emi_set))))
        ids.append(best)
    print(f"  Identity of unique ISO seqs to nearest EMI: {min(ids)*100:.1f}%–{max(ids)*100:.1f}% (mean {np.mean(ids)*100:.1f}%)")

# IgG vs EMI
igg_in_emi = igg_set & emi_set
igg_not_emi = igg_set - emi_set
print(f"\n{'='*70}")
print(f"  IgG vs EMI")
print(f"{'='*70}")
print(f"  Exact matches: {len(igg_in_emi)}/{len(igg_set)} ({len(igg_in_emi)/len(igg_set)*100:.1f}%)")
print(f"  Unique to IgG: {len(igg_not_emi)}")

if igg_not_emi:
    ids = []
    for s1 in list(igg_not_emi):
        best = max(sum(a==b for a,b in zip(s1,s2))/max(len(s1),len(s2)) for s2 in random.sample(list(emi_set), min(500, len(emi_set))))
        ids.append(best)
    print(f"  Identity of unique IgG seqs to nearest EMI: {min(ids)*100:.1f}%–{max(ids)*100:.1f}% (mean {np.mean(ids)*100:.1f}%)")

# ISO vs IgG
iso_igg = iso_set & igg_set
print(f"\n{'='*70}")
print(f"  ISO vs IgG")
print(f"{'='*70}")
print(f"  Shared sequences: {len(iso_igg)}")
print(f"    ({len(iso_igg)/len(iso_set)*100:.1f}% of ISO, {len(iso_igg)/len(igg_set)*100:.1f}% of IgG)")

# All three
all_three = emi_set & iso_set & igg_set
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  Sequences in all three datasets: {len(all_three)}")
print(f"  ISO is {'a SUBSET' if iso_set.issubset(emi_set) else 'NOT a subset'} of EMI")
print(f"  IgG is {'a SUBSET' if igg_set.issubset(emi_set) else 'NOT a subset'} of EMI")

if not igg_set.issubset(emi_set):
    print(f"\n  NOTE: {len(igg_not_emi)} IgG sequences are NOT in EMI training data.")
    print(f"  This means IgG evaluation includes genuinely unseen sequences.")
    print(f"  Paper should mention this distinction between ISO (subset) and IgG (partial overlap).")
