# Data

This directory contains the curated emibetuzumab benchmark data used by the
MOLM reproduction pipeline. These are sequence-level benchmark files, not
aggregated summary tables.

Source: Makowski et al. (2022), BioProject `PRJNA850089`, and
`Tessier-Lab-UMich/Emi_Pareto_Opt_ML`.

## Files

| File | Rows | Description |
| --- | ---: | --- |
| `emi_binding.csv` | 4,000 | EMI training labels: target `ANT Binding` and off-target `OVA Binding`. |
| `emi_reps.csv` | 4,000 | Precomputed 64-dimensional sequence representations for EMI. |
| `iso_binding.csv` | 126 | Independent ISO yeast-display continuous measurements. |
| `iso_reps.csv` | 126 | Precomputed sequence representations for ISO. |
| `igg_binding.csv` | 96 | Independent soluble IgG continuous measurements and metadata. |
| `igg_reps.csv` | 96 | Precomputed sequence representations for IgG. |
| `residue_dict.csv` | 20 | Amino-acid property annotations. |

Row counts exclude headers.

## Usage

`phase1_features.py` loads the `*_binding.csv` files and matching `*_reps.csv`
files. EMI is used for training, cross-validation, and residue-level grouped
mutation holdout. ISO and IgG are independent evaluation datasets.

`ANT Binding` is target binding. `OVA Binding` is off-target ovalbumin binding;
lower OVA binding is the desired specificity direction.
