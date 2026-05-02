# MOLM: Multi-Objective Learning for Antibody Sequence Co-Optimization

A multi-task deep learning framework for joint prediction of antibody affinity and specificity from binary deep-sequencing labels.

**Paper:** *Multi-Objective Learning for Antibody Sequence Co-Optimization: Generalization Under Sparse Binary Supervision*  
**Authors:** Diganta Das, Feng Cui, Haibo Yang  

---

## Overview

MOLM jointly predicts affinity and specificity using a shared encoder with task-specific towers, trained on the [emibetuzumab co-optimization benchmark](https://www.nature.com/articles/s41467-022-31457-3) (Makowski et al., *Nature Communications*, 2022). The pipeline evaluates three feature representations (OneHot, ESM-2, Fusion-ESM2) across four evaluation paradigms: cross-validation, mutation-site holdout, cross-platform generalization, and Pareto-front recovery.

All experiments are repeated across **five random seeds** (42, 123, 456, 789, 2024) for reproducibility.

### Key Results (5-seed, mean +/- std)

| Metric | MOLM | Best Baseline | Improvement |
| --- | --- | --- | --- |
| Holdout (combined accuracy, 5/8 sites won) | **91.1 +/- 1.2%** | 90.0 +/- 1.1% (NN) | +1.1 pp |
| Holdout MCC at hardest site (Kabat 50) | **0.291 +/- 0.072** | 0.184 +/- 0.062 (NN) | +58% |
| ISO Affinity Spearman rho | **0.884 +/- 0.010** | 0.876 +/- 0.004 (NN) | +0.008 |
| Pareto Recall (latent PCA, ISO) | **0.253 +/- 0.073** | 0.160 +/- 0.037 (NN logits) | +58% |
| Gradient alignment (% positive steps) | **89-95%** | - | Cooperative |

## Architecture

```
Input (OneHot 2300D / ESM2 320D / Fusion-ESM2 2620D)
        |
  Shared Encoder (256 -> 128, LayerNorm, GELU, Dropout 0.2)
        |
   +----+----+
   |         |
Affinity   Specificity
 Tower       Tower
(64->32->16)(64->32->16)
   |         |
Latent 16D  Latent 16D  --> PCA (train-fit) --> Pareto Recovery
   |         |
 Logit      Logit
   |         |
 y_aff     y_spec
```

**Training Objective** (per task): Focal Loss (gamma=2) + Ranking Loss (margin=1.0) + Gap Hinge Loss (margin=1.0)  
**Configuration**: Simple MTL (ADV_WEIGHT=0, ORTHO_WEIGHT=0)  
**Optional**: Pareto-diversity regularizer (lambda=0.1, 5-epoch warmup) - evaluated but not recommended

---

## Pipeline Structure

### Core Pipeline (6 phases)

| Phase | File | Description | Output |
| --- | --- | --- | --- |
| 0 | `phase0_config.py` | Configuration, model definitions, losses, metrics | - |
| 1 | `phase1_features.py` | Data loading, ESM-2 embedding computation | `features.pkl` |
| 2 | `phase2_baselines.py` | LDA + NN baselines (5-fold CV) | `baselines.pkl` |
| 3 | `phase3_molm_cv.py` | MOLM + MOLM-ST cross-validation (full feature grid) | `molm_cv.pkl` |
| 4 | `phase4_holdout.py` | Mutation-site holdout (8 sites x all models) | `holdout.pkl` |
| 5 | `phase5_generalization.py` | Cross-platform generalization + Pareto recovery | `generalization.pkl` |

### Multi-Seed Orchestration

| Script | Description |
| --- | --- |
| `run_multiseed.py` | Run all phases across 5 seeds (Pareto loss OFF) |
| `run_pareto_on.py` | Run Phase 3+5 with Pareto loss ON across 5 seeds |

### Result Extraction

| Script | Output CSV |
| --- | --- |
| `aggregate_multiseed.py` | `multiseed_summary.csv` |
| `extract_phase2_results.py` | `phase2_baselines.csv` |
| `extract_phase3_results.py` | `phase3_molm_cv.csv` |
| `extract_phase5_results.py` | `phase5_generalization.csv`, `phase5_pareto.csv` |
| `extract_persite_holdout.py` | `persite_holdout.csv` |
| `extract_gradient_diagnostics.py` | `gradient_cosine.csv` |
| `extract_pareto_on_results.py` | `pareto_on_vs_off.csv` |
| `extract_phase3_pareto_on.py` | `phase3_pareto_on_vs_off_raw.csv` |

### Verification Scripts

| Script | Description |
| --- | --- |
| `verify_sequence_homology.py` | EMI sequence identity, variable positions, CD-HIT simulation |
| `verify_homology_fixed.py` | Cross-dataset overlap: ISO vs EMI, IgG vs EMI |

### Visualization

| Script | Description |
| --- | --- |
| `visualize_gradient_convergence.py` | 9-panel convergence figure |
| `visualize_pareto_aggregate.py` | Pareto recall barplot, per-seed dotplot, precision-recall scatter |
| `find_pareto_seed.py` | Find median-recall seed for representative diagnostics |
| `collect_images.py` | Organize figures into `figures/` directory |

---

## Setup and Usage

### Requirements

```
Python 3.10+
PyTorch (cu128 for RTX 5060 Ti, or CPU)
TensorFlow 2.x
scikit-learn
NumPy, Pandas
fair-esm (for ESM-2 embeddings)
```

### Running on Local Machine (Windows + GPU)

```bash
# Create environment
conda create -n molm python=3.10
conda activate molm
pip install torch tensorflow scikit-learn pandas fair-esm

# Run full 5-seed pipeline (Pareto OFF - primary results)
python run_multiseed.py

# Run Pareto ON ablation (Phase 3+5 only)
python run_pareto_on.py

# Extract all results
python aggregate_multiseed.py
python extract_persite_holdout.py
python extract_phase5_results.py
python extract_gradient_diagnostics.py
python extract_pareto_on_results.py

# Verify sequence homology (for journal requirements)
python verify_sequence_homology.py
python verify_homology_fixed.py
```

### Dataset

The pipeline expects the [emibetuzumab dataset](https://github.com/Tessier-Lab-UMich/Emi_Pareto_Opt_ML) files in `data/`:

| File | Description | Sequences |
| --- | --- | --- |
| `emi_binding.csv` | EMI binary labels (training) | 4,000 |
| `iso_binding.csv` | ISO continuous measurements (eval) | 126 |
| `igg_binding.csv` | IgG continuous measurements (eval) | 96 (42 used) |
| `*_reps.csv` | Sequence representations | - |
| `residue_dict.csv` | Residue mapping | - |

**Sequence homology:** All EMI sequences are 115 residues with >91.3% pairwise identity (8 designed CDR positions vary). ISO sequences have zero exact overlap with EMI; 24/96 IgG sequences match EMI exactly. See [Makowski et al. (2022)](https://doi.org/10.1038/s41467-022-31457-3) for dataset details.

---

## Key Findings

1. **MOLM improves design-relevant extrapolation** - wins/ties at 5/8 held-out mutation sites; advantage increases at harder sites (r = -0.535)
2. **Cooperative gradient structure** - affinity and specificity gradients aligned in 89-95% of training steps; all 15 per-seed means positive
3. **Latent-space Pareto recovery** - MOLM latent PCA recall 0.253 vs logit recall 0.080; trade-off geometry captured in shared representations
4. **Simple MTL wins** - adversarial regularization and Pareto-diversity loss both degrade primary metrics
5. **Feature-task specificity** - OneHot best for holdout, ESM2 best for cross-platform affinity, Fusion-ESM2 balanced

---

## Output Structure

```
outputs/
├── phase1/                    # Shared features (deterministic)
├── seeds/                     # 5-seed results (Pareto OFF)
│   ├── seed_42/
│   │   ├── phase2/
│   │   ├── phase3/
│   │   ├── phase4/
│   │   └── phase5/
│   ├── seed_123/
│   ├── seed_456/
│   ├── seed_789/
│   └── seed_2024/
├── seeds_pareto_on/           # 5-seed results (Pareto ON ablation)
├── multiseed/                 # Aggregated CSVs
└── pareto_on_logs/            # Run logs
```

---

## Citation

```bibtex
@article{das2026molm,
  title={Multi-Objective Learning for Antibody Sequence Co-Optimization: 
         Generalization Under Sparse Binary Supervision},
  author={Das, Diganta and Cui, Feng and Yang, Haibo},
  journal={Bioinformatics},
  year={2026}
}
```

## References

- Makowski, E.K. et al. (2022). Co-optimization of therapeutic antibody affinity and specificity using machine learning models that generalize to novel mutational space. *Nature Communications*, 13:3788.
- Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637):1123-1130.

## License

MIT License
