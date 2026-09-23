# MOLM: Multi-Objective Learning for Computational Antibody Sequence Co-Optimization

Code and reproducibility notebooks for the manuscript:

**Multi-Objective Learning for Computational Antibody Sequence Co-Optimization: Generalization Under Sparse Binary Supervision**

**Authors:** Diganta Das, Weikang Fu, Feng Cui, Haibo Yang

## Overview

This repository studies computational co-optimization of emibetuzumab VH variants under two assay-derived objectives:

- **target-binding proxy** â€” a binary training endpoint derived from the source library-sorting workflow, not a direct thermodynamic affinity measurement;
- **OVA-binding proxy** â€” a binary training endpoint used as a limited off-target/nonspecific-binding proxy, not a comprehensive measure of specificity or polyspecificity.

The 4,000-sequence EMI dataset is used for model training. Continuous target- and OVA-binding measurements from ISO and IgG panels are used for external evaluation within the same emibetuzumab scaffold.

The revised evaluation compares:

- **Standard-MOLM** â€” conventional hard parameter sharing with task-specific towers;
- **MOLM-ST** â€” independent same-loss control using the same focal + ranking + gap task objective;
- **Routed-MOLM** â€” shared backbone with task-private adapters, conditional shared-gradient projection, and a low-weight dominance-aware ordering term;
- **NN** â€” neural single-task baseline;
- **LDA** â€” linear baseline.

Five optimization seeds are used throughout: `42, 123, 456, 789, 2024`. Seeds quantify optimization variability rather than biological replication.

## Representations

| Representation | Description | Dimensionality |
| --- | --- | ---: |
| OneHot | flattened sequence one-hot encoding | 2,300 |
| Mean-ESM2 | whole-VH mean-pooled ESM-2 representation | 320 |
| Mean-Fusion | OneHot + Mean-ESM2 | 2,620 |
| Site-ESM2 | ESM-2 representations at the eight experimentally varied CDR positions | 2,560 |
| Site-Fusion | OneHot + Site-ESM2 | 4,860 |

## Evaluation suite

The repository covers:

1. 5-fold cross-validation;
2. grouped residue mutation holdout;
3. Hamming-distance-stratified within-scaffold generalization;
4. ISO and IgG continuous-binding transfer;
5. fixed-budget Pareto candidate prioritization at `K={5,10,15,20,25}`;
6. same-loss Standard-MOLM versus MOLM-ST comparisons;
7. Routed-MOLM component/capacity ablations;
8. ranking/gap loss ablations;
9. latent-PCA robustness for Standard-MOLM;
10. publication figure generation.

For the measured external objective values, the ISO reference Pareto front contains **15 of 126 variants**, while the secondary IgG-all96 reference front contains **4 of 96 variants**. Exact-front Recall is therefore much coarser for IgG-all96 and is interpreted together with hypervolume (HV) and inverted generational distance (IGD).

## Revision / extended-analysis notebooks

The cleaned notebooks supporting the current manuscript are in [`notebooks/`](notebooks/):

| Notebook | Purpose |
| --- | --- |
| `01_unified_experiment_suite.ipynb` | Standard-MOLM extension, mutation/external statistics, fixed-budget Pareto, latent-PCA, A-H controls, loss ablation, and Hamming analysis |
| `02_molm_st_mutation_holdout.ipynb` | dedicated original MOLM-ST mutation-holdout evaluation |
| `03_molm_st_fixed_budget_pareto.ipynb` | original MOLM-ST fixed-budget Pareto and external evaluation |
| `04_component_architecture_ablation.ipynb` | architecture/component and ranking/gap ablations under fixed-budget Pareto evaluation |
| `05_publication_figure_generator.ipynb` | regeneration of manuscript and supplementary figures from finalized result bundles |

The notebooks are intentionally written as scientific/reproducibility workflows and contain no response-to-review process annotations. Notebook outputs are cleared in the repository; running the notebooks recreates the analysis artifacts.

### Pinned core pipeline

The extended notebooks pin the original core-pipeline commit:

```text
c5923984f0d5176977edb4a4ffd8fc5f98536043
```

This preserves the original data-processing and model definitions while the notebooks define the additional controlled experiments and analysis procedures explicitly.

## Main interpretation

The current results do **not** support a universal marginal-prediction advantage from hard parameter sharing. Standard-MOLM and MOLM-ST are broadly comparable across many marginal endpoints, and model ordering varies by task, representation, dataset, and screening budget.

Two recurring findings are:

- residue-focused ESM-2 representations substantially improve neural mutation extrapolation relative to whole-VH mean pooling in several settings;
- Routed-MOLM shows favorable mean multi-objective prioritization in selected fixed-budget regimes, most consistently in the secondary IgG-all96 OneHot analysis, but the five-seed contrasts do not remain significant after multiplicity correction.

Accordingly, fixed-budget Pareto results are treated as conditional effect-size patterns rather than universal superiority claims.

## Representative values from the current analysis

These values are included for orientation; complete results and uncertainty estimates are produced by the notebooks and reported in the manuscript/supplement.

| Evaluation | Example result |
| --- | --- |
| Standard-MOLM Mean-ESM2 â†’ Site-ESM2, target-proxy mutation MCC | `0.610 -> 0.737` |
| Standard-MOLM Mean-ESM2 â†’ Site-ESM2, OVA-proxy mutation MCC | `0.665 -> 0.746` |
| ISO Mean-ESM2 target-binding Spearman, Standard-MOLM | `0.884 +/- 0.005` |
| ISO Mean-Fusion, K=20, Routed-MOLM Recall / HV / IGD | `0.307 / 0.664 / 0.0588` |
| ISO Mean-Fusion, K=20, MOLM-ST Recall / HV / IGD | `0.293 / 0.659 / 0.0643` |
| IgG-all96 OneHot, K=20, Routed-MOLM Recall / HV / IGD | `0.450 / 0.922 / 0.0500` |
| IgG-all96 OneHot, K=20, MOLM-ST Recall / HV / IGD | `0.350 / 0.903 / 0.0853` |

The ISO ordering changes with `K`. The IgG-all96 analysis is secondary and has a four-variant measured reference front, so its exact-front Recall is interpreted jointly with HV and IGD.

## Core pipeline

The original phase-based pipeline remains available at the repository root.

| Phase | File | Description |
| --- | --- | --- |
| 0 | `phase0_config.py` | configuration, model definitions, losses, and metrics |
| 1 | `phase1_features.py` | data loading and ESM-2 feature computation |
| 2 | `phase2_baselines.py` | LDA and NN baselines |
| 3 | `phase3_molm_cv.py` | Standard-MOLM / MOLM-ST cross-validation |
| 4 | `phase4_holdout.py` | mutation-site holdout |
| 5 | `phase5_generalization.py` | external generalization and original Pareto analyses |

The revision notebooks in `notebooks/` implement the additional controlled experiments used in the current manuscript.

## Data

The benchmark data were originally reported by Makowski et al. (2022) and are available from:

- BioProject **PRJNA850089**;
- [Tessier-Lab-UMich/Emi_Pareto_Opt_ML](https://github.com/Tessier-Lab-UMich/Emi_Pareto_Opt_ML).

Expected benchmark files include:

| File | Role |
| --- | --- |
| `emi_binding.csv` | 4,000 binary-labeled training sequences |
| `iso_binding.csv` | 126 continuous external measurements |
| `igg_binding.csv` | 96 continuous IgG measurements; the 42-sequence primary panel is also analyzed separately |
| representation/cache files | reusable sequence encodings used by the notebooks |

All sequences belong to the emibetuzumab VH scaffold. The external evaluations therefore do not establish cross-scaffold or cross-antigen generalization.

## Environment

Typical dependencies used across the core pipeline and notebooks include:

```text
Python 3.10+
PyTorch
NumPy
Pandas
SciPy
scikit-learn
Matplotlib
fair-esm
TensorFlow 2.x  # used by parts of the original pipeline
```

For the large extended experiment notebooks, the tested workflow uses a Kaggle **T4 x2** accelerator and resumable intermediate result bundles.

## Basic usage

The current analysis is provided as reproducibility notebooks in the [`notebooks/`](notebooks/) directory.

### Environment

```bash
conda create -n molm python=3.10
conda activate molm
pip install torch scikit-learn scipy pandas numpy matplotlib fair-esm jupyter
```

See [`notebooks/README.md`](notebooks/README.md) for the purpose, inputs, and outputs of each workflow.
## Statistical notes

- Optimization seeds are not treated as biological replicates.
- Mutation inference uses the predefined mutation-site blocks where appropriate.
- External model comparisons use paired sequence bootstrap uncertainty.
- Fixed-budget neural Pareto selection is performed separately for each optimization seed before across-seed summaries are calculated; predictions are not averaged across seeds before Pareto selection.
- Multiple-testing correction is applied to the planned comparison families described in the manuscript/supplement.

## Citation

The manuscript associated with this repository is:

> Das D, Fu W, Cui F, Yang H. **Multi-Objective Learning for Computational Antibody Sequence Co-Optimization: Generalization Under Sparse Binary Supervision.** 2026.

A formal journal citation will be added after publication.

## References

- Makowski EK et al. (2022). Co-optimization of therapeutic antibody affinity and specificity using machine learning models that generalize to novel mutational space. *Nature Communications* 13, 3788.
- Lin Z et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379, 1123-1130.

## License

MIT License


