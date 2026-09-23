# Extended analysis notebooks

These notebooks contain the cleaned reproducibility workflows supporting the current MOLM manuscript. Process-specific drafting annotations have been removed; the scientific settings, model definitions, seeds, statistics, and output logic are retained.

## Recommended order

### 1. `01_unified_experiment_suite.ipynb`
Runs the broad extended analysis suite:

- Standard-MOLM mutation grid across five representations and five seeds;
- paired mutation statistics;
- ISO / IgG external Spearman analyses;
- fixed-budget Pareto evaluation at `K={5,10,15,20,25}`;
- Standard-MOLM latent-PCA robustness;
- A-H Routed-MOLM component/capacity controls;
- ranking/gap loss ablation;
- Hamming-distance-stratified within-scaffold generalization;
- unified result and reproducibility bundle.

The notebook records an analysis lock before the extended runs. ISO/IgG were observed in earlier development, so later analyses are retrospective robustness/controlled analyses rather than newly blind validation.

### 2. `02_molm_st_mutation_holdout.ipynb`
Runs the original plain MOLM-ST implementation under the current grouped mutation-holdout protocol. It provides the dedicated same-loss independent control used for Standard-MOLM versus MOLM-ST mutation comparisons.

### 3. `03_molm_st_fixed_budget_pareto.ipynb`
Runs the original MOLM-ST control for external prediction and fixed-budget multi-objective prioritization. It can optionally merge with compatible result bundles from the unified experiment suite.

### 4. `04_component_architecture_ablation.ipynb`
Runs two controlled experiment suites:

- an eight-arm architecture/component comparison;
- a focal/ranking/gap loss-factorization comparison.

All component arms are evaluated on the same fixed-budget Pareto endpoint rather than relying only on marginal prediction metrics.

### 5. `05_publication_figure_generator.ipynb`
Loads finalized analysis bundles, validates key configuration metadata when available, and regenerates the manuscript/supplementary figures.

## Reproducibility conventions

- Core repository commit pinned by the notebooks: `c5923984f0d5176977edb4a4ffd8fc5f98536043`.
- Optimization seeds: `42, 123, 456, 789, 2024`.
- Main fixed budgets: `5, 10, 15, 20, 25`.
- Neural Pareto selection is performed independently per seed before across-seed aggregation.
- ISO measured reference Pareto front: 15 of 126 variants.
- Secondary IgG-all96 measured reference Pareto front: 4 of 96 variants.

Notebook outputs are intentionally cleared in GitHub so the repository remains compact and the displayed results cannot be confused with stale interactive state.
