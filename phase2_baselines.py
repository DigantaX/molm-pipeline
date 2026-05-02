"""
MOLM Pipeline — Phase 2: LDA + NN Baselines
=============================================
Trains LDA and DeepProjectorDecider NN on all 3 feature types.

Requires: features.pkl (from Phase 1)
Saves: baselines.pkl (LDA + NN models + CV results)
Runtime: ~20 min
"""
# Auto-import: works when phases are run as local scripts
try:
    config  # Already loaded if Phase 0 ran in this kernel
except NameError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())
    from phase0_config import *

def train_lda_baselines(X, y_aff, y_spec, feature_name, n_folds=None):
    if n_folds is None: n_folds = config.CV_FOLDS
    print(f"\n  LDA on {feature_name} ({n_folds}-fold CV)...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Affinity
    lda_aff = LDA()
    cv_aff = cv(lda_aff, X, y_aff, cv=skf, scoring='accuracy')
    oof_aff = cross_val_predict(LDA(), X, y_aff, cv=skf)
    oof_aff_scores = cross_val_predict(LDA(), X, y_aff, cv=skf, method='decision_function')
    mcc_aff = safe_mcc(y_aff, oof_aff)
    auc_aff = safe_auc(y_aff, oof_aff_scores)
    lda_aff.fit(X, y_aff)
    
    # Specificity
    lda_spec = LDA()
    cv_spec = cv(lda_spec, X, y_spec, cv=skf, scoring='accuracy')
    oof_spec = cross_val_predict(LDA(), X, y_spec, cv=skf)
    oof_spec_scores = cross_val_predict(LDA(), X, y_spec, cv=skf, method='decision_function')
    mcc_spec = safe_mcc(y_spec, oof_spec)
    auc_spec = safe_auc(y_spec, oof_spec_scores)
    lda_spec.fit(X, y_spec)
    
    results = {
        'affinity_cv_mean': cv_aff['test_score'].mean(), 'affinity_cv_std': cv_aff['test_score'].std(),
        'affinity_mcc': mcc_aff, 'affinity_auc': auc_aff,
        'specificity_cv_mean': cv_spec['test_score'].mean(), 'specificity_cv_std': cv_spec['test_score'].std(),
        'specificity_mcc': mcc_spec, 'specificity_auc': auc_spec,
        'lda_aff': lda_aff, 'lda_spec': lda_spec
    }
    print(f"    Aff: {results['affinity_cv_mean']:.4f}±{results['affinity_cv_std']:.4f} MCC:{mcc_aff:.3f} AUC:{auc_aff:.4f}")
    print(f"    Spec: {results['specificity_cv_mean']:.4f}±{results['specificity_cv_std']:.4f} MCC:{mcc_spec:.3f} AUC:{auc_spec:.4f}")
    return results

def train_nn_baseline_cv(X, y, n_folds=5, epochs=50, batch_size=50, intermed_dim=20):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs, mccs, aucs = [], [], []
    
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        hard_reset_rng(SEED + fold, f"NN fold {fold}")
        model = train_deep_projector_decider(
            X[tr], y[tr], input_dim=X.shape[1], intermed_dim=intermed_dim,
            epochs=epochs, batch_size=batch_size, seed=SEED + fold)
        logits = predict_deep_projector_logits(model, X[te])
        preds = np.argmax(logits, 1)
        probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()[:, 1]  # P(class=1)
        accs.append((preds == y[te]).mean())
        mccs.append(safe_mcc(y[te], preds))
        aucs.append(safe_auc(y[te], probs))
        del model; gc.collect(); torch.cuda.empty_cache()
    
    hard_reset_rng(SEED + 999, "NN final")
    final = train_deep_projector_decider(
        X, y, input_dim=X.shape[1], intermed_dim=intermed_dim,
        epochs=epochs, batch_size=batch_size, seed=SEED + 999)
    
    return {'cv_mean': np.mean(accs), 'cv_std': np.std(accs), 'cv_scores': accs,
            'mcc_mean': np.nanmean(mccs), 'mcc_std': np.nanstd(mccs),
            'auc_mean': np.nanmean(aucs), 'auc_std': np.nanstd(aucs), 'auc_scores': aucs,
            'model': final}

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("PHASE 2: LDA + NN Baselines")
    print("=" * 60)
    
    features = load_phase_data("features")
    
    # LDA on all feature types
    lda_results = {}
    for feat_type in config.FEATURE_TYPES:
        if feat_type not in features['emi']: continue
        fl = FEAT_LABELS.get(feat_type, feat_type)
        X, y_aff, y_spec = features['emi'][feat_type]
        lda_results[f'LDA ({fl})'] = train_lda_baselines(X, y_aff, y_spec, fl)
    
    # NN on all feature types
    nn_results = {}
    for feat_type in config.FEATURE_TYPES:
        if feat_type not in features['emi']: continue
        fl = FEAT_LABELS.get(feat_type, feat_type)
        X, y_aff, y_spec = features['emi'][feat_type]
        print(f"\n  NN on {fl} ({X.shape[1]}D)...")
        aff_res = train_nn_baseline_cv(X, y_aff)
        spec_res = train_nn_baseline_cv(X, y_spec)
        nn_results[feat_type] = {
            'affinity': aff_res, 'specificity': spec_res,
            'affinity_cv_mean': aff_res['cv_mean'], 'affinity_cv_std': aff_res['cv_std'],
            'specificity_cv_mean': spec_res['cv_mean'], 'specificity_cv_std': spec_res['cv_std'],
        }
        print(f"    Aff: {aff_res['cv_mean']:.4f}±{aff_res['cv_std']:.4f} MCC:{aff_res['mcc_mean']:.3f} AUC:{aff_res['auc_mean']:.4f}")
        print(f"    Spec: {spec_res['cv_mean']:.4f}±{spec_res['cv_std']:.4f} MCC:{spec_res['mcc_mean']:.3f} AUC:{spec_res['auc_mean']:.4f}")
    
    save_phase_data({'lda': lda_results, 'nn': nn_results}, "baselines")
    
    # Print grid
    print(f"\n{'='*65}")
    print(f"{'Model':<25} {'Features':<15} {'Aff':>10} {'Spec':>10}")
    print(f"{'-'*65}")
    for fl_key, r in lda_results.items():
        print(f"{fl_key:<25} {'':15} {r['affinity_cv_mean']:.4f}±{r['affinity_cv_std']:.4f} {r['specificity_cv_mean']:.4f}±{r['specificity_cv_std']:.4f}")
    for ft, r in nn_results.items():
        fl = FEAT_LABELS.get(ft, ft)
        print(f"{'NN':<25} {fl:<15} {r['affinity_cv_mean']:.4f}±{r['affinity_cv_std']:.4f} {r['specificity_cv_mean']:.4f}±{r['specificity_cv_std']:.4f}")
    
    print(f"\n✓ Phase 2 complete in {time.time()-t0:.0f}s")
