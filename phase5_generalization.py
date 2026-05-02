"""
MOLM Pipeline — Phase 5: Generalization & Pareto Front
=======================================================
Trains final models on all features, evaluates ISO/IgG Spearman ρ
with bootstrap CIs, and runs Pareto front analysis.

Requires: features.pkl (Phase 1), baselines.pkl (Phase 2)
Saves: generalization.pkl (Spearman + Pareto results)
Runtime: ~30 min
"""
# Auto-import: works when phases are run as local scripts
try:
    config  # Already loaded if Phase 0 ran in this kernel
except NameError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())
    from phase0_config import *

def predict_model(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=device))

def tensor_to_numpy(x):
    return x.detach().cpu().numpy()

def predict_nn_positive_class_score(model, X):
    logits = predict_deep_projector_logits(model, X)
    return torch.softmax(torch.as_tensor(logits), dim=1).numpy()[:, 1]

def evaluate_generalization(model, features, model_type='molm', feature_type='fusion_esm2'):
    fl = FEAT_LABELS.get(feature_type, feature_type)
    print(f"\n  📈 Generalization ({model_type}, {fl})...")
    
    results = {}
    for ds_key, n_eval, ds_label in [('iso', 126, 'ISO'), ('igg', 42, 'IgG')]:
        if feature_type not in features[ds_key]:
            print(f"    ⚠ {ds_label}: feature '{feature_type}' not available, skipping")
            continue
        
        X = features[ds_key][feature_type][0][:n_eval]
        y_aff = features[ds_key][feature_type][1][:n_eval]
        y_spec = features[ds_key][feature_type][2][:n_eval]
        
        if model_type == 'molm':
            out = predict_model(model, X)
            pred_aff = tensor_to_numpy(out['logit_aff']).flatten()
            pred_spec = tensor_to_numpy(out['logit_spec']).flatten()
        elif model_type == 'molm_st':
            out_a = predict_model(model['affinity'], X)
            out_s = predict_model(model['specificity'], X)
            pred_aff = tensor_to_numpy(out_a['logit_aff']).flatten()
            pred_spec = tensor_to_numpy(out_s['logit_spec']).flatten()
        elif model_type == 'lda':
            X_oh = features[ds_key][feature_type][0][:n_eval]
            pred_aff = model['lda_aff'].transform(X_oh).flatten()
            pred_spec = model['lda_spec'].transform(X_oh).flatten()
        elif model_type == 'nn':
            pred_aff = predict_nn_positive_class_score(model['aff'], X)
            pred_spec = predict_nn_positive_class_score(model['spec'], X)
        
        rho_aff, p_aff = stats.spearmanr(pred_aff, y_aff)
        rho_spec, p_spec = stats.spearmanr(pred_spec, y_spec)
        
        results[f'{ds_key}_aff_rho'] = rho_aff
        results[f'{ds_key}_spec_rho'] = rho_spec
        results[f'{ds_key}_aff_pval'] = p_aff
        results[f'{ds_key}_spec_pval'] = p_spec
        
        # Bootstrap CIs
        results[f'{ds_key}_aff_ci'] = bootstrap_spearman_ci(pred_aff, y_aff, config.BOOTSTRAP_N)
        results[f'{ds_key}_spec_ci'] = bootstrap_spearman_ci(pred_spec, y_spec, config.BOOTSTRAP_N)
        
        print(f"    {ds_label}: Aff ρ={rho_aff:.4f} {results[f'{ds_key}_aff_ci']}, Spec ρ={rho_spec:.4f} {results[f'{ds_key}_spec_ci']}")
    
    return results

def evaluate_nn_generalization(nn_results, features, feature_type='onehot'):
    fl = FEAT_LABELS.get(feature_type, feature_type)
    print(f"\n  📈 NN Generalization ({fl})...")
    
    model_aff = nn_results[feature_type]['affinity']['model']
    model_spec = nn_results[feature_type]['specificity']['model']
    nn_model = {'aff': model_aff, 'spec': model_spec}
    return evaluate_generalization(nn_model, features, model_type='nn', feature_type=feature_type)

def pareto_analysis(features, molm_model, molm_st_models=None, lda_models=None,
                    nn_models=None, feature_type='fusion_esm2'):
    """
    Multi-score Pareto evaluation (Option 2):
    - logits: raw score head outputs
    - probs: sigmoid-calibrated probabilities
    - latent_pca: PCA-reduced 16D latent → 1D per task
    
    LEAKAGE-FREE PCA: PCA is fit on EMI training latents, then applied to
    ISO/IgG eval latents. Sign alignment uses EMI training labels.
    """
    from sklearn.decomposition import PCA
    
    # ---- Leakage-free PCA: fit on EMI training data ----
    X_train = features['emi'][feature_type][0]
    y_train_aff = features['emi'][feature_type][1]
    y_train_spec = features['emi'][feature_type][2]
    
    # MOLM training latents for PCA fitting
    pca_models = {}  # cache fitted PCA objects
    if molm_model is not None:
        train_out = predict_model(molm_model, X_train)
        train_aff_lat = tensor_to_numpy(train_out['z_aff'])
        train_spec_lat = tensor_to_numpy(train_out['z_spec'])
        pca_a_molm = PCA(n_components=1).fit(train_aff_lat)
        pca_s_molm = PCA(n_components=1).fit(train_spec_lat)
        # Determine sign from training labels (leakage-free)
        train_pca_a = pca_a_molm.transform(train_aff_lat).flatten()
        train_pca_s = pca_s_molm.transform(train_spec_lat).flatten()
        sign_a_molm = -1 if stats.spearmanr(train_pca_a, y_train_aff)[0] < 0 else 1
        sign_s_molm = -1 if stats.spearmanr(train_pca_s, y_train_spec)[0] > 0 else 1
        pca_models['molm'] = (pca_a_molm, pca_s_molm, sign_a_molm, sign_s_molm)
        print(f"  ✓ PCA fit on EMI training latents (MOLM, n={len(X_train)})")
    
    if molm_st_models:
        train_out_a = predict_model(molm_st_models['affinity'], X_train)
        train_out_s = predict_model(molm_st_models['specificity'], X_train)
        pca_a_st = PCA(n_components=1).fit(tensor_to_numpy(train_out_a['z_aff']))
        pca_s_st = PCA(n_components=1).fit(tensor_to_numpy(train_out_s['z_spec']))
        t_pca_a = pca_a_st.transform(tensor_to_numpy(train_out_a['z_aff'])).flatten()
        t_pca_s = pca_s_st.transform(tensor_to_numpy(train_out_s['z_spec'])).flatten()
        sign_a_st = -1 if stats.spearmanr(t_pca_a, y_train_aff)[0] < 0 else 1
        sign_s_st = -1 if stats.spearmanr(t_pca_s, y_train_spec)[0] > 0 else 1
        pca_models['molm_st'] = (pca_a_st, pca_s_st, sign_a_st, sign_s_st)
        print(f"  ✓ PCA fit on EMI training latents (MOLM-ST, n={len(X_train)})")
    
    all_results = {}
    for ds_name, ds_key, n_eval in [('ISO', 'iso', 126), ('IgG', 'igg', 42)]:
        if feature_type not in features[ds_key]:
            continue
        
        X = features[ds_key][feature_type][0][:n_eval]
        y_aff = features[ds_key][feature_type][1][:n_eval]
        y_spec = features[ds_key][feature_type][2][:n_eval]
        true_pareto = compute_pareto_front(y_aff, -y_spec)
        n_true = true_pareto.sum()
        
        print(f"\n  📊 Pareto ({ds_name}, n={n_eval}): {n_true} true Pareto-optimal")
        ds_results = {'true_pareto_count': int(n_true)}
        
        def eval_pareto(name, aff_s, spec_s, score_type='logits'):
            pp = compute_pareto_front(aff_s, spec_s)
            overlap = (pp & true_pareto).sum()
            n_p = pp.sum()
            prec = float(overlap / max(n_p, 1)); rec = float(overlap / max(n_true, 1))
            key = f"{name}_{score_type}" if score_type != 'logits' else name
            ds_results[key] = {'predicted': int(n_p), 'overlap': int(overlap),
                               'precision': prec, 'recall': rec, 'score_type': score_type}
            print(f"    {key:30s}: {n_p} pred, {overlap} overlap (P={prec:.2f}, R={rec:.2f})")
            return aff_s, spec_s
        
        # ---- MOLM: all 3 score types ----
        out = predict_model(molm_model, X)
        
        for score_type in config.PARETO_SCORE_TYPES:
            if score_type == 'logits':
                aff_s = tensor_to_numpy(out['logit_aff']).flatten()
                spec_s = tensor_to_numpy(out['logit_spec']).flatten()
            elif score_type == 'probs':
                aff_s = tensor_to_numpy(out['aff_prob']).flatten()
                spec_s = tensor_to_numpy(out['spec_prob']).flatten()
            elif score_type == 'latent_pca':
                pca_a, pca_s, sign_a, sign_s = pca_models['molm']
                aff_s = sign_a * pca_a.transform(tensor_to_numpy(out['z_aff'])).flatten()
                spec_s = sign_s * pca_s.transform(tensor_to_numpy(out['z_spec'])).flatten()
            
            eval_pareto('MOLM', aff_s, spec_s, score_type)
        
        # ---- MOLM-ST: all 3 score types ----
        if molm_st_models:
            out_a = predict_model(molm_st_models['affinity'], X)
            out_s = predict_model(molm_st_models['specificity'], X)
            
            for score_type in config.PARETO_SCORE_TYPES:
                if score_type == 'logits':
                    aff_s = tensor_to_numpy(out_a['logit_aff']).flatten()
                    spec_s = tensor_to_numpy(out_s['logit_spec']).flatten()
                elif score_type == 'probs':
                    aff_s = tensor_to_numpy(out_a['aff_prob']).flatten()
                    spec_s = tensor_to_numpy(out_s['spec_prob']).flatten()
                elif score_type == 'latent_pca':
                    pca_a, pca_s, sign_a, sign_s = pca_models['molm_st']
                    aff_s = sign_a * pca_a.transform(tensor_to_numpy(out_a['z_aff'])).flatten()
                    spec_s = sign_s * pca_s.transform(tensor_to_numpy(out_s['z_spec'])).flatten()
                
                eval_pareto('MOLM-ST', aff_s, spec_s, score_type)
        
        # ---- LDA (logits only — LDA has no latent) ----
        if lda_models:
            X_oh = features[ds_key]['onehot'][0][:n_eval]
            eval_pareto('LDA', lda_models['lda_aff'].transform(X_oh).flatten(),
                        lda_models['lda_spec'].transform(X_oh).flatten())
        
        # ---- NN (logits only) ----
        if nn_models:
            for nn_name, nn_pair in nn_models.items():
                ft = nn_pair.get('feature_type', 'onehot')
                if ft in features[ds_key]:
                    X_nn = features[ds_key][ft][0][:n_eval]
                    eval_pareto(nn_name, predict_nn_positive_class_score(nn_pair['aff'], X_nn),
                                predict_nn_positive_class_score(nn_pair['spec'], X_nn))
        
        all_results[ds_name] = ds_results
    
    return all_results


def plot_pareto_diagnostics(features, molm_model, molm_st_models=None, feature_type='fusion_esm2'):
    """
    Diagnostic plots to understand Pareto geometry inside the model:
    1. Score scatter (aff vs spec) colored by ground truth quadrants
    2. Latent space PCA (2D) colored by ground truth
    3. Score distributions (histograms)
    4. Pareto fronts overlay (all score types)
    
    LEAKAGE-FREE: PCA is fit on EMI training latents, applied to eval sets.
    """
    from sklearn.decomposition import PCA
    
    # Fit PCA on EMI training latents (leakage-free)
    X_train = features['emi'][feature_type][0]
    train_out = predict_model(molm_model, X_train)
    train_combined_lat = np.concatenate([
        tensor_to_numpy(train_out['z_aff']),
        tensor_to_numpy(train_out['z_spec'])
    ], axis=1)
    pca_2d = PCA(n_components=2).fit(train_combined_lat)
    print(f"  ✓ PCA (2D viz) fit on EMI training latents (n={len(X_train)})")
    
    for ds_name, ds_key, n_eval in [('ISO', 'iso', 126), ('IgG', 'igg', 42)]:
        if feature_type not in features[ds_key]:
            continue
        
        X = features[ds_key][feature_type][0][:n_eval]
        y_aff = features[ds_key][feature_type][1][:n_eval]
        y_spec = features[ds_key][feature_type][2][:n_eval]
        true_pareto = compute_pareto_front(y_aff, -y_spec)
        
        out = predict_model(molm_model, X)
        aff_logit = tensor_to_numpy(out['logit_aff']).flatten()
        spec_logit = tensor_to_numpy(out['logit_spec']).flatten()
        aff_prob = tensor_to_numpy(out['aff_prob']).flatten()
        spec_prob = tensor_to_numpy(out['spec_prob']).flatten()
        aff_lat = tensor_to_numpy(out['z_aff'])
        spec_lat = tensor_to_numpy(out['z_spec'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Pareto Diagnostics — MOLM on {ds_name} (n={n_eval})", fontsize=14, fontweight='bold')
        
        # --- Row 1: Score spaces ---
        
        # (1) Logit scatter
        ax = axes[0, 0]
        sc = ax.scatter(aff_logit, spec_logit, c=y_aff, cmap='coolwarm', alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
        pred_pareto = compute_pareto_front(aff_logit, spec_logit)
        ax.scatter(aff_logit[pred_pareto], spec_logit[pred_pareto], s=100, facecolors='none', edgecolors='lime', linewidth=2, label=f'Pred Pareto ({pred_pareto.sum()})')
        ax.scatter(aff_logit[true_pareto], spec_logit[true_pareto], s=100, facecolors='none', edgecolors='gold', linewidth=2, marker='D', label=f'True Pareto ({true_pareto.sum()})')
        ax.set_xlabel("Aff logit"); ax.set_ylabel("Spec logit"); ax.set_title("(A) Logit Space"); ax.legend(fontsize=7)
        plt.colorbar(sc, ax=ax, label='True Aff')
        
        # (2) Probability scatter
        ax = axes[0, 1]
        sc = ax.scatter(aff_prob, spec_prob, c=y_aff, cmap='coolwarm', alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
        pred_pareto_p = compute_pareto_front(aff_prob, spec_prob)
        ax.scatter(aff_prob[pred_pareto_p], spec_prob[pred_pareto_p], s=100, facecolors='none', edgecolors='lime', linewidth=2, label=f'Pred Pareto ({pred_pareto_p.sum()})')
        ax.scatter(aff_prob[true_pareto], spec_prob[true_pareto], s=100, facecolors='none', edgecolors='gold', linewidth=2, marker='D', label=f'True Pareto ({true_pareto.sum()})')
        ax.set_xlabel("Aff prob"); ax.set_ylabel("Spec prob"); ax.set_title("(B) Probability Space"); ax.legend(fontsize=7)
        plt.colorbar(sc, ax=ax, label='True Aff')
        
        # (3) Latent PCA scatter — LEAKAGE-FREE: use training-fit PCA
        ax = axes[0, 2]
        combined_lat = np.concatenate([aff_lat, spec_lat], axis=1)
        lat_2d = pca_2d.transform(combined_lat)  # transform, NOT fit_transform
        sc = ax.scatter(lat_2d[:, 0], lat_2d[:, 1], c=y_aff, cmap='coolwarm', alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
        ax.scatter(lat_2d[true_pareto, 0], lat_2d[true_pareto, 1], s=100, facecolors='none', edgecolors='gold', linewidth=2, marker='D', label=f'True Pareto ({true_pareto.sum()})')
        ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})"); ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
        ax.set_title("(C) Latent Space (PCA, train-fit)"); ax.legend(fontsize=7)
        plt.colorbar(sc, ax=ax, label='True Aff')
        
        # --- Row 2: Distributions & Correlations ---
        
        # (4) Score distributions
        ax = axes[1, 0]
        ax.hist(aff_logit, bins=30, alpha=0.5, color='#3498db', label='Aff logit', density=True)
        ax.hist(spec_logit, bins=30, alpha=0.5, color='#e74c3c', label='Spec logit', density=True)
        ax.set_title("(D) Score Distributions"); ax.legend(fontsize=8)
        ax.set_xlabel("Logit value"); ax.set_ylabel("Density")
        
        # (5) Score correlation: predicted vs ground truth
        ax = axes[1, 1]
        rho_a, _ = stats.spearmanr(aff_logit, y_aff)
        rho_s, _ = stats.spearmanr(spec_logit, y_spec)
        ax.scatter(y_aff, aff_logit, alpha=0.5, s=20, label=f'Aff (ρ={rho_a:.3f})', color='#3498db')
        ax.scatter(y_spec, spec_logit, alpha=0.5, s=20, label=f'Spec (ρ={rho_s:.3f})', color='#e74c3c')
        ax.set_xlabel("Ground truth"); ax.set_ylabel("Predicted logit"); ax.set_title("(E) Pred vs Truth")
        ax.legend(fontsize=8)
        
        # (6) Cross-task correlation
        ax = axes[1, 2]
        corr_pred = np.corrcoef(aff_logit, spec_logit)[0, 1]
        corr_true = np.corrcoef(y_aff, y_spec)[0, 1]
        ax.scatter(aff_logit, spec_logit, c='gray', alpha=0.3, s=20)
        ax.scatter(aff_logit[true_pareto], spec_logit[true_pareto], c='gold', s=80, marker='D', zorder=5, label='True Pareto')
        ax.set_xlabel("Aff logit"); ax.set_ylabel("Spec logit")
        ax.set_title(f"(F) Cross-task Corr\nPred r={corr_pred:.3f}, True r={corr_true:.3f}")
        ax.legend(fontsize=8)
        
        # Add text box with key metrics
        info = f"Score corr: {corr_pred:.3f}\nTrue corr: {corr_true:.3f}\nAff ρ: {rho_a:.3f}\nSpec ρ: {rho_s:.3f}"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        save_path = phase_output_path(f"pareto_diagnostics_{ds_name.lower()}.png", "phase5")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {save_path}")
    
    # === MOLM vs MOLM-ST comparison plot ===
    if molm_st_models is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("MOLM vs MOLM-ST Score Geometry", fontsize=14, fontweight='bold')
        
        for ds_name, ds_key, n_eval, ax in [('ISO', 'iso', 126, axes[0]), ('IgG', 'igg', 42, axes[1])]:
            if feature_type not in features[ds_key]: continue
            X = features[ds_key][feature_type][0][:n_eval]
            y_aff = features[ds_key][feature_type][1][:n_eval]
            y_spec = features[ds_key][feature_type][2][:n_eval]
            true_p = compute_pareto_front(y_aff, -y_spec)
            
            out_m = predict_model(molm_model, X)
            out_a = predict_model(molm_st_models['affinity'], X)
            out_s = predict_model(molm_st_models['specificity'], X)
            
            ma = tensor_to_numpy(out_m['aff_prob']).flatten()
            ms = tensor_to_numpy(out_m['spec_prob']).flatten()
            sa = tensor_to_numpy(out_a['aff_prob']).flatten()
            ss = tensor_to_numpy(out_s['spec_prob']).flatten()
            
            ax.scatter(ma, ms, alpha=0.5, s=25, c='#3498db', label=f'MOLM (r={np.corrcoef(ma,ms)[0,1]:.2f})')
            ax.scatter(sa, ss, alpha=0.5, s=25, c='#e74c3c', marker='^', label=f'MOLM-ST (r={np.corrcoef(sa,ss)[0,1]:.2f})')
            ax.scatter(ma[true_p], ms[true_p], s=100, facecolors='none', edgecolors='gold', linewidth=2, marker='D', label='True Pareto', zorder=5)
            ax.set_xlabel("Aff prob"); ax.set_ylabel("Spec prob"); ax.set_title(f"{ds_name} (n={n_eval})")
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        save_path = phase_output_path("pareto_molm_vs_st.png", "phase5")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {save_path}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("PHASE 5: Generalization & Pareto Front")
    print("=" * 60)
    
    features = load_phase_data("features")
    baselines = load_phase_data("baselines")
    lda_results = baselines['lda']
    nn_results = baselines['nn']
    
    primary_feat, primary_label = get_primary_feat(features)
    aff_pw = features['emi']['aff_pos_weight']
    spec_pw = features['emi']['spec_pos_weight']
    
    gen_results = {}
    final_models = {}
    final_st_models = {}
    
    # Train final models on ALL feature types
    for feat_type in config.FEATURE_TYPES:
        if feat_type not in features['emi']: continue
        fl = FEAT_LABELS.get(feat_type, feat_type)
        X, y_aff, y_spec = features['emi'][feat_type]
        
        # ISO/IgG gen data
        X_iso = features['iso'].get(feat_type, (None,))[0] if feat_type in features['iso'] else None
        y_iso_aff = features['iso'][feat_type][1] if X_iso is not None else None
        y_iso_spec = features['iso'][feat_type][2] if X_iso is not None else None
        X_igg = features['igg'].get(feat_type, (None,))[0] if feat_type in features['igg'] else None
        y_igg_aff = features['igg'][feat_type][1] if X_igg is not None else None
        y_igg_spec = features['igg'][feat_type][2] if X_igg is not None else None
        
        # MOLM
        print(f"\n🏋️  Final MOLM ({fl})...")
        hard_reset_rng(SEED, f"Final MOLM {fl}")
        model = DiagnosticMOLM(input_dim=X.shape[1], latent_dim=config.LATENT_DIM,
                               shared_dims=config.SHARED_DIMS, tower_dims=config.TOWER_DIMS,
                               dropout_rate=config.DROPOUT_RATE, grl_lambda=config.GRL_LAMBDA)
        trainer = DiagnosticTrainer(model, aff_pos_weight=aff_pw, spec_pos_weight=spec_pw, learning_rate=config.LEARNING_RATE)
        verbose = 1 if feat_type == primary_feat else 0
        trainer.fit(X, y_aff, y_spec, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                    X_iso=X_iso, y_iso_aff=y_iso_aff, y_iso_spec=y_iso_spec,
                    X_igg=X_igg, y_igg_aff=y_igg_aff, y_igg_spec=y_igg_spec, verbose=verbose)
        final_models[feat_type] = model
        gen_results[f'MOLM ({fl})'] = evaluate_generalization(model, features, 'molm', feat_type)
        
        # MOLM-ST
        if config.RUN_MOLM_ST:
            print(f"  Final MOLM-ST ({fl})...")
            st_a = train_molm_st(X, y_aff, y_spec, 'affinity', config, aff_pw, spec_pw, 500)
            st_s = train_molm_st(X, y_aff, y_spec, 'specificity', config, aff_pw, spec_pw, 600)
            final_st_models[feat_type] = {'affinity': st_a, 'specificity': st_s}
            gen_results[f'MOLM-ST ({fl})'] = evaluate_generalization(
                final_st_models[feat_type], features, 'molm_st', feat_type)
    
    # LDA generalization — all available
    for lda_key, feat_type in [('LDA (OneHot)', 'onehot'), ('LDA (ESM2)', 'esm2'), ('LDA (Fusion-ESM2)', 'fusion_esm2')]:
        if lda_key in lda_results and feat_type in features['iso']:
            gen_results[lda_key] = evaluate_generalization(lda_results[lda_key], features, 'lda', feat_type)
    
    # NN generalization — all features
    for feat_type in config.FEATURE_TYPES:
        if feat_type in nn_results and feat_type in features['iso']:
            fl = FEAT_LABELS.get(feat_type, feat_type)
            gen_results[f'NN ({fl})'] = evaluate_nn_generalization(nn_results, features, feat_type)
    
    # Pareto front analysis
    print("\n" + "=" * 60)
    print("PARETO FRONT ANALYSIS")
    print(f"  Score types: {config.PARETO_SCORE_TYPES}")
    print(f"  Pareto loss: {'ON (weight={})'.format(config.PARETO_LOSS_WEIGHT) if config.PARETO_LOSS else 'OFF'}")
    print("=" * 60)
    
    nn_pareto = {}
    for ft_key, label in [('onehot', 'NN (OneHot)'), (primary_feat, f'NN ({primary_label})')]:
        if ft_key in nn_results:
            nn_pareto[label] = {
                'aff': nn_results[ft_key]['affinity']['model'],
                'spec': nn_results[ft_key]['specificity']['model'],
                'feature_type': ft_key}
    
    pareto_results = pareto_analysis(
        features, final_models.get(primary_feat),
        molm_st_models=final_st_models.get(primary_feat),
        lda_models=lda_results.get('LDA (OneHot)'),
        nn_models=nn_pareto, feature_type=primary_feat)
    
    # Diagnostic plots
    print("\n" + "=" * 60)
    print("PARETO DIAGNOSTIC PLOTS")
    print("=" * 60)
    plot_pareto_diagnostics(
        features, final_models.get(primary_feat),
        molm_st_models=final_st_models.get(primary_feat),
        feature_type=primary_feat)
    
    # Save everything
    save_phase_data({
        'generalization': gen_results,
        'pareto': pareto_results,
    }, "generalization")
    
    # CSV export
    gen_rows = []
    for model_name, res in gen_results.items():
        row = {'model': model_name}
        row.update(res)
        gen_rows.append(row)
    pd.DataFrame(gen_rows).to_csv(phase_output_path("generalization_results.csv", "phase5"), index=False)
    
    # Print full grid
    print(f"\n{'='*85}")
    print("GENERALIZATION GRID (Spearman ρ)")
    print(f"{'='*85}")
    print(f"{'Model':<30} {'ISO Aff':>10} {'ISO Spec':>10} {'IgG Aff':>10} {'IgG Spec':>10}")
    print(f"{'-'*85}")
    for name, r in gen_results.items():
        print(f"{name:<30} {r.get('iso_aff_rho',0):>10.4f} {r.get('iso_spec_rho',0):>10.4f} "
              f"{r.get('igg_aff_rho',0):>10.4f} {r.get('igg_spec_rho',0):>10.4f}")
    
    print(f"\n✓ Phase 5 complete in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
