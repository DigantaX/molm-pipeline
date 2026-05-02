"""
MOLM Pipeline — Phase 1: Data Loading & Feature Preparation
=============================================================
Loads EMI/ISO/IgG data, computes ESM-2 embeddings (cached),
builds OneHot + ESM2 + Fusion-ESM2 features for all datasets.

Saves: features.pkl (~50MB)
Runtime: ~5 min (first run with ESM-2), <30s (cached)
"""
# Auto-import: works when phases are run as local scripts
try:
    config  # Already loaded if Phase 0 ran in this kernel
except NameError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())
    from phase0_config import *

def _compute_esm2_embeddings(sequences, esm_model, alphabet, batch_converter, device, embed_layer=6, batch_size=64):
    import torch
    all_emb = []
    data_tuples = list(zip([f"s{i}" for i in range(len(sequences))], sequences))
    for i in range(0, len(data_tuples), batch_size):
        batch = data_tuples[i:i+batch_size]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[embed_layer], return_contacts=False)
        token_reps = results["representations"][embed_layer]
        for j, (_, seq) in enumerate(batch):
            all_emb.append(token_reps[j, 1:len(seq)+1, :].mean(dim=0).cpu().numpy())
        if (i // batch_size + 1) % 10 == 0:
            print(f"      {min(i+batch_size, len(data_tuples))}/{len(data_tuples)} sequences...")
    return np.stack(all_emb, axis=0)

def _compute_and_cache_esm2(data, esm2_files, embed_dim=320):
    import torch, esm
    print("    Loading ESM-2 model (esm2_t6_8M_UR50D)...")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = esm_model.to(device)
    print(f"    ESM-2 on: {device}")
    
    for ds_key, (binding_key, esm2_key) in {'emi': ('emi_binding', 'emi_esm2'),
                                              'iso': ('iso_binding', 'iso_esm2'),
                                              'igg': ('igg_binding', 'igg_esm2')}.items():
        seqs = list(data[binding_key].index)
        print(f"    Computing {ds_key.upper()} ({len(seqs)} seqs)...")
        emb = _compute_esm2_embeddings(seqs, esm_model, alphabet, batch_converter, device)
        emb_df = pd.DataFrame(emb, index=data[binding_key].index, columns=[f"esm2_{i}" for i in range(embed_dim)])
        emb_df.to_csv(esm2_files[esm2_key])
        data[esm2_key] = emb_df
        print(f"      ✓ Saved: {esm2_files[esm2_key]}")
    
    del esm_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return True

def load_all_data():
    print(f"\n📂 Loading data from: {config.DATA_PATH}")
    data = {}
    data['emi_binding'] = pd.read_csv(os.path.join(config.DATA_PATH, "emi_binding.csv"), header=0, index_col=0)
    data['iso_binding'] = pd.read_csv(os.path.join(config.DATA_PATH, "iso_binding.csv"), header=0, index_col=0)
    data['igg_binding'] = pd.read_csv(os.path.join(config.DATA_PATH, "igg_binding.csv"), header=0, index_col=0)
    print(f"  EMI: {len(data['emi_binding'])}, ISO: {len(data['iso_binding'])}, IgG: {len(data['igg_binding'])}")
    
    # Column resolution
    for prefix, df in [('emi', data['emi_binding']), ('iso', data['iso_binding']), ('igg', data['igg_binding'])]:
        for col_candidates, key in [(['ANT Binding', 'ANT'], f'{prefix}_aff_col'),
                                     (['OVA Binding', 'OVA', 'PSY'], f'{prefix}_spec_col')]:
            for c in col_candidates:
                if c in df.columns:
                    data[key] = c; break
    
    # Continuous values for ISO/IgG
    for prefix in ['iso', 'igg']:
        df = data[f'{prefix}_binding']
        data[f'{prefix}_aff_cont'] = df[data[f'{prefix}_aff_col']].values
        data[f'{prefix}_spec_cont'] = df[data[f'{prefix}_spec_col']].values
    
    # UniRep
    data['emi_reps'] = pd.read_csv(os.path.join(config.DATA_PATH, "emi_reps.csv"), header=0, index_col=0)
    data['iso_reps'] = pd.read_csv(os.path.join(config.DATA_PATH, "iso_reps.csv"), header=0, index_col=0)
    data['igg_reps'] = pd.read_csv(os.path.join(config.DATA_PATH, "igg_reps.csv"), header=0, index_col=0)
    
    # ESM-2
    if config.USE_ESM2:
        default_esm2_files = {k: os.path.join(config.ESM2_DIR, f"{k}.csv") for k in ['emi_esm2', 'iso_esm2', 'igg_esm2']}
        legacy_esm2_dir = os.path.join(config.OUTPUT_DIR, "esm2")
        legacy_esm2_files = {k: os.path.join(legacy_esm2_dir, f"{k}.csv") for k in default_esm2_files}
        esm2_files = default_esm2_files
        cache_source = config.ESM2_DIR
        if not all(os.path.exists(f) for f in default_esm2_files.values()) and all(os.path.exists(f) for f in legacy_esm2_files.values()):
            esm2_files = legacy_esm2_files
            cache_source = legacy_esm2_dir
        if all(os.path.exists(f) for f in esm2_files.values()):
            print(f"  Loading cached ESM-2 from: {cache_source}")
            for k in esm2_files:
                data[k] = pd.read_csv(esm2_files[k], header=0, index_col=0)
            data['has_esm2'] = True
            print(f"  ✓ ESM-2 loaded ({data['emi_esm2'].shape[1]}D)")
        else:
            print("  Computing ESM-2 embeddings...")
            try:
                data['has_esm2'] = _compute_and_cache_esm2(data, default_esm2_files, config.ESM2_DIM)
            except Exception as e:
                print(f"  ✗ ESM-2 failed: {e}. Falling back to UniRep.")
                data['has_esm2'] = False
    else:
        data['has_esm2'] = False
    
    # pI / residue dict (optional)
    try:
        data['residue_info'] = pd.read_csv(os.path.join(config.DATA_PATH, "residue_dict.csv"), header=0, index_col=0)
    except: pass
    
    return data

def prepare_features(data):
    print("\n🔧 Preparing features...")
    features = {'emi': {}, 'iso': {}, 'igg': {}}
    
    for ds_key, binding_key, reps_key in [('emi', 'emi_binding', 'emi_reps'),
                                            ('iso', 'iso_binding', 'iso_reps'),
                                            ('igg', 'igg_binding', 'igg_reps')]:
        df = data[binding_key]
        
        # Labels
        if ds_key == 'emi':
            y_aff = (df[data['emi_aff_col']].values > 0).astype(np.int64)
            y_spec = (df[data['emi_spec_col']].values > 0).astype(np.int64)
        else:
            y_aff = data[f'{ds_key}_aff_cont']
            y_spec = data[f'{ds_key}_spec_cont']
        
        # OneHot
        onehot = generate_onehot(df).values.astype(np.float32)
        features[ds_key]['onehot'] = (onehot, y_aff, y_spec)
        
        # UniRep
        unirep = data[reps_key].loc[df.index].values.astype(np.float32)
        features[ds_key]['unirep'] = (unirep, y_aff, y_spec)
        
        # Fusion (UniRep)
        features[ds_key]['fusion'] = (np.concatenate([onehot, unirep], 1).astype(np.float32), y_aff, y_spec)
        
        # ESM-2
        if data.get('has_esm2'):
            esm2 = data[f'{ds_key}_esm2'].loc[df.index].values.astype(np.float32)
            features[ds_key]['esm2'] = (esm2, y_aff, y_spec)
            features[ds_key]['fusion_esm2'] = (np.concatenate([onehot, esm2], 1).astype(np.float32), y_aff, y_spec)
        
        features[ds_key]['binding'] = df
        print(f"  {ds_key.upper()}: {', '.join(f'{k}={v[0].shape}' for k, v in features[ds_key].items() if isinstance(v, tuple))}")
    
    # Class weights
    y_aff_emi = features['emi']['onehot'][1]
    features['emi']['aff_pos_weight'] = (1 - y_aff_emi.mean()) / max(y_aff_emi.mean(), 1e-6)
    features['emi']['spec_pos_weight'] = 1.0
    
    return features

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("PHASE 1: Data Loading & Feature Preparation")
    print("=" * 60)
    
    data = load_all_data()
    features = prepare_features(data)
    
    save_phase_data(features, "features")
    
    print(f"\n✓ Phase 1 complete in {time.time()-t0:.0f}s")
