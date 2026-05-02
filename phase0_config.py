"""
MOLM Pipeline — Phase 0: Config & Shared Utilities
====================================================
Contains all imports, Config, model classes, loss functions, metrics,
and utility functions shared across all phases.

Usage: Every other phase does `from phase0_config import *`
"""

# ============================================================================
# DETERMINISTIC SETUP — MUST BE AT VERY TOP
# ============================================================================
import os
import random

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ============================================================================
# IMPORTS
# ============================================================================
import gc
gc.collect()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate as cv
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import stats
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import warnings
import json
import sys
import time
import hashlib
import io
import pickle

# ============================================================================
# DETERMINISTIC SEEDING
# ============================================================================
# Allow seed to be set by the multi-seed orchestrator via env var.
# Falls back to 42 so single-phase runs are unchanged.
SEED = int(os.environ.get("MOLM_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()
torch.use_deterministic_algorithms(True)
gc.collect()

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# DEVICE SETUP
# ============================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    DATA_PATH = os.environ.get("MOLM_DATA_PATH", os.path.join(PROJECT_DIR, "data"))
    OUTPUT_DIR = os.environ.get("MOLM_OUTPUT_DIR", os.path.join(PROJECT_DIR, "outputs"))
    
    RUN_LDA_BASELINES = True
    RUN_NN_BASELINES = True
    RUN_MOLM = True
    RUN_MOLM_ST = True
    
    DEBUG_MODE = True
    SAVE_FOLD_MANIFEST = True
    SAVE_EPOCH_CSV = True
    SAVE_PAPER_FIG = True
    PAPER_FIG_NAME = "paper_diag.png"
    GEN_EVAL_EVERY = 5
    
    SEQ_LENGTH = 115
    MUTATION_SITES = [32, 49, 54, 55, 56, 98, 100, 103]
    MUTATION_SITES_KABAT = [33, 50, 54, 55, 56, 95, 97, 102]
    WILDTYPE_RESIDUES = ['Y', 'R', 'R', 'R', 'G', 'A', 'W', 'Y']
    TOP_RESIDUES = ['V', 'E', 'G', 'G', 'D', 'S', 'L', 'D']
    SAMPLED_RESIDUES = [
        'YFVASD', 'RKGATEM', 'RKGATE', 'RKGATE',
        'GANSTD', 'AFVYSD', 'WLVGAS', 'YFVASDT',
    ]
    
    SHARED_DIMS = [256, 128]
    TOWER_DIMS = [64, 32]
    LATENT_DIM = 16
    DROPOUT_RATE = 0.2
    
    EPOCHS = 25
    BATCH_SIZE = 64
    CV_FOLDS = 5
    MC_SAMPLES = 50
    LEARNING_RATE = 5e-5
    
    ADVERSARIAL_WEIGHT = 0.0
    ORTHO_WEIGHT = 0.0
    ADV_WARMUP_EPOCHS = 10
    
    RANKING_WEIGHT_AFF = 0.3
    RANKING_WEIGHT_SPEC = 0.6
    GAP_WEIGHT_AFF = 0.2
    GAP_WEIGHT_SPEC = 0.6
    FOCAL_GAMMA = 2.0
    GRL_LAMBDA = 1.0
    RANKING_MARGIN = 0.3
    GAP_MARGIN = 0.2
    
    FEATURE_TYPES = ['onehot', 'esm2', 'fusion_esm2']
    GRID_FEATURE_TYPES = ['onehot', 'esm2', 'fusion_esm2']
    
    ESM2_DIR = os.environ.get("MOLM_ESM2_DIR", os.path.join(OUTPUT_DIR, "phase1", "esm2"))
    ESM2_DIM = 320
    USE_ESM2 = True
    
    COMPUTE_AUC = True
    BOOTSTRAP_N = 1000
    
    # === PARETO EXPERIMENT FLAGS ===
    PARETO_LOSS = os.environ.get("MOLM_PARETO_LOSS", "0") == "1"         # Option 3: Add Pareto-diversity loss during training
    PARETO_LOSS_WEIGHT = 0.1     # Moderate weight — acts as regularizer
    PARETO_LOSS_WARMUP = 5       # Don't apply until epoch 5 (let classification stabilize)
    PARETO_SCORE_TYPES = ['logits', 'probs', 'latent_pca']  # Option 2: Evaluate all 3

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.ESM2_DIR, exist_ok=True)

# Feature label mapping (used across all phases)
FEAT_LABELS = {
    'onehot': 'OneHot', 'esm2': 'ESM2',
    'fusion_esm2': 'Fusion-ESM2', 'fusion': 'Fusion-UniRep'
}

ARTIFACT_PHASES = {
    'features': 'phase1',
    'baselines': 'phase2',
    'molm_cv': 'phase3',
    'holdout': 'phase4',
    'generalization': 'phase5',
}

def get_phase_output_dir(phase):
    path = os.path.join(config.OUTPUT_DIR, phase)
    os.makedirs(path, exist_ok=True)
    return path

def phase_output_path(filename, phase):
    return os.path.join(get_phase_output_dir(phase), filename)

def get_artifact_phase(name, phase=None):
    return phase if phase is not None else ARTIFACT_PHASES.get(name)

def get_artifact_candidates(name, suffix, phase=None):
    candidates = []
    artifact_phase = get_artifact_phase(name, phase)
    if artifact_phase:
        candidates.append(phase_output_path(f"{name}{suffix}", artifact_phase))
    candidates.append(os.path.join(config.OUTPUT_DIR, f"{name}{suffix}"))

    # Multi-seed support: fall back to a shared output dir for deterministic
    # artifacts (e.g. Phase 1 features.pkl) that are reused across all seeds.
    # Set MOLM_SHARED_OUTPUT_DIR to the original outputs/ when running per-seed.
    shared_dir = os.environ.get("MOLM_SHARED_OUTPUT_DIR")
    if shared_dir and os.path.abspath(shared_dir) != os.path.abspath(config.OUTPUT_DIR):
        if artifact_phase:
            candidates.append(os.path.join(shared_dir, artifact_phase, f"{name}{suffix}"))
        candidates.append(os.path.join(shared_dir, f"{name}{suffix}"))

    return candidates

def get_primary_feat(features):
    """Determine primary feature type."""
    if 'fusion_esm2' in features.get('emi', {}):
        return 'fusion_esm2', 'Fusion-ESM2'
    return 'fusion', 'Fusion-UniRep'

# ============================================================================
# LOGGING
# ============================================================================
class DebugLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.logs = []
        with open(log_path, 'w') as f:
            f.write(f"# Debug Log - {datetime.now().isoformat()}\n{'='*70}\n\n")
    
    def log(self, msg, to_console=True):
        ts = datetime.now().strftime("%H:%M:%S")
        fmt = f"[{ts}] {msg}"
        self.logs.append(fmt)
        with open(self.log_path, 'a') as f:
            f.write(fmt + "\n")
        if to_console:
            print(fmt)
    
    def log_json(self, data, label=""):
        self.log(f"{label}:\n{json.dumps(data, indent=2, default=str)}", to_console=config.DEBUG_MODE)
    
    def log_separator(self, title=""):
        sep = "-" * 50
        self.log(f"\n{sep}\n{title}\n{sep}" if title else sep)

logger = DebugLogger(os.path.join(config.OUTPUT_DIR, "debug_log.txt"))

# ============================================================================
# DETERMINISTIC UTILITIES
# ============================================================================
def hard_reset_rng(seed, context=""):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
    gc.collect()
    if config.DEBUG_MODE:
        logger.log(f"🔄 RNG Reset: seed={seed} ({context})")

def hash_indices(indices):
    return hashlib.md5(np.array(indices).tobytes()).hexdigest()[:12]

def get_index_fingerprint(indices):
    indices = np.array(indices)
    return {'count': len(indices), 'hash': hash_indices(indices),
            'head_10': indices[:10].tolist(), 'tail_10': indices[-10:].tolist(),
            'min': int(indices.min()), 'max': int(indices.max()), 'sum': int(indices.sum())}

# ============================================================================
# JOINT LABEL STRATIFICATION
# ============================================================================
def create_joint_labels(y_aff, y_spec):
    return y_aff * 2 + y_spec

def get_stratified_kfold(y_aff, y_spec, n_splits=5, random_state=42):
    y_joint = create_joint_labels(y_aff, y_spec)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf, y_joint

def print_joint_label_distribution(y_aff, y_spec, prefix=""):
    y_joint = create_joint_labels(y_aff, y_spec)
    labels = ['(Aff=0,Spec=0)', '(Aff=0,Spec=1)', '(Aff=1,Spec=0)', '(Aff=1,Spec=1)']
    print(f"{prefix}Joint label distribution:")
    for i, label in enumerate(labels):
        count = (y_joint == i).sum()
        print(f"{prefix}  {label}: {count:4d} ({100*count/len(y_joint):5.1f}%)")

# ============================================================================
# FEATURE GENERATION
# ============================================================================
ALPH_LETTERS = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
_le = LabelEncoder()
_integer_encoded_letters = _le.fit_transform(ALPH_LETTERS).reshape(-1, 1)
_one = OneHotEncoder(sparse_output=False)
_ohe_letters = _one.fit_transform(_integer_encoded_letters)

def generate_onehot(seqs_binding):
    ohe = []
    for seq in seqs_binding.index:
        chars = _le.transform(list(seq))
        let = chars.reshape(config.SEQ_LENGTH, 1)
        ohe.append(_one.transform(let).flatten())
    return pd.DataFrame(np.stack(ohe))

# ============================================================================
# HELD-OUT UTILITIES
# ============================================================================
def get_sequences_with_residue(sequences, site_idx, residue):
    return [s for s in sequences if len(s) > site_idx and s[site_idx] == residue]

def get_sequences_without_residue(sequences, site_idx, residue):
    return [s for s in sequences if len(s) > site_idx and s[site_idx] != residue]

def create_holdout_indices(sequences, site_idx, residue, mode='top'):
    seq_list = list(sequences)
    if mode == 'top':
        test_seqs = get_sequences_with_residue(sequences, site_idx, residue)
        train_seqs = get_sequences_without_residue(sequences, site_idx, residue)
    else:
        train_seqs = get_sequences_with_residue(sequences, site_idx, residue)
        test_seqs = get_sequences_without_residue(sequences, site_idx, residue)
    train_idx = np.array([seq_list.index(s) for s in train_seqs])
    test_idx = np.array([seq_list.index(s) for s in test_seqs])
    return train_idx, test_idx

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
def as_torch_float(x, device=None):
    if torch.is_tensor(x):
        tensor = x.float()
    else:
        tensor = torch.as_tensor(x, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def focal_bce_with_logits(y_true, logit, gamma=2.0, pos_weight=1.0):
    logit = as_torch_float(logit)
    y_true = as_torch_float(y_true, device=logit.device)
    pos_weight = torch.as_tensor(pos_weight, dtype=logit.dtype, device=logit.device)
    bce = F.binary_cross_entropy_with_logits(logit, y_true, pos_weight=pos_weight, reduction='none')
    p = torch.sigmoid(logit)
    p_t = y_true * p + (1 - y_true) * (1 - p)
    return ((1.0 - p_t).pow(gamma) * bce).mean()

def safe_gap(scores, labels):
    scores = as_torch_float(scores)
    labels = as_torch_float(labels, device=scores.device)
    pos_mask = labels > 0.5
    neg_mask = labels < 0.5
    if pos_mask.sum().item() > 0 and neg_mask.sum().item() > 0:
        return scores[pos_mask].mean() - scores[neg_mask].mean()
    return scores.new_tensor(0.0)

def gap_hinge_loss(scores, labels, margin=0.2):
    scores = as_torch_float(scores)
    labels = as_torch_float(labels, device=scores.device)
    pos_mask = labels > 0.5
    neg_mask = labels < 0.5
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos.item() > 0 and n_neg.item() > 0:
        gap = scores[pos_mask].mean() - scores[neg_mask].mean()
        return torch.clamp(margin - gap, min=0.0)
    return scores.new_tensor(0.0)

def ranking_loss(scores, labels, margin=0.3):
    scores = as_torch_float(scores)
    labels = as_torch_float(labels, device=scores.device)
    pos_mask = labels > 0.5
    neg_mask = labels < 0.5
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos.item() > 0 and n_neg.item() > 0:
        pos_sc = scores[pos_mask]
        neg_sc = scores[neg_mask]
        diff = pos_sc.unsqueeze(1) - neg_sc.unsqueeze(0)
        return torch.clamp(margin - diff, min=0.0).mean()
    return scores.new_tensor(0.0)

def adversarial_loss(disc_pred_aff, disc_pred_spec):
    disc_pred_aff = as_torch_float(disc_pred_aff)
    disc_pred_spec = as_torch_float(disc_pred_spec, device=disc_pred_aff.device)
    loss_aff = F.binary_cross_entropy(disc_pred_aff, torch.ones_like(disc_pred_aff))
    loss_spec = F.binary_cross_entropy(disc_pred_spec, torch.zeros_like(disc_pred_spec))
    return loss_aff + loss_spec

def independence_loss(z1, z2):
    z1 = as_torch_float(z1)
    z2 = as_torch_float(z2, device=z1.device)
    z1_n = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
    z2_n = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
    corr = torch.matmul(z1_n.T, z2_n) / z1.shape[0]
    return corr.square().mean()

def pareto_diversity_loss(aff_scores, spec_scores):
    """
    Pareto-diversity loss: Encourages the model to spread predictions
    across the affinity-specificity tradeoff surface instead of clustering.
    
    Three components:
    1. SPREAD: Maximize variance of predictions in both dimensions
    2. DECORRELATION: Penalize positive correlation between task scores
       (high corr = all predictions in one corner, not along the frontier)
    3. FRONTIER COVERAGE: Encourage predictions at different tradeoff ratios
    
    Returns: scalar loss to MINIMIZE (higher = worse Pareto spread)
    """
    aff_scores = as_torch_float(aff_scores)
    spec_scores = as_torch_float(spec_scores, device=aff_scores.device)
    aff_p = torch.sigmoid(aff_scores)
    spec_p = torch.sigmoid(spec_scores)
    
    # 1. Spread: penalize low variance in either dimension
    spread_aff = aff_p.std(unbiased=False)
    spread_spec = spec_p.std(unbiased=False)
    spread_loss = -torch.log(spread_aff + 1e-6) - torch.log(spread_spec + 1e-6)
    
    # 2. Decorrelation: penalize positive correlation between scores
    # We want: some seqs high-aff/low-spec, some low-aff/high-spec
    aff_c = aff_p - aff_p.mean()
    spec_c = spec_p - spec_p.mean()
    corr = (aff_c * spec_c).sum() / (torch.norm(aff_c) * torch.norm(spec_c) + 1e-8)
    decorr_loss = torch.clamp(corr, min=0.0)  # Only penalize POSITIVE correlation
    
    # 3. Frontier coverage: encourage diverse tradeoff ratios
    # Angle in (aff, spec) space — want uniform spread of angles
    angles = torch.atan2(spec_p - spec_p.mean(), aff_p - aff_p.mean())
    angle_std = angles.std(unbiased=False)
    coverage_loss = -torch.log(angle_std + 1e-6)
    
    return spread_loss + decorr_loss + 0.5 * coverage_loss

def tfp_correlation(x, y):
    x = as_torch_float(x).reshape(-1)
    y = as_torch_float(y, device=x.device).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / (torch.norm(x) * torch.norm(y) + 1e-8)

# ============================================================================
# METRIC UTILITIES
# ============================================================================
def safe_auc(y_true, y_prob, metric='roc'):
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_prob) if metric == 'roc' else average_precision_score(y_true, y_prob)

def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float('nan')
    return matthews_corrcoef(y_true, y_pred)

def bootstrap_spearman_ci(x, y, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    rhos = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = np.sort(rhos)
    alpha = (1 - ci) / 2
    return np.percentile(rhos, 100 * alpha), np.percentile(rhos, 100 * (1 - alpha))

def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)
    n01 = np.sum((correct_a == 1) & (correct_b == 0))
    n10 = np.sum((correct_a == 0) & (correct_b == 1))
    n_discord = n01 + n10
    if n_discord == 0:
        return 0.0, 1.0
    if n_discord < 25:
        try:
            result = binomtest(n01, n_discord, 0.5)
            p_val = result.pvalue
        except Exception:
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p_val = stats.chi2.sf(chi2, df=1)
    else:
        chi2 = (n01 - n10) ** 2 / (n01 + n10)
        p_val = stats.chi2.sf(chi2, df=1)
    return float(n01 - n10), float(p_val)

def compute_pareto_front(aff_scores, spec_scores):
    n = len(aff_scores)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j: continue
            if (aff_scores[j] >= aff_scores[i] and spec_scores[j] >= spec_scores[i] and
                (aff_scores[j] > aff_scores[i] or spec_scores[j] > spec_scores[i])):
                is_pareto[i] = False
                break
    return is_pareto

# ============================================================================
# DIAGNOSTIC STORAGE
# ============================================================================
@dataclass
class EpochDiagnostics:
    epoch: int
    loss_total: float = 0.0
    loss_cls_aff: float = 0.0; loss_cls_spec: float = 0.0
    loss_rank_aff: float = 0.0; loss_rank_spec: float = 0.0
    loss_gap_aff: float = 0.0; loss_gap_spec: float = 0.0
    loss_adv: float = 0.0; loss_indep: float = 0.0
    acc_aff: float = 0.0; acc_spec: float = 0.0
    aff_score_mean: float = 0.0; aff_score_std: float = 0.0
    spec_score_mean: float = 0.0; spec_score_std: float = 0.0
    aff_gap: float = 0.0; spec_gap: float = 0.0
    aff_latent_mean: float = 0.0; aff_latent_std: float = 0.0
    spec_latent_mean: float = 0.0; spec_latent_std: float = 0.0
    latent_correlation: float = 0.0
    dead_neurons_aff: int = 0; dead_neurons_spec: int = 0
    grad_norm_shared: float = 0.0
    grad_norm_aff_tower: float = 0.0; grad_norm_spec_tower: float = 0.0
    grad_norm_aff_head: float = 0.0; grad_norm_spec_head: float = 0.0
    grad_cosine_aff_spec: float = 0.0
    disc_acc: float = 0.0
    iso_aff_rho: Optional[float] = None; iso_spec_rho: Optional[float] = None
    igg_aff_rho: Optional[float] = None; igg_spec_rho: Optional[float] = None

# ============================================================================
# MODEL: Gradient Reversal
# ============================================================================
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)

# ============================================================================
# MODEL: DiagnosticMOLM
# ============================================================================
class DiagnosticMOLM(nn.Module):
    def __init__(self, input_dim, latent_dim=16, shared_dims=[256, 128],
                 tower_dims=[64, 32], dropout_rate=0.2, grl_lambda=1.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.grl_lambda = grl_lambda
        self._shared_dims = list(shared_dims)
        self._tower_dims = list(tower_dims)
        self._dropout_rate = dropout_rate
        
        self.shared_layers = self._make_blocks(input_dim, shared_dims, dropout_rate)
        
        self.discriminator = nn.Sequential(
            GradientReversalLayer(alpha=grl_lambda),
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        shared_out_dim = shared_dims[-1] if shared_dims else input_dim
        tower_out_dim = tower_dims[-1] if tower_dims else shared_out_dim

        self.aff_layers = self._make_blocks(shared_out_dim, tower_dims, dropout_rate)
        self.aff_proj = nn.Linear(tower_out_dim, latent_dim)
        self.aff_proj_norm = nn.LayerNorm(latent_dim)
        
        self.spec_layers = self._make_blocks(shared_out_dim, tower_dims, dropout_rate)
        self.spec_proj = nn.Linear(tower_out_dim, latent_dim)
        self.spec_proj_norm = nn.LayerNorm(latent_dim)
        
        self.aff_head = nn.Linear(latent_dim, 1)
        self.spec_head = nn.Linear(latent_dim, 1)
        self.to(get_device())

    @staticmethod
    def _make_blocks(input_dim, dims, dropout_rate):
        blocks = nn.ModuleList()
        prev_dim = input_dim
        for dim in dims:
            blocks.append(nn.Linear(prev_dim, dim))
            blocks.append(nn.LayerNorm(dim))
            blocks.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        return blocks

    def _coerce_input(self, inputs):
        device = next(self.parameters()).device
        if torch.is_tensor(inputs):
            return inputs.to(device=device, dtype=torch.float32)
        return torch.as_tensor(inputs, dtype=torch.float32, device=device)

    @staticmethod
    def _apply_dropout(layer, x, training):
        if training is None:
            return layer(x)
        return F.dropout(x, p=layer.p, training=training)
    
    def run_shared(self, x, training):
        return self._run_blocks(x, self.shared_layers, training)
    
    def run_tower(self, x, tower_layers, proj, proj_norm, training):
        x = self._run_blocks(x, tower_layers, training)
        return proj_norm(proj(x))

    def _run_blocks(self, x, blocks, training):
        for i in range(0, len(blocks), 3):
            x = blocks[i](x)
            x = blocks[i+1](x)
            x = F.gelu(x)
            x = self._apply_dropout(blocks[i+2], x, training)
        return x
    
    def run_discriminator(self, x):
        return self.discriminator(x.float())
    
    def forward(self, inputs, training=None):
        inputs = self._coerce_input(inputs)
        shared = self.run_shared(inputs, training)
        aff_latent = self.run_tower(shared, self.aff_layers, self.aff_proj, self.aff_proj_norm, training)
        spec_latent = self.run_tower(shared, self.spec_layers, self.spec_proj, self.spec_proj_norm, training)
        aff_logit = self.aff_head(aff_latent).squeeze(-1)
        spec_logit = self.spec_head(spec_latent).squeeze(-1)
        return {
            'logit_aff': aff_logit, 'logit_spec': spec_logit,
            'z_aff': aff_latent, 'z_spec': spec_latent,
            'aff_score': aff_logit, 'spec_score': spec_logit,
            'aff_prob': torch.sigmoid(aff_logit.float()),
            'spec_prob': torch.sigmoid(spec_logit.float()),
            'aff_latent': aff_latent, 'spec_latent': spec_latent,
            'aff_latent_grl': aff_latent,
            'spec_latent_grl': spec_latent,
            'shared': shared,
        }
    
    def get_config(self):
        return {'input_dim': self.input_dim, 'latent_dim': self.latent_dim,
                'grl_lambda': self.grl_lambda, 'shared_dims': self._shared_dims,
                'tower_dims': self._tower_dims, 'dropout_rate': self._dropout_rate}
    
    def get_layer_groups(self):
        return {
            'shared': list(self.shared_layers),
            'aff_tower': list(self.aff_layers) + [self.aff_proj, self.aff_proj_norm],
            'spec_tower': list(self.spec_layers) + [self.spec_proj, self.spec_proj_norm],
            'aff_head': [self.aff_head], 'spec_head': [self.spec_head],
            'discriminator': list(self.discriminator),
        }
    
    def get_projections(self, inputs):
        self.eval()
        with torch.no_grad():
            out = self(inputs, training=False)
        return {'affinity': out['aff_score'].detach().cpu().numpy().flatten(),
                'specificity': out['spec_score'].detach().cpu().numpy().flatten()}

# ============================================================================
# TRAINER: DiagnosticTrainer (Multi-Task)
# ============================================================================
class DiagnosticTrainer:
    def __init__(self, model, aff_pos_weight=1.0, spec_pos_weight=1.0, learning_rate=5e-5):
        self.device = get_device()
        self.model = model.to(self.device)
        self.aff_pos_weight = aff_pos_weight
        self.spec_pos_weight = spec_pos_weight
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.epoch_diagnostics = []
        self.history = defaultdict(list)

    def _shared_params(self):
        return [p for p in self.model.shared_layers.parameters() if p.requires_grad]

    @staticmethod
    def _flatten_grads(grads, params):
        flat = []
        for grad, param in zip(grads, params):
            if grad is None:
                flat.append(torch.zeros_like(param).reshape(-1))
            else:
                flat.append(grad.reshape(-1))
        if not flat:
            return torch.zeros(1)
        return torch.cat(flat)

    @staticmethod
    def _safe_gap_torch(scores, labels):
        scores = scores.float()
        labels = labels.float().to(scores.device)
        pos_mask = labels > 0.5
        neg_mask = labels < 0.5
        if pos_mask.sum().item() > 0 and neg_mask.sum().item() > 0:
            return scores[pos_mask].mean() - scores[neg_mask].mean()
        return scores.new_tensor(0.0)

    @staticmethod
    def _correlation_torch(x, y):
        x = x.reshape(-1).float()
        y = y.reshape(-1).float().to(x.device)
        x = x - x.mean()
        y = y - y.mean()
        return (x * y).sum() / (torch.norm(x) * torch.norm(y) + 1e-8)

    @staticmethod
    def _adversarial_loss_torch(disc_pred_aff, disc_pred_spec):
        loss_aff = F.binary_cross_entropy(disc_pred_aff, torch.ones_like(disc_pred_aff))
        loss_spec = F.binary_cross_entropy(disc_pred_spec, torch.zeros_like(disc_pred_spec))
        return loss_aff + loss_spec

    @staticmethod
    def _independence_loss_torch(z1, z2):
        z1 = z1.float()
        z2 = z2.float().to(z1.device)
        z1_n = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
        z2_n = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
        corr = torch.matmul(z1_n.T, z2_n) / z1.shape[0]
        return corr.square().mean()

    @staticmethod
    def _pareto_diversity_loss_torch(aff_scores, spec_scores):
        aff_p = torch.sigmoid(aff_scores.float())
        spec_p = torch.sigmoid(spec_scores.float())
        spread_loss = -torch.log(aff_p.std(unbiased=False) + 1e-6) - torch.log(spec_p.std(unbiased=False) + 1e-6)
        aff_c = aff_p - aff_p.mean()
        spec_c = spec_p - spec_p.mean()
        corr = (aff_c * spec_c).sum() / (torch.norm(aff_c) * torch.norm(spec_c) + 1e-8)
        decorr_loss = torch.clamp(corr, min=0.0)
        angles = torch.atan2(spec_p - spec_p.mean(), aff_p - aff_p.mean())
        coverage_loss = -torch.log(angles.std(unbiased=False) + 1e-6)
        return spread_loss + decorr_loss + 0.5 * coverage_loss

    def compute_total_loss(self, out, y_aff, y_spec):
        loss_cls_aff = focal_bce_with_logits(y_aff, out['aff_score'], config.FOCAL_GAMMA, self.aff_pos_weight)
        loss_cls_spec = focal_bce_with_logits(y_spec, out['spec_score'], config.FOCAL_GAMMA, self.spec_pos_weight)
        loss_rank_aff = ranking_loss(out['aff_score'], y_aff, config.RANKING_MARGIN)
        loss_rank_spec = ranking_loss(out['spec_score'], y_spec, config.RANKING_MARGIN)
        loss_gap_aff = gap_hinge_loss(out['aff_score'], y_aff, config.GAP_MARGIN)
        loss_gap_spec = gap_hinge_loss(out['spec_score'], y_spec, config.GAP_MARGIN)

        zero = out['aff_score'].new_tensor(0.0)
        loss_adv = zero
        if config.ADVERSARIAL_WEIGHT != 0:
            disc_on_aff = self.model.run_discriminator(out['aff_latent_grl'])
            disc_on_spec = self.model.run_discriminator(out['spec_latent_grl'])
            loss_adv = self._adversarial_loss_torch(disc_on_aff, disc_on_spec)

        loss_indep = zero
        if config.ORTHO_WEIGHT != 0:
            loss_indep = self._independence_loss_torch(out['aff_latent'], out['spec_latent'])

        total = (loss_cls_aff + loss_cls_spec
                 + config.RANKING_WEIGHT_AFF * loss_rank_aff + config.RANKING_WEIGHT_SPEC * loss_rank_spec
                 + config.GAP_WEIGHT_AFF * loss_gap_aff + config.GAP_WEIGHT_SPEC * loss_gap_spec
                 + config.ADVERSARIAL_WEIGHT * loss_adv + config.ORTHO_WEIGHT * loss_indep)

        loss_pareto = zero
        if config.PARETO_LOSS:
            loss_pareto = self._pareto_diversity_loss_torch(out['aff_score'], out['spec_score'])
            total = total + config.PARETO_LOSS_WEIGHT * loss_pareto

        return {
            'total': total,
            'loss_cls_aff': loss_cls_aff,
            'loss_cls_spec': loss_cls_spec,
            'loss_rank_aff': loss_rank_aff,
            'loss_rank_spec': loss_rank_spec,
            'loss_gap_aff': loss_gap_aff,
            'loss_gap_spec': loss_gap_spec,
            'loss_adv': loss_adv,
            'loss_indep': loss_indep,
            'loss_pareto': loss_pareto,
        }

    def train_step_with_diagnostics(self, x, y_aff, y_spec):
        self.model.train()
        x = x.to(self.device, dtype=torch.float32)
        y_aff = y_aff.to(self.device, dtype=torch.float32)
        y_spec = y_spec.to(self.device, dtype=torch.float32)

        self.optimizer.zero_grad()
        out = self.model(x, training=True)
        losses = self.compute_total_loss(out, y_aff, y_spec)

        shared_params = self._shared_params()
        grad_aff = torch.autograd.grad(losses['loss_cls_aff'], shared_params, retain_graph=True, allow_unused=True)
        grad_spec = torch.autograd.grad(losses['loss_cls_spec'], shared_params, retain_graph=True, allow_unused=True)
        flat_aff = self._flatten_grads(grad_aff, shared_params)
        flat_spec = self._flatten_grads(grad_spec, shared_params)
        cosine_sim = (flat_aff * flat_spec).sum() / (torch.norm(flat_aff) * torch.norm(flat_spec) + 1e-8)

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        acc_aff = ((out['aff_score'] > 0) == (y_aff > 0.5)).float().mean()
        acc_spec = ((out['spec_score'] > 0) == (y_spec > 0.5)).float().mean()
        
        return {
            'loss_total': float(losses['total'].detach().cpu()),
            'loss_cls_aff': float(losses['loss_cls_aff'].detach().cpu()),
            'loss_cls_spec': float(losses['loss_cls_spec'].detach().cpu()),
            'loss_rank_aff': float(losses['loss_rank_aff'].detach().cpu()),
            'loss_rank_spec': float(losses['loss_rank_spec'].detach().cpu()),
            'loss_gap_aff': float(losses['loss_gap_aff'].detach().cpu()),
            'loss_gap_spec': float(losses['loss_gap_spec'].detach().cpu()),
            'loss_adv': float(losses['loss_adv'].detach().cpu()),
            'loss_indep': float(losses['loss_indep'].detach().cpu()),
            'acc_aff': float(acc_aff.detach().cpu()),
            'acc_spec': float(acc_spec.detach().cpu()),
            'aff_gap': float(self._safe_gap_torch(out['aff_score'].detach(), y_aff).cpu()),
            'spec_gap': float(self._safe_gap_torch(out['spec_score'].detach(), y_spec).cpu()),
            'latent_correlation': float(torch.abs(self._correlation_torch(out['aff_latent'].detach(), out['spec_latent'].detach())).cpu()),
            'dead_neurons_aff': int((out['aff_latent'].detach().var(dim=0, unbiased=False) < 1e-6).sum().cpu()),
            'dead_neurons_spec': int((out['spec_latent'].detach().var(dim=0, unbiased=False) < 1e-6).sum().cpu()),
            'grad_cosine_aff_spec': float(cosine_sim.detach().cpu()),
            'aff_score_mean': float(out['aff_score'].detach().mean().cpu()),
            'spec_score_mean': float(out['spec_score'].detach().mean().cpu()),
            'loss_pareto': float(losses['loss_pareto'].detach().cpu()),
        }
    
    def create_dataset(self, X, y_aff, y_spec, batch_size, shuffle=True):
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y_aff), torch.FloatTensor(y_spec))
        generator = torch.Generator()
        generator.manual_seed(SEED)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
    
    def fit(self, X, y_aff, y_spec, epochs=25, batch_size=64,
            X_iso=None, y_iso_aff=None, y_iso_spec=None,
            X_igg=None, y_igg_aff=None, y_igg_spec=None, verbose=1):
        dataset = self.create_dataset(X, y_aff, y_spec, batch_size)
        _pareto_orig = config.PARETO_LOSS  # Save original flag
        for epoch in range(epochs):
            # Pareto warmup: don't apply until training stabilizes
            if _pareto_orig and epoch < config.PARETO_LOSS_WARMUP:
                config.PARETO_LOSS = False
            elif _pareto_orig:
                config.PARETO_LOSS = True
            
            epoch_metrics = defaultdict(list)
            for x_b, ya_b, ys_b in dataset:
                bm = self.train_step_with_diagnostics(x_b, ya_b, ys_b)
                for k, v in bm.items():
                    epoch_metrics[k].append(v)
            
            diag = EpochDiagnostics(epoch=epoch)
            for k in epoch_metrics:
                if hasattr(diag, k):
                    setattr(diag, k, np.mean(epoch_metrics[k]))
            
            if (epoch + 1) % config.GEN_EVAL_EVERY == 0 or epoch == epochs - 1:
                if X_iso is not None:
                    self.model.eval()
                    with torch.no_grad():
                        out_iso = self.model(torch.as_tensor(X_iso, dtype=torch.float32, device=self.device), training=False)
                    diag.iso_aff_rho, _ = stats.spearmanr(out_iso['aff_score'].detach().cpu().numpy(), y_iso_aff)
                    diag.iso_spec_rho, _ = stats.spearmanr(out_iso['spec_score'].detach().cpu().numpy(), y_iso_spec)
                if X_igg is not None:
                    self.model.eval()
                    with torch.no_grad():
                        out_igg = self.model(torch.as_tensor(X_igg, dtype=torch.float32, device=self.device), training=False)
                    diag.igg_aff_rho, _ = stats.spearmanr(out_igg['aff_score'].detach().cpu().numpy()[:42], y_igg_aff[:42])
                    diag.igg_spec_rho, _ = stats.spearmanr(out_igg['spec_score'].detach().cpu().numpy()[:42], y_igg_spec[:42])
            
            self.epoch_diagnostics.append(diag)
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"  Ep {epoch+1:2d}/{epochs} | Acc: {diag.acc_aff:.3f}/{diag.acc_spec:.3f} | "
                      f"Gap: {diag.aff_gap:+.2f}/{diag.spec_gap:+.2f} | GradCos: {diag.grad_cosine_aff_spec:+.3f}",
                      flush=True)
        config.PARETO_LOSS = _pareto_orig  # Restore original flag
        return self.history

# ============================================================================
# TRAINER: MOLMSingleTaskTrainer (MOLM-ST)
# ============================================================================
class MOLMSingleTaskTrainer:
    def __init__(self, model, task='affinity', aff_pos_weight=1.0, spec_pos_weight=1.0,
                 learning_rate=5e-5, ranking_weight=0.3, gap_weight=0.2,
                 focal_gamma=2.0, ranking_margin=0.3, gap_margin=0.2):
        self.device = get_device()
        self.model = model.to(self.device); self.task = task
        self.aff_pos_weight = aff_pos_weight; self.spec_pos_weight = spec_pos_weight
        self.focal_gamma = focal_gamma; self.ranking_margin = ranking_margin
        self.gap_margin = gap_margin; self.ranking_weight = ranking_weight; self.gap_weight = gap_weight
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    def train_step(self, x, y_aff, y_spec):
        self.model.train()
        x = x.to(self.device, dtype=torch.float32)
        y_aff = y_aff.to(self.device, dtype=torch.float32)
        y_spec = y_spec.to(self.device, dtype=torch.float32)

        self.optimizer.zero_grad()
        out = self.model(x, training=True)
        score = out['aff_score'] if self.task == 'affinity' else out['spec_score']
        labels = y_aff if self.task == 'affinity' else y_spec
        pw = self.aff_pos_weight if self.task == 'affinity' else self.spec_pos_weight
        total = (focal_bce_with_logits(labels, score, self.focal_gamma, pw)
                 + self.ranking_weight * ranking_loss(score, labels, self.ranking_margin)
                 + self.gap_weight * gap_hinge_loss(score, labels, self.gap_margin))
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return total
    
    def fit(self, X, y_aff, y_spec, epochs=25, batch_size=64, verbose=0):
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y_aff), torch.FloatTensor(y_spec))
        generator = torch.Generator()
        generator.manual_seed(SEED)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)
        for epoch in range(epochs):
            losses = []
            for x_b, ya_b, ys_b in loader:
                loss = self.train_step(x_b, ya_b, ys_b)
                losses.append(float(loss.detach().cpu()))
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"  ST {self.task} Ep {epoch+1:2d}/{epochs} | Loss: {np.mean(losses):.4f}", flush=True)

def train_molm_st(X, y_aff, y_spec, task, config_obj, aff_pos_weight=1.0, spec_pos_weight=1.0,
                  seed_offset=0, verbose=0):
    hard_reset_rng(SEED + seed_offset, f"MOLM-ST {task}")
    model = DiagnosticMOLM(input_dim=X.shape[1], latent_dim=config_obj.LATENT_DIM,
                           shared_dims=config_obj.SHARED_DIMS, tower_dims=config_obj.TOWER_DIMS,
                           dropout_rate=config_obj.DROPOUT_RATE, grl_lambda=config_obj.GRL_LAMBDA)
    rw = config_obj.RANKING_WEIGHT_AFF if task == 'affinity' else config_obj.RANKING_WEIGHT_SPEC
    gw = config_obj.GAP_WEIGHT_AFF if task == 'affinity' else config_obj.GAP_WEIGHT_SPEC
    trainer = MOLMSingleTaskTrainer(model, task=task, aff_pos_weight=aff_pos_weight,
                                     spec_pos_weight=spec_pos_weight, learning_rate=config_obj.LEARNING_RATE,
                                     ranking_weight=rw, gap_weight=gw, focal_gamma=config_obj.FOCAL_GAMMA,
                                     ranking_margin=config_obj.RANKING_MARGIN, gap_margin=config_obj.GAP_MARGIN)
    trainer.fit(X, y_aff, y_spec, epochs=config_obj.EPOCHS, batch_size=config_obj.BATCH_SIZE,
                verbose=verbose)
    return model

# ============================================================================
# BASELINE: DeepProjectorDecider NN
# ============================================================================
class DeepProjectorDecider(nn.Module):
    def __init__(self, input_dim, intermed_dim=20, proj_dim=1, **kwargs):
        super().__init__()
        self._input_dim = input_dim
        self._intermed_dim = intermed_dim
        self._proj_dim = proj_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, intermed_dim),
            nn.ReLU(),
            nn.Linear(intermed_dim, proj_dim),
        )
        self.decider = nn.Linear(proj_dim, 2)
        self.to(get_device())

    def _coerce_input(self, x):
        device = next(self.parameters()).device
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    
    def forward(self, x, training=None):
        x = self._coerce_input(x)
        return self.decider(self.projector(x))
    
    def get_projection(self, x):
        self.eval()
        with torch.no_grad():
            return self.projector(self._coerce_input(x)).detach().cpu()
    
    def get_config(self):
        return {'input_dim': self._input_dim, 'intermed_dim': self._intermed_dim, 'proj_dim': self._proj_dim}

def train_deep_projector_decider(X_train, y_train, input_dim=None, intermed_dim=20,
                                 proj_dim=1, epochs=50, batch_size=50, seed=SEED,
                                 learning_rate=1e-3):
    if input_dim is None:
        input_dim = X_train.shape[1]
    device = get_device()
    model = DeepProjectorDecider(input_dim=input_dim, intermed_dim=intermed_dim,
                                 proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model.train()
    for _ in range(epochs):
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_b), y_b)
            loss.backward()
            optimizer.step()
    return model

def predict_deep_projector_logits(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))
    return logits.detach().cpu().numpy()

# ============================================================================
# SAVE/LOAD HELPERS
# ============================================================================
def save_phase_data(data, name, phase=None):
    artifact_phase = get_artifact_phase(name, phase)
    path = phase_output_path(f"{name}.pkl", artifact_phase) if artifact_phase else os.path.join(config.OUTPUT_DIR, f"{name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  ✓ Saved: {path}")

def _safe_pickle_load(path):
    original_loader = torch.storage._load_from_bytes
    map_location = get_device() if torch.cuda.is_available() else torch.device("cpu")
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location=map_location)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = original_loader

def load_phase_data(name, phase=None):
    candidates = get_artifact_candidates(name, ".pkl", phase)
    for path in candidates:
        if os.path.exists(path):
            data = _safe_pickle_load(path)
            print(f"  ✓ Loaded: {path}")
            return data
    expected = candidates[0]
    raise FileNotFoundError(f"Phase data not found: {expected}\n  Run the previous phase first.")

def save_results_csv(results_dict, name, phase=None):
    artifact_phase = get_artifact_phase(name, phase)
    path = phase_output_path(f"{name}.csv", artifact_phase) if artifact_phase else os.path.join(config.OUTPUT_DIR, f"{name}.csv")
    pd.DataFrame([results_dict]).to_csv(path, index=False)
    print(f"  ✓ Saved: {path}")

def save_results_json(results_dict, name, phase=None):
    artifact_phase = get_artifact_phase(name, phase)
    path = phase_output_path(f"{name}.json", artifact_phase) if artifact_phase else os.path.join(config.OUTPUT_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"  ✓ Saved: {path}")

print("✓ Phase 0 loaded: Config, models, utilities ready")
print(f"  OUTPUT_DIR: {config.OUTPUT_DIR}")
print(f"  FEATURE_TYPES: {config.FEATURE_TYPES}")
print(f"  GPU: {torch.cuda.device_count()} available")
