#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import math
import json
import random
import argparse
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path
from glob import glob

import cv2
import timm
import numpy as np
import pandas as pd
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm.auto import tqdm

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from transformers import AutoProcessor, AutoImageProcessor, AutoModel, AutoTokenizer
try:
    from google.protobuf import message_factory as _pb_message_factory
    from google.protobuf.internal import message_factory as _pb_internal_factory
    def _ensure_getprototype(mf):
        if not hasattr(mf, 'GetPrototype'):
            def _getprototype(self, descriptor):
                return self.GetMessageClass(descriptor)
            mf.GetPrototype = _getprototype
    _ensure_getprototype(_pb_message_factory.MessageFactory)
    _ensure_getprototype(_pb_internal_factory.MessageFactory)
except Exception:
    pass



# In[2]:


def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def _fast_mode_flag() -> bool:
    try:
        return bool(int(os.environ.get("FAST_MODE", "0")))
    except ValueError:
        return False


@dataclass
class Config:
    DATA_PATH: Path = Path("/kaggle/input/csiro-biomass/")
    TRAIN_DATA_PATH: Path = DATA_PATH/'train'
    TEST_DATA_PATH: Path = DATA_PATH/'test'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    stack_kfolds: int = 10
    legacy_kfolds: int = 8
    dino_kfolds: int = 10
    siglip_hybrid_weight: float = 0.5
    # Execution controls / staging
    run_siglip_stack: bool = True
    run_siglip_legacy: bool = True
    run_dino1: bool = True
    run_mvp: bool = True
    run_dinov2: bool = True
    # Fast-mode / performance controls
    fast_mode: bool = field(default_factory=_fast_mode_flag)
    use_semantic_features: bool = True
    siglip_cluster_sizes: Tuple[int, ...] = (12, 24, 36, 48, 64)
    dino_cluster_sizes: Tuple[int, ...] = (16, 32, 48, 64, 80)
    siglip_fast_models: Tuple[str, ...] = field(default_factory=tuple)
    dino_fast_models: Tuple[str, ...] = field(default_factory=tuple)
    enable_legacy_siglip: bool = True

    def __post_init__(self):
        if self.fast_mode:
            self.stack_kfolds = min(self.stack_kfolds, 5)
            self.legacy_kfolds = min(self.legacy_kfolds, 4)
            self.dino_kfolds = min(self.dino_kfolds, 5)
            self.siglip_cluster_sizes = (16, 32)
            self.dino_cluster_sizes = (24, 48)
            self.use_semantic_features = False
            self.enable_legacy_siglip = False
            self.run_siglip_legacy = False
            # Keep MVP for final blend; drop Dinov2 to shorten runtime
            self.run_dinov2 = False
            # SigLIP stack dominates in fast mode
            self.siglip_fast_models = (
                "cat_seed0",
                "svr_seed0",
                "xgb_seed0",
                "bayes_ridge",
                "elastic_net",
            )
            self.dino_fast_models = ("dinov2_cat", "dinov2_xgb")

cfg = Config()
seeding(cfg.seed)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def maybe_parallelize(model, device=None, min_batch=2):
    if device is not None and hasattr(model, 'to'):
        model.to(device)
    use_multi = (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and (min_batch is None or min_batch >= torch.cuda.device_count())
    )
    if use_multi:
        model = torch.nn.DataParallel(model)
    return model


def assign_folds(df: pd.DataFrame, n_splits: int, seed: int):
    fold_assign = df[['image_path']].drop_duplicates().reset_index(drop=True)
    fold_assign['fold'] = -1
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(kf.split(fold_assign)):
        fold_assign.loc[val_idx, 'fold'] = fold
    df = df.drop(columns=['fold'], errors='ignore').merge(fold_assign, on='image_path', how='left')
    df['fold'] = df['fold'].astype(int)
    return df


def ensure_abs_path(path_str: str) -> str:
    path_str = str(path_str)
    if path_str.startswith('/') or path_str.startswith('\\'):
        return path_str
    return str(cfg.DATA_PATH / path_str)


# In[3]:


TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
weights = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,
}
TARGET_MAX = {
    "Dry_Clover_g": 71.7865,
    "Dry_Dead_g": 83.8407,
    "Dry_Green_g": 157.9836,
    "Dry_Total_g": 185.70,
    "GDM_g": 157.9836,
}

def competition_metric(y_true, y_pred) -> float:
    y_weighted = 0
    for l, label in enumerate(TARGET_NAMES):
        y_weighted = y_weighted + y_true[:, l].mean() * weights[label]
    ss_res = 0
    ss_tot = 0
    for l, label in enumerate(TARGET_NAMES):
        ss_res = ss_res + ((y_true[:, l] - y_pred[:, l])**2).mean() * weights[label]
        ss_tot = ss_tot + ((y_true[:, l] - y_weighted)**2).mean() * weights[label]
    return 1 - ss_res / ss_tot

def pivot_table(df: pd.DataFrame)->pd.DataFrame:
    if 'target' in df.columns.tolist():
        df_pt = pd.pivot_table(
            df,
            values='target',
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name',
            aggfunc='mean'
        ).reset_index()
    else:
        df['target'] = 0
        df_pt = pd.pivot_table(
            df,
            values='target',
            index='image_path',
            columns='target_name',
            aggfunc='mean'
        ).reset_index()
    return df_pt

def melt_table(df: pd.DataFrame) -> pd.DataFrame:
    melted = df.melt(
        id_vars='image_path',
        value_vars=TARGET_NAMES,
        var_name='target_name',
        value_name='target'
    )
    melted['sample_id'] = (
        melted['image_path']
        .str.replace(r'^.*/', '', regex=True)
        .str.replace('.jpg', '', regex=False)
        + '__' + melted['target_name']
    )
    return melted[['sample_id', 'image_path', 'target_name', 'target']]

def post_process_biomass(df_preds):
    ordered_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    Y = df_preds[ordered_cols].values.T
    C = np.array([[1, 1, 0, -1,  0], [0, 0, 1,  1, -1]])
    C_T = C.T
    inv_CCt = np.linalg.inv(C @ C_T)
    P = np.eye(5) - C_T @ inv_CCt @ C
    Y_reconciled = P @ Y
    Y_reconciled = Y_reconciled.T.clip(min=0)
    df_out = df_preds.copy()
    df_out[ordered_cols] = Y_reconciled
    return df_out

def split_image(image, patch_size=520, overlap=16):
    h, w, c = image.shape
    stride = patch_size - overlap
    patches, coords = [], []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1, y2, x2 = y, x, y + patch_size, x + patch_size
            patch = image[y1:y2, x1:x2, :]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0,pad_h), (0,pad_w), (0,0)), mode='reflect')
            patches.append(patch)
            coords.append((y1, x1, y2, x2))
    return patches, coords

def get_model(model_path: str, device: str = 'cpu'):
    model_lower = model_path.lower()
    if 'clip' in model_lower:
        from transformers import CLIPModel, CLIPImageProcessor
        model = CLIPModel.from_pretrained(model_path, local_files_only=True)
        try:
            processor = CLIPImageProcessor.from_pretrained(model_path, local_files_only=True)
        except OSError:
            processor = CLIPImageProcessor.from_dict({
                'do_resize': True,
                'size': {'shortest_edge': 336},
                'resample': 3,
                'do_center_crop': True,
                'crop_size': {'height': 336, 'width': 336},
                'do_normalize': True,
                'image_mean': [0.48145466, 0.4578275, 0.40821073],
                'image_std': [0.26862954, 0.26130258, 0.27577711],
            })
    else:
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
    return model.eval().to(device), processor

def compute_embeddings(model_path, df, patch_size=520):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = get_model(model_path=model_path, device=device)
    IMAGE_PATHS, EMBEDDINGS = [], []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        patches, coords = split_image(img, patch_size=patch_size)
        images = [Image.fromarray(p).convert("RGB") for p in patches]
        inputs = processor(images=images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model_lower = model_path.lower()
            if 'siglip' in model_lower or 'clip' in model_lower:
                if hasattr(model, 'get_image_features'):
                    features = model.get_image_features(**inputs, interpolate_pos_encoding=True)
                else:
                    outputs = model(**inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        features = outputs.last_hidden_state.mean(dim=1)
                    else:
                        features = outputs[0].mean(dim=1)
            elif 'dino' in model_lower:
                outputs = model(**inputs)
                if hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                else:
                    features = outputs[0]
            else:
                outputs = model(**inputs)
                if hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state.mean(dim=1)
                else:
                    features = outputs[0]

        embeds = features.mean(dim=0).detach().cpu().numpy()
        EMBEDDINGS.append(embeds)
        IMAGE_PATHS.append(img_path)
    embeddings = np.stack(EMBEDDINGS, axis=0)
    n_features = embeddings.shape[1]
    emb_columns = [f"emb{i+1}" for i in range(n_features)]
    emb_df = pd.DataFrame(embeddings, columns=emb_columns)
    emb_df['image_path'] = IMAGE_PATHS
    df_final = df.merge(emb_df, on='image_path', how='left')
    flush()
    return df_final

def generate_semantic_features(image_embeddings, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = AutoModel.from_pretrained(
            model_path, local_files_only=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, local_files_only=True
        )
    except Exception as e:
        return None
    concept_groups = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass", "overgrown vegetation"],
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material", "standing hay"],
        "clover": ["white clover", "trifolium repens", "broadleaf legume", "clover flowers"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"],
        "weeds": ["broadleaf weeds", "thistles", "non-pasture vegetation"]
    }
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concept_groups.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)
    if isinstance(image_embeddings, np.ndarray):
        img_tensor = torch.tensor(image_embeddings, dtype=torch.float32).to(device)
    else:
        img_tensor = image_embeddings.to(device)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
    scores = {}
    for name, vec in concept_vectors.items():
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()
    df_scores = pd.DataFrame(scores)
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    df_scores['ratio_clover'] = df_scores['clover'] / (df_scores['clover'] + df_scores['grass'] + 1e-6)
    df_scores['ratio_cover'] = (df_scores['dense'] + df_scores['medium']) / (df_scores['bare'] + df_scores['sparse'] + 1e-6)
    df_scores['max_density'] = df_scores[['bare', 'sparse', 'medium', 'dense']].max(axis=1)
    return df_scores.values


def append_cluster_features(train_df, test_df, embed_prefix='emb', cluster_sizes=(12, 24, 36, 48, 64), random_state=42):
    embed_cols = [c for c in train_df.columns if c.startswith(embed_prefix)]
    if not embed_cols:
        return train_df, test_df
    train_emb = train_df[embed_cols].values.astype(np.float32, copy=False)
    test_emb = test_df[embed_cols].values.astype(np.float32, copy=False)
    stacked = np.vstack([train_emb, test_emb]).astype(np.float32, copy=False)
    if len(stacked) == 0:
        return train_df, test_df
    batch_size = min(4096, max(256, stacked.shape[0]))
    for n_clusters in cluster_sizes:
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            continue
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
            n_init=10,
        )
        km.fit(stacked)
        train_df[f"meta_cluster_{n_clusters}"] = km.predict(train_emb)
        test_df[f"meta_cluster_{n_clusters}"] = km.predict(test_emb)
    return train_df, test_df


# In[4]:


class SupervisedEmbeddingEngine(BaseEstimator, TransformerMixin):
    def __init__(self, n_pca=0.98, n_pls=8, n_gmm=5, random_state=42):
        self.n_pca = n_pca
        self.n_pls = n_pls
        self.n_gmm = n_gmm
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=random_state)
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=random_state)
        self.pls_fitted_ = False

    def fit(self, X, y=None, X_semantic=None):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        if y is not None:
            y_clean = y.values if hasattr(y, 'values') else y
            y_array = np.asarray(y_clean)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            rank_bound = min(max(1, X_scaled.shape[0] - 1), X_scaled.shape[1], y_array.shape[1])
            n_comp = min(self.n_pls, rank_bound)
            if n_comp >= 1:
                if getattr(self.pls, 'n_components', None) != n_comp:
                    self.pls = PLSRegression(n_components=n_comp, scale=False)
                self.pls.fit(X_scaled, y_array)
                self.pls_fitted_ = True
            else:
                self.pls_fitted_ = False
        return self

    def transform(self, X, X_semantic=None):
        X_scaled = self.scaler.transform(X)
        return self._generate_features(X_scaled, X_semantic)

    def _generate_features(self, X_scaled, X_semantic=None):
        features = []
        f_pca = self.pca.transform(X_scaled)
        features.append(f_pca)
        if self.pls_fitted_:
            f_pls = self.pls.transform(X_scaled)
            features.append(f_pls)
        f_gmm = self.gmm.predict_proba(X_scaled)
        features.append(f_gmm)
        if X_semantic is not None:
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            features.append(sem_norm)
        return np.hstack(features)


# In[5]:


def compare_results(oof, train_data, return_scores=False):
    y_oof_df = pd.DataFrame(oof, columns=TARGET_NAMES)
    raw_score = competition_metric(train_data[TARGET_NAMES].values, y_oof_df.values)
    print(f"Raw CV Score: {raw_score:.6f}")
    y_oof_proc = post_process_biomass(y_oof_df)
    proc_score = competition_metric(train_data[TARGET_NAMES].values, y_oof_proc.values)
    print(f"Processed CV Score: {proc_score:.6f}")
    print(f"Weight for blend (processed): {proc_score:.6f}")
    print(f"Improvement: {raw_score - proc_score:.6f}")
    if return_scores:
        return raw_score, proc_score

def cross_validate(model, train_data, test_data, feature_engine, semantic_train=None, semantic_test=None, target_transform='max', seed=42, n_splits_override=None):
    n_splits = n_splits_override or train_data['fold'].nunique()
    target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
    y_true = train_data[TARGET_NAMES]
    y_pred = pd.DataFrame(0.0, index=train_data.index, columns=TARGET_NAMES)
    y_pred_test = np.zeros([len(test_data), len(TARGET_NAMES)], dtype=float)

    # Feature space: raw embeddings + cluster assignments
    emb_columns = [col for col in train_data.columns if col.startswith('emb')]
    cluster_columns = [col for col in train_data.columns if col.startswith('meta_cluster_')]
    COLUMNS = emb_columns + cluster_columns

    for fold in range(n_splits):
        seeding(seed*(seed//2 + fold))
        train_mask = train_data['fold'] != fold
        valid_mask = train_data['fold'] == fold
        val_idx = train_data[valid_mask].index
        X_train_raw = train_data[train_mask][COLUMNS].values
        X_valid_raw = train_data[valid_mask][COLUMNS].values
        X_test_raw = test_data[COLUMNS].values
        sem_train_fold = semantic_train[train_mask] if semantic_train is not None else None
        sem_valid_fold = semantic_train[valid_mask] if semantic_train is not None else None
        y_train = train_data[train_mask][TARGET_NAMES].values
        y_valid = train_data[valid_mask][TARGET_NAMES].values
        if target_transform == 'log':
            y_train_proc = np.log1p(y_train)
        elif target_transform == 'max':
            y_train_proc = y_train / target_max_arr
        else:
            y_train_proc = y_train
        engine = deepcopy(feature_engine)
        engine.fit(X_train_raw, y=y_train_proc, X_semantic=sem_train_fold)
        x_train_eng = engine.transform(X_train_raw, X_semantic=sem_train_fold)
        x_valid_eng = engine.transform(X_valid_raw, X_semantic=sem_valid_fold)
        x_test_eng = engine.transform(X_test_raw, X_semantic=semantic_test)
        fold_valid_pred = np.zeros_like(y_valid)
        fold_test_pred = np.zeros([len(test_data), len(TARGET_NAMES)])
        for k in range(len(TARGET_NAMES)):
            regr = deepcopy(model)
            regr.fit(x_train_eng, y_train_proc[:, k])
            pred_valid_raw = regr.predict(x_valid_eng)
            pred_test_raw = regr.predict(x_test_eng)
            if target_transform == 'log':
                pred_valid_inv = np.expm1(pred_valid_raw)
                pred_test_inv = np.expm1(pred_test_raw)
            elif target_transform == 'max':
                pred_valid_inv = (pred_valid_raw * target_max_arr[k])
                pred_test_inv = (pred_test_raw * target_max_arr[k])
            else:
                pred_valid_inv = pred_valid_raw
                pred_test_inv = pred_test_raw
            fold_valid_pred[:, k] = pred_valid_inv
            fold_test_pred[:, k] = pred_test_inv
        y_pred.loc[val_idx] = fold_valid_pred
        y_pred_test += fold_test_pred / n_splits
    full_cv = competition_metric(y_true.values, y_pred.values)
    print(f"Full CV Score: {full_cv:.6f}")
    return y_pred.values, y_pred_test

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, dim), nn.Dropout(dropout))

    def forward(self, x): return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, heads=4, depth=2, patch=(2, 2), dropout=0.0):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1), nn.GELU())
        self.patch = patch
        self.transformer = nn.ModuleList([AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)])
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        local_feat = self.local(x)
        B, C, H, W = local_feat.shape
        ph, pw = self.patch
        new_h = math.ceil(H / ph) * ph
        new_w = math.ceil(W / pw) * pw
        if new_h != H or new_w != W:
            local_feat = F.interpolate(local_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
            H, W = new_h, new_w
        tokens = local_feat.unfold(2, ph, ph).unfold(3, pw, pw)
        tokens = tokens.contiguous().view(B, C, -1, ph, pw)
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        for blk in self.transformer: tokens = blk(tokens)
        feat = tokens.view(B, -1, ph * pw, C).permute(0, 3, 1, 2)
        nh = H // ph
        nw = W // pw
        feat = feat.view(B, C, nh, nw, ph, pw).permute(0, 1, 2, 4, 3, 5)
        feat = feat.reshape(B, C, H, W)
        if feat.shape[-2:] != x.shape[-2:]:
            feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, feat], dim=1))

class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else: self.sr = None
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, hw: Tuple[int, int]):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        if self.sr is not None:
            H, W = hw
            feat = x.transpose(1, 2).reshape(B, C, H, W)
            feat = self.sr(feat)
            feat = feat.reshape(B, C, -1).transpose(1, 2)
            feat = self.norm(feat)
        else: feat = x
        kv = self.kv(feat)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 3, 1)
        v = v.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)

class PVTBlock(nn.Module):
    def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.sra = SpatialReductionAttention(dim, heads=heads, sr_ratio=sr_ratio, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, hw: Tuple[int, int]):
        x = x + self.sra(self.norm1(x), hw)
        x = x + self.ff(self.norm2(x))
        return x

class LocalMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = (x * g).transpose(1, 2)
        x = self.dwconv(x).transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x

class T2TRetokenizer(nn.Module):
    def __init__(self, dim, depth=2, heads=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)])

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]):
        B, T, C = tokens.shape
        H, W = grid_hw
        feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
        seq = feat_map.flatten(2).transpose(1, 2)
        for blk in self.blocks: seq = blk(seq)
        seq_map = seq.transpose(1, 2).reshape(B, C, H, W)
        pooled = F.adaptive_avg_pool2d(seq_map, (2, 2))
        retokens = pooled.flatten(2).transpose(1, 2)
        return retokens, seq_map

class CrossScaleFusion(nn.Module):
    def __init__(self, dim, heads=6, dropout=0.0, layers=2):
        super().__init__()
        self.layers_s = nn.ModuleList([AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)])
        self.layers_b = nn.ModuleList([AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)])
        self.cross_s = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim) for _ in range(layers)])
        self.cross_b = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim) for _ in range(layers)])
        self.norm_s = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, tok_s: torch.Tensor, tok_b: torch.Tensor):
        B, Ts, C = tok_s.shape
        Tb = tok_b.shape[1]
        cls_s = tok_s.new_zeros(B, 1, C)
        cls_b = tok_b.new_zeros(B, 1, C)
        tok_s = torch.cat([cls_s, tok_s], dim=1)
        tok_b = torch.cat([cls_b, tok_b], dim=1)
        for ls, lb, cs, cb in zip(self.layers_s, self.layers_b, self.cross_s, self.cross_b):
            tok_s = ls(tok_s)
            tok_b = lb(tok_b)
            q_s = self.norm_s(tok_s[:, :1])
            q_b = self.norm_b(tok_b[:, :1])
            cls_s_upd, _ = cs(q_s, torch.cat([tok_b, q_b], dim=1), torch.cat([tok_b, q_b], dim=1), need_weights=False)
            cls_b_upd, _ = cb(q_b, torch.cat([tok_s, q_s], dim=1), torch.cat([tok_s, q_s], dim=1), need_weights=False)
            tok_s = torch.cat([tok_s[:, :1] + cls_s_upd, tok_s[:, 1:]], dim=1)
            tok_b = torch.cat([tok_b[:, :1] + cls_b_upd, tok_b[:, 1:]], dim=1)
        tokens = torch.cat([tok_s[:, :1], tok_b[:, :1], tok_s[:, 1:], tok_b[:, 1:]], dim=1)
        return tokens

class TileEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, input_res: int):
        super().__init__()
        self.backbone = backbone
        self.input_res = input_res

    def forward(self, x: torch.Tensor, grid: Tuple[int, int]):
        B, C, H, W = x.shape
        r, c = grid
        hs = torch.linspace(0, H, steps=r + 1, device=x.device).round().long()
        ws = torch.linspace(0, W, steps=c + 1, device=x.device).round().long()
        tiles = []
        for i in range(r):
            for j in range(c):
                rs, re = hs[i].item(), hs[i + 1].item()
                cs, ce = ws[j].item(), ws[j + 1].item()
                xt = x[:, :, rs:re, cs:ce]
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res), mode="bilinear", align_corners=False)
                tiles.append(xt)
        tiles = torch.stack(tiles, dim=1)
        flat = tiles.view(-1, C, self.input_res, self.input_res)
        feats = self.backbone(flat)
        return feats.view(B, -1, feats.shape[-1])

class PyramidMixer(nn.Module):
    def __init__(self, dim_in: int, dims: Tuple[int, int, int], mobilevit_heads=4, mobilevit_depth=2, sra_heads=6, sra_ratio=2, mamba_depth=3, mamba_kernel=5, dropout=0.0):
        super().__init__()
        c1, c2, c3 = dims
        self.proj1 = nn.Linear(dim_in, c1)
        self.mobilevit = MobileViTBlock(c1, heads=mobilevit_heads, depth=mobilevit_depth, dropout=dropout)
        self.proj2 = nn.Linear(c1, c2)
        self.pvt = PVTBlock(c2, heads=sra_heads, sr_ratio=sra_ratio, dropout=dropout, mlp_ratio=3.0)
        self.mamba_local = LocalMambaBlock(c2, kernel_size=mamba_kernel, dropout=dropout)
        self.proj3 = nn.Linear(c2, c3)
        self.mamba_global = nn.ModuleList([LocalMambaBlock(c3, kernel_size=mamba_kernel, dropout=dropout) for _ in range(mamba_depth)])
        self.final_attn = AttentionBlock(c3, heads=min(8, c3//64+1), dropout=dropout, mlp_ratio=2.0)

    def _tokens_to_map(self, tokens: torch.Tensor, target_hw: Tuple[int, int]):
        B, N, C = tokens.shape
        H, W = target_hw
        need = H * W
        if N < need:
            pad = tokens.new_zeros(B, need-N, C)
            tokens = torch.cat([tokens, pad], dim=1)
        tokens = tokens[:, :need, :]
        return tokens.transpose(1, 2).reshape(B, C, H, W)

    @staticmethod
    def _fit_hw(n_tokens: int) -> Tuple[int, int]:
        h = int(math.sqrt(n_tokens))
        w = h
        while h * w < n_tokens:
            w += 1
            if h * w < n_tokens: h += 1
        return h, w

    def forward(self, tokens: torch.Tensor):
        B, N, C = tokens.shape
        map_hw = (3, 4)
        feat_map = self._tokens_to_map(tokens, map_hw)
        t1 = self.proj1(tokens)
        m1 = self._tokens_to_map(t1, map_hw)
        m1 = self.mobilevit(m1)
        t1_out = m1.flatten(2).transpose(1, 2)[:, :N]
        t2 = self.proj2(t1_out)
        new_len = max(4, N//2)
        t2 = t2[:, :new_len] + F.adaptive_avg_pool1d(t2.transpose(1, 2), new_len).transpose(1, 2)
        hw2 = self._fit_hw(t2.size(1))
        if t2.size(1) < hw2[0] * hw2[1]:
            pad = t2.new_zeros(B, hw2[0]*hw2[1]-t2.size(1), t2.size(2))
            t2 = torch.cat([t2, pad], dim=1)
        t2 = self.pvt(t2, hw2)
        t2 = self.mamba_local(t2)
        t3 = self.proj3(t2)
        pooled = torch.stack([t3.mean(dim=1), t3.max(dim=1).values], dim=1)
        t3 = pooled
        for blk in self.mamba_global: t3 = blk(t3)
        t3 = self.final_attn(t3)
        return t3.mean(dim=1), {"stage1_map": m1.detach(), "stage2_tokens": t2.detach(), "stage3_tokens": t3.detach()}

@dataclass
class TrainCFG:
    dropout: float = 0.1
    hidden_ratio: float = 0.35
    dino_candidates: Tuple[str, ...] = ("vit_base_patch14_dinov2", "vit_base_patch14_reg4_dinov2", "vit_small_patch14_dinov2")
    small_grid: Tuple[int, int] = (4, 4)
    big_grid: Tuple[int, int] = (2, 2)
    t2t_depth: int = 2
    cross_layers: int = 2
    cross_heads: int = 6
    pyramid_dims: Tuple[int, int, int] = (384, 512, 640)
    mobilevit_heads: int = 4
    mobilevit_depth: int = 2
    sra_heads: int = 8
    sra_ratio: int = 2
    mamba_depth: int = 3
    mamba_kernel: int = 5
    aux_head: bool = True
    aux_loss_weight: float = 0.4
    ALL_TARGET_COLS: Tuple[str, ...] = ("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g")

CFG = TrainCFG()

def update_cfg_from_checkpoint(cfg_dict: dict):
    global CFG
    if not cfg_dict: return
    for k, v in cfg_dict.items():
        if hasattr(CFG, k): setattr(CFG, k, v)

class CrossPVT_T2T_MambaDINO(nn.Module):
    def __init__(self, dropout: float = 0.1, hidden_ratio: float = 0.35):
        super().__init__()
        self.backbone, self.feat_dim, self.backbone_name, self.input_res = self._build_dino_backbone()
        self.tile_encoder = TileEncoder(self.backbone, self.input_res)
        self.t2t = T2TRetokenizer(self.feat_dim, depth=CFG.t2t_depth, heads=CFG.cross_heads, dropout=dropout)
        self.cross = CrossScaleFusion(self.feat_dim, heads=CFG.cross_heads, dropout=dropout, layers=CFG.cross_layers)
        self.pyramid = PyramidMixer(dim_in=self.feat_dim, dims=CFG.pyramid_dims, mobilevit_heads=CFG.mobilevit_heads, mobilevit_depth=CFG.mobilevit_depth, sra_heads=CFG.sra_heads, sra_ratio=CFG.sra_ratio, mamba_depth=CFG.mamba_depth, mamba_kernel=CFG.mamba_kernel, dropout=dropout)
        self.combined_dim = CFG.pyramid_dims[-1] * 2
        hidden = max(32, int(self.combined_dim * hidden_ratio))
        def head(): return nn.Sequential(nn.Linear(self.combined_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.head_green = head()
        self.head_clover = head()
        self.head_dead = head()
        self.score_head = nn.Sequential(nn.LayerNorm(self.combined_dim), nn.Linear(self.combined_dim, 1))
        self.aux_head = nn.Sequential(nn.LayerNorm(CFG.pyramid_dims[1]), nn.Linear(CFG.pyramid_dims[1], 5)) if CFG.aux_head else None
        self.softplus = nn.Softplus(beta=1.0)
        self.cross_gate_left = nn.Linear(CFG.pyramid_dims[-1], CFG.pyramid_dims[-1])
        self.cross_gate_right = nn.Linear(CFG.pyramid_dims[-1], CFG.pyramid_dims[-1])

    def _build_dino_backbone(self):
        last_err = None
        for name in CFG.dino_candidates:
            for gp in ["token", "avg", "__default__"]:
                try:
                    if gp == "__default__":
                        m = timm.create_model(name, pretrained=False, num_classes=0)
                        gp_str = "default"
                    else:
                        m = timm.create_model(name, pretrained=False, num_classes=0, global_pool=gp)
                        gp_str = gp
                    feat = m.num_features
                    input_res = self._infer_input_res(m)
                    if hasattr(m, "set_grad_checkpointing"):
                        m.set_grad_checkpointing(True)
                    return m, feat, name, int(input_res)
                except Exception as e: last_err = e; continue
        raise RuntimeError(f"无法创建任何 DINO 主干。最后错误: {last_err}")

    @staticmethod
    def _infer_input_res(m) -> int:
        if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "img_size"):
            isz = m.patch_embed.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        if hasattr(m, "img_size"):
            isz = m.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        dc = getattr(m, "default_cfg", {}) or {}
        ins = dc.get("input_size", None)
        if ins:
            if isinstance(ins, (tuple, list)) and len(ins) >= 2:
                return int(ins[1])
            return int(ins if isinstance(ins, (int, float)) else 224)
        return 518

    def _half_forward(self, x_half: torch.Tensor):
        tiles_small = self.tile_encoder(x_half, CFG.small_grid)
        tiles_big = self.tile_encoder(x_half, CFG.big_grid)
        t2, stage1_map = self.t2t(tiles_small, CFG.small_grid)
        fused = self.cross(t2, tiles_big)
        feat, feat_maps = self.pyramid(fused)
        feat_maps["stage1_map"] = stage1_map
        return feat, feat_maps

    def _merge_heads(self, f_l: torch.Tensor, f_r: torch.Tensor):
        g_l = torch.sigmoid(self.cross_gate_left(f_r))
        g_r = torch.sigmoid(self.cross_gate_right(f_l))
        f_l = f_l * g_l
        f_r = f_r * g_r
        f = torch.cat([f_l, f_r], dim=1)
        green_pos = self.softplus(self.head_green(f))
        clover_pos = self.softplus(self.head_clover(f))
        dead_pos = self.softplus(self.head_dead(f))
        gdm = green_pos + clover_pos
        total = gdm + dead_pos
        return total, gdm, green_pos, f

    def forward(self, *inputs, x_left=None, x_right=None, return_features: bool = False):
        if inputs:
            if len(inputs) == 1:
                first = inputs[0]
                if isinstance(first, (tuple, list)):
                    if len(first) >= 1: x_left = first[0]
                    if len(first) >= 2: x_right = first[1]
                else: x_left = first
            else: x_left = inputs[0]; x_right = inputs[1]
        if x_left is None or (isinstance(x_left, torch.Tensor) and x_left.shape[0] == 0):
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            zero = torch.zeros(0, 1, device=device, dtype=dtype)
            out = {"total": zero, "gdm": zero, "green": zero, "score_feat": torch.zeros(0, self.combined_dim, device=device, dtype=dtype)}
            if self.aux_head is not None:
                out["aux"] = torch.zeros(0, len(CFG.ALL_TARGET_COLS), device=device, dtype=dtype)
            if return_features: out["feature_maps"] = {}
            return out
        if x_right is None:
            if isinstance(x_left, torch.Tensor) and x_left.shape[1] % 2 == 0:
                x_left, x_right = torch.chunk(x_left, 2, dim=1)
            else: raise ValueError("缺少 x_right 输入。")
        feat_l, feats_l = self._half_forward(x_left)
        feat_r, feats_r = self._half_forward(x_right)
        total, gdm, green, f_concat = self._merge_heads(feat_l, feat_r)
        out = {"total": total, "gdm": gdm, "green": green, "score_feat": f_concat}
        if self.aux_head is not None:
            aux_tokens = torch.cat([feats_l["stage2_tokens"], feats_r["stage2_tokens"]], dim=1)
            aux_pred = self.softplus(self.aux_head(aux_tokens.mean(dim=1)))
            out["aux"] = aux_pred
        if return_features:
            out["feature_maps"] = {"stage1_left": feats_l.get("stage1_map"), "stage1_right": feats_r.get("stage1_map"), "stage3_left": feats_l.get("stage3_tokens"), "stage3_right": feats_r.get("stage3_tokens")}
        return out


# In[ ]:


from transformers import CLIPModel

@dataclass
class EVA_FINE_CFG:
    model_path: str = "/kaggle/input/eva02-large-patch14-clip-336-merged2b/pytorch/default/1"
    image_size: int = 224
    train_batch_size: int = 4
    valid_batch_size: int = 8
    epochs: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-3
    head_dropout: float = 0.2
    num_workers: int = 2
    use_amp: bool = True
    freeze_backbone: bool = True
    unfreeze_blocks: int = 2
    train_folds: List[int] = None
    output_dir: Path = Path("./eva_supervised_ckpts")
    force_retrain: bool = False
    max_grad_norm: float = 1.0
    smooth_l1_beta: float = 0.5

    def __post_init__(self):
        if self.train_folds is None:
            self.train_folds = list(range(5))
EVA_FINE_CFG = EVA_FINE_CFG()


def build_eva_transforms(train: bool = True, size: Optional[int] = None):
    size = size or EVA_FINE_CFG.image_size
    normalize = A.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        max_pixel_value=255.0,
    )
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
            A.HueSaturationValue(10, 15, 10, p=0.2),
            normalize,
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=size, width=size),
        normalize,
        ToTensorV2(),
    ])


class EVAImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms, has_targets: bool = True):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.has_targets = has_targets and all(col in self.df.columns for col in TARGET_NAMES)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((EVA_FINE_CFG.image_size, EVA_FINE_CFG.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            transformed = self.transforms(image=img)
            img_tensor = transformed['image']
        else:
            img_resized = cv2.resize(img, (EVA_FINE_CFG.image_size, EVA_FINE_CFG.image_size), interpolation=cv2.INTER_AREA)
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        if not torch.is_tensor(img_tensor):
            img_tensor = torch.tensor(img_tensor)
        img = img_tensor.contiguous().clone()
        item = {'pixel_values': img}
        if self.has_targets:
            targets = row[TARGET_NAMES].values.astype(np.float32)
            item['targets'] = torch.tensor(targets, dtype=torch.float32)
        return item


class EVARegressor(nn.Module):
    def __init__(self, cfg: EVA_FINE_CFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = CLIPModel.from_pretrained(cfg.model_path, local_files_only=True)
        vision_dim = self.backbone.config.vision_config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(vision_dim, len(TARGET_NAMES)),
        )
        self._configure_trainable_layers()

    def _configure_trainable_layers(self):
        if not self.cfg.freeze_backbone:
            return
        for param in self.backbone.parameters():
            param.requires_grad = False
        if self.cfg.unfreeze_blocks > 0:
            layers = self.backbone.vision_model.encoder.layers
            for block in layers[-self.cfg.unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        for param in self.backbone.visual_projection.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        vision_outputs = self.backbone.vision_model(pixel_values=pixel_values, output_attentions=False, output_hidden_states=False)
        if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
            feats = vision_outputs.pooler_output
        else:
            feats = vision_outputs[1]
        return self.head(feats)


def create_eva_loader(df: pd.DataFrame, train: bool, has_targets: bool):
    transforms = build_eva_transforms(train=train)
    dataset = EVAImageDataset(df, transforms, has_targets=has_targets)
    batch_size = EVA_FINE_CFG.train_batch_size if train else EVA_FINE_CFG.valid_batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and has_targets,
        num_workers=EVA_FINE_CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def evaluate_eva(model, loader, device, criterion=None):
    model.eval()
    preds, loss_sum, count = [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch['pixel_values'].to(device)
            outputs = model(images)
            preds.append(outputs.detach().cpu().numpy())
            targets = batch.get('targets')
            if criterion is not None and targets is not None:
                targets = targets.to(device)
                batch_loss = criterion(outputs, targets).item()
                loss_sum += batch_loss * images.size(0)
                count += images.size(0)
    stacked = np.concatenate(preds, axis=0)
    loss = loss_sum / max(1, count) if count else 0.0
    return loss, stacked


def train_eva_models(train_df: pd.DataFrame):
    cfg = EVA_FINE_CFG
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df = train_df.reset_index(drop=True)
    oof = np.zeros((len(train_df), len(TARGET_NAMES)), dtype=np.float32)
    fold_scores = {}
    criterion = nn.SmoothL1Loss(beta=cfg.smooth_l1_beta)
    for fold in cfg.train_folds:
        valid_mask = train_df['fold'] == fold
        if valid_mask.sum() == 0:
            continue
        train_split = train_df[~valid_mask].reset_index(drop=True)
        valid_split = train_df[valid_mask].reset_index(drop=True)
        val_indices = np.where(valid_mask.values)[0]
        ckpt_path = cfg.output_dir / f"eva_fold_{fold}.pt"
        need_train = cfg.force_retrain or not ckpt_path.exists()
        valid_loader = create_eva_loader(valid_split, train=False, has_targets=True)
        model = EVARegressor(cfg).to(device)
        if need_train:
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
            scaler = GradScaler(enabled=cfg.use_amp and device.type == 'cuda')
            best_metric = -np.inf
            best_state = None
            best_preds = None
            train_loader = create_eva_loader(train_split, train=True, has_targets=True)
            for epoch in range(cfg.epochs):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    images = batch['pixel_values'].to(device)
                    targets = batch['targets'].to(device)
                    with autocast(enabled=cfg.use_amp and device.type == 'cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    if cfg.max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                val_loss, val_preds = evaluate_eva(model, valid_loader, device, criterion)
                metric = competition_metric(valid_split[TARGET_NAMES].values, val_preds)
                if metric > best_metric:
                    best_metric = metric
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_preds = val_preds.copy()
            torch.save({'model_state': best_state, 'metric': best_metric}, ckpt_path)
            fold_scores[fold] = best_metric
            oof[val_indices] = best_preds
            print(f"[EVA-FT] Fold {fold}: CV {best_metric:.6f}")
        else:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model_state'], strict=False)
            _, val_preds = evaluate_eva(model, valid_loader, device, criterion)
            metric = competition_metric(valid_split[TARGET_NAMES].values, val_preds)
            fold_scores[fold] = metric
            oof[val_indices] = val_preds
            print(f"[EVA-FT] Fold {fold}: CV {metric:.6f} (loaded)")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    overall = competition_metric(train_df[TARGET_NAMES].values, oof)
    print(f"[EVA-FT] Overall CV: {overall:.6f}")
    return oof, overall


def predict_eva_test(test_df: pd.DataFrame):
    cfg = EVA_FINE_CFG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_df = test_df.reset_index(drop=True)
    loader = create_eva_loader(test_df, train=False, has_targets=False)
    preds = []
    for fold in cfg.train_folds:
        ckpt_path = cfg.output_dir / f"eva_fold_{fold}.pt"
        if not ckpt_path.exists():
            continue
        model = EVARegressor(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state'], strict=False)
        _, fold_pred = evaluate_eva(model, loader, device, criterion=None)
        preds.append(fold_pred)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not preds:
        raise RuntimeError("No EVA checkpoints were found. Run training before inference.")
    return np.mean(preds, axis=0)


def run_eva_supervised_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    oof, overall = train_eva_models(train_df)
    preds = predict_eva_test(test_df)
    return preds, overall


# In[6]:


class INF_CFG:
    BASE_PATH = "/kaggle/input/csiro-biomass"
    TEST_CSV = os.path.join(BASE_PATH, "test.csv")
    TEST_IMAGE_DIR = os.path.join(BASE_PATH, "test")
    EXPERIMENT_DIR = "/kaggle/input/csiro/pytorch/default/12"
    CKPT_PATTERN_FOLD_X = os.path.join(EXPERIMENT_DIR, "fold_{fold}", "checkpoints", "best_wr2.pt")
    CKPT_PATTERN_FOLDX = os.path.join(EXPERIMENT_DIR, "fold{fold}", "checkpoints", "best_wr2.pt")
    N_FOLDS = 5
    SUBMISSION_FILE = "submission3.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    MIXED_PRECISION = True
    USE_TTA = True
    TTA_TRANSFORMS = ["original", "hflip", "vflip"]
    TTA_STACK = True
    TTA_STACK_WEIGHT = 0.5
    CKPT_SEARCH_PATTERNS = ("best*.pt", "model*.pt", "*.pth")
    EXTRA_CKPT_DIRS: List[str] = []
    MAX_MODELS = 8
    PER_FOLD_FILENAMES = ("best_wr2.pt", "best_loss.pt", "last.pt")
    ALL_TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


class TestBiomassDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, image_dir: str):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_dir = image_dir
        self.paths = self.df["image_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = os.path.basename(self.paths[idx])
        full_path = os.path.join(self.image_dir, filename)
        img = cv2.imread(full_path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        left_t = self.transform(image=left)["image"]
        right_t = self.transform(image=right)["image"]
        return left_t, right_t


def strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    return state


def load_model_from_checkpoint(ckpt_path: str) -> nn.Module:
    state = load_checkpoint(ckpt_path)
    cfg_dict = state.get("cfg", {})
    update_cfg_from_checkpoint(cfg_dict)
    dropout = cfg_dict.get("dropout", CFG.dropout)
    hidden_ratio = cfg_dict.get("hidden_ratio", CFG.hidden_ratio)
    model = CrossPVT_T2T_MambaDINO(dropout=dropout, hidden_ratio=hidden_ratio)
    model_state = state.get("model_state", state)
    model_state = strip_module_prefix(model_state)
    model.load_state_dict(model_state, strict=False)
    model = maybe_parallelize(model, INF_CFG.DEVICE, min_batch=INF_CFG.BATCH_SIZE)
    model.eval()
    return model


def pack5_targets(total: torch.Tensor, gdm: torch.Tensor, green: torch.Tensor) -> torch.Tensor:
    clover = gdm - green
    dead = total - gdm
    return torch.cat([green, dead, clover, gdm, total], dim=1)


@torch.no_grad()
def predict_one_view(models: List[nn.Module], loader: DataLoader) -> np.ndarray:
    preds_list = []
    amp_dtype = "cuda" if INF_CFG.DEVICE.type == "cuda" else "cpu"
    for xl, xr in tqdm(loader, desc="  Predicting", leave=False):
        xl = xl.to(INF_CFG.DEVICE, non_blocking=True)
        xr = xr.to(INF_CFG.DEVICE, non_blocking=True)
        x_cat = torch.cat([xl, xr], dim=1)
        per_model_preds = []
        with torch.amp.autocast(amp_dtype, enabled=INF_CFG.MIXED_PRECISION):
            for model in models:
                out = model(x_cat, return_features=False)
                total = out["total"]
                gdm = out["gdm"]
                green = out["green"]
                five = torch.clamp(pack5_targets(total, gdm, green), min=0.0)
                per_model_preds.append(five.float().cpu())
        stacked = torch.mean(torch.stack(per_model_preds, dim=0), dim=0)
        preds_list.append(stacked.numpy())
    return np.concatenate(preds_list, axis=0)


def _gather_configured_ckpts():
    ckpts = []
    for fold in range(INF_CFG.N_FOLDS):
        candidate_paths = []
        for pattern in (INF_CFG.CKPT_PATTERN_FOLD_X, INF_CFG.CKPT_PATTERN_FOLDX):
            base = pattern.format(fold=fold)
            candidate_paths.append(base)
            base_dir = os.path.dirname(base)
            if os.path.isdir(base_dir):
                for fname in INF_CFG.PER_FOLD_FILENAMES:
                    candidate_paths.append(os.path.join(base_dir, fname))
        for path in candidate_paths:
            if os.path.exists(path):
                ckpts.append(path)
                break
    return ckpts


def _discover_additional_ckpts():
    search_dirs = [d for d in [INF_CFG.EXPERIMENT_DIR, *(INF_CFG.EXTRA_CKPT_DIRS or [])] if d and os.path.isdir(d)]
    if not search_dirs:
        default_root = "/kaggle/input"
        if os.path.isdir(default_root):
            search_dirs.append(default_root)
    found = []
    for base in search_dirs:
        for pattern in INF_CFG.CKPT_SEARCH_PATTERNS:
            glob_pattern = os.path.join(base, "**", pattern)
            for path in glob(glob_pattern, recursive=True):
                if os.path.isfile(path):
                    found.append(path)
    unique = []
    seen = set()
    for path in sorted(found):
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def collect_checkpoint_paths():
    ckpts = _gather_configured_ckpts()
    if ckpts:
        return ckpts
    return _discover_additional_ckpts()


def run_inference(test_df: pd.DataFrame, image_dir: str) -> np.ndarray:
    models = []
    input_res = None
    checkpoint_paths = collect_checkpoint_paths()
    if not checkpoint_paths:
        raise RuntimeError("No checkpoints were found for the configured folds.")
    usable = checkpoint_paths[:INF_CFG.MAX_MODELS]
    print(f"[DINO] Using {len(usable)} checkpoint(s) from: {usable}")
    for ckpt_path in usable:
        model = load_model_from_checkpoint(ckpt_path)
        if model is None:
            continue
        models.append(model)
        if input_res is None:
            input_res = getattr(unwrap_model(model), "input_res", 518)
    if not models:
        raise RuntimeError("Unable to load any DINO checkpoints.")
    transforms = get_tta_transforms(input_res)
    if INF_CFG.USE_TTA and len(transforms) > 1:
        final_pred = apply_tta_with_stacking(models, test_df, image_dir, transforms)
    else:
        single = transforms[0]
        ds = TestBiomassDataset(test_df, single, image_dir)
        dl = DataLoader(ds, batch_size=INF_CFG.BATCH_SIZE, shuffle=False, num_workers=INF_CFG.NUM_WORKERS, pin_memory=True)
        final_pred = predict_one_view(models, dl)
    return final_pred


def create_submission(final_pred: np.ndarray, test_long: pd.DataFrame, test_unique: pd.DataFrame) -> pd.DataFrame:
    def clean(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(0, x)
    green, dead, clover, gdm, total = map(clean, [final_pred[:, 0], final_pred[:, 1], final_pred[:, 2], final_pred[:, 3], final_pred[:, 4]])
    wide = pd.DataFrame({
        "image_path": test_unique["image_path"],
        "Dry_Green_g": green,
        "Dry_Dead_g": dead,
        "Dry_Clover_g": clover,
        "GDM_g": gdm,
        "Dry_Total_g": total,
    })
    wide = post_process_biomass(wide)
    long_preds = wide.melt(id_vars=["image_path"], value_vars=INF_CFG.ALL_TARGET_COLS, var_name="target_name", value_name="target")
    sub = pd.merge(
        test_long[["sample_id", "image_path", "target_name"]],
        long_preds,
        on=["image_path", "target_name"],
        how="left",
    )[["sample_id", "target"]]
    sub["target"] = np.nan_to_num(sub["target"], nan=0.0, posinf=0.0, neginf=0.0)
    sub.to_csv(INF_CFG.SUBMISSION_FILE, index=False)
    return sub


def parse_args():
    parser = argparse.ArgumentParser(description="CSIRO CrossPVT DINO inference")
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--test-image-dir", type=str, default=None)
    parser.add_argument("--experiment-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--ckpt-root", action="append", default=None)
    args, _ = parser.parse_known_args()
    return args


def run_dino_inference():
    args = parse_args()
    if args.test_csv:
        INF_CFG.TEST_CSV = args.test_csv
    if args.test_image_dir:
        INF_CFG.TEST_IMAGE_DIR = args.test_image_dir
    if args.experiment_dir:
        INF_CFG.EXPERIMENT_DIR = args.experiment_dir
        INF_CFG.CKPT_PATTERN_FOLD_X = os.path.join(INF_CFG.EXPERIMENT_DIR, "fold_{fold}", "checkpoints", "best_wr2.pt")
        INF_CFG.CKPT_PATTERN_FOLDX = os.path.join(INF_CFG.EXPERIMENT_DIR, "fold{fold}", "checkpoints", "best_wr2.pt")
    if args.output:
        INF_CFG.SUBMISSION_FILE = args.output
    if args.batch_size:
        INF_CFG.BATCH_SIZE = args.batch_size
    if args.no_tta:
        INF_CFG.USE_TTA = False
    if args.ckpt_root:
        INF_CFG.EXTRA_CKPT_DIRS = [p for p in args.ckpt_root if p]
    test_long = pd.read_csv(INF_CFG.TEST_CSV)
    test_unique = test_long.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    final_pred = run_inference(test_unique, INF_CFG.TEST_IMAGE_DIR)
    submission = create_submission(final_pred, test_long, test_unique)
    print(f"[DINO] Saved {len(submission)} rows to {INF_CFG.SUBMISSION_FILE}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return submission


def get_input_size(model):
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size"):
        size = model.patch_embed.img_size
        return int(size if isinstance(size, (int, float)) else size[0])
    if hasattr(model, "img_size"):
        size = model.img_size
        return int(size if isinstance(size, (int, float)) else size[0])
    cfg = getattr(model, "default_cfg", {}) or {}
    ins = cfg.get("input_size", None)
    if ins:
        if isinstance(ins, (tuple, list)) and len(ins) >= 2:
            return int(ins[1])
        return int(ins if isinstance(ins, (int, float)) else 224)
    arch = cfg.get("architecture", "") or str(type(model))
    return 518 if "dinov2" in arch.lower() or "dinov3" in arch.lower() else 224


def build_backbone(name):
    model = timm.create_model(name, pretrained=False, num_classes=0)
    features = model.num_features
    input_size = get_input_size(model)
    return model, features, input_size


class BaseDINO(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone, feat_dim, input_size = build_backbone(backbone_name)
        self.input_size = int(input_size)
        self.feat_dim = feat_dim
        self.combined_dim = feat_dim * 2
        hidden_size = max(8, int(self.combined_dim * 0.25))
        def make_head():
            return nn.Sequential(
                nn.Linear(self.combined_dim, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 1),
            )
        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()
        self.softplus = nn.Softplus(beta=1.0)

    def merge_features(self, left_feat, right_feat):
        combined = torch.cat([left_feat, right_feat], dim=1)
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        gdm = green + clover
        total = gdm + dead
        return total, gdm, green


class TiledFiLMDINO(BaseDINO):
    def __init__(self, backbone_name):
        super().__init__(backbone_name)
        self.grid = (2, 2)

        class FiLM(nn.Module):
            def __init__(self, feat_dim):
                super().__init__()
                hidden = max(32, feat_dim // 2)
                self.mlp = nn.Sequential(
                    nn.Linear(feat_dim, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, feat_dim * 2),
                )

            def forward(self, context):
                gamma_beta = self.mlp(context)
                return torch.chunk(gamma_beta, 2, dim=1)

        self.film_left = FiLM(self.feat_dim)
        self.film_right = FiLM(self.feat_dim)

    def extract_tile_features(self, x):
        B, C, H, W = x.shape
        rows, cols = self.grid

        def split_dimension(length, parts):
            step = length // parts
            segments = []
            start = 0
            for _ in range(parts - 1):
                segments.append((start, start + step))
                start += step
            segments.append((start, length))
            return segments

        row_segments = split_dimension(H, rows)
        col_segments = split_dimension(W, cols)
        features = []
        for (rs, re) in row_segments:
            for (cs, ce) in col_segments:
                tile = x[:, :, rs:re, cs:ce]
                if tile.shape[-2:] != (self.input_size, self.input_size):
                    tile = F.interpolate(tile, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
                feat = self.backbone(tile)
                features.append(feat)
        return torch.stack(features, dim=0).permute(1, 0, 2)

    def process_stream(self, x, film_layer):
        tiles = self.extract_tile_features(x)
        context = tiles.mean(dim=1)
        gamma, beta = film_layer(context)
        modulated = tiles * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return modulated.mean(dim=1)

    def forward(self, left_img, right_img):
        left_feat = self.process_stream(left_img, self.film_left)
        right_feat = self.process_stream(right_img, self.film_right)
        return self.merge_features(left_feat, right_feat)


def clean_state_dict(state_dict):
    if not state_dict:
        return state_dict
    cleaned_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("student."):
            k = k[8:]
        skip_prefixes = ("txt_enc.", "img_proj.", "txt_film", "teacher.", "momentum_teacher.")
        if any(k.startswith(prefix) for prefix in skip_prefixes):
            continue
        cleaned_dict[k] = v
    return cleaned_dict


def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        raw_state = torch.load(checkpoint_path, map_location=INF_CFG.DEVICE, weights_only=False)
    except Exception:
        return None
    if isinstance(raw_state, dict):
        if "state_dict" in raw_state:
            state_dict = raw_state["state_dict"]
        elif "model" in raw_state:
            state_dict = raw_state["model"]
        else:
            state_dict = raw_state
    else:
        state_dict = raw_state
    state_dict = clean_state_dict(state_dict)
    if not state_dict:
        return None
    backbones = ["vit_base_patch14_reg4_dinov2", "vit_base_patch14_reg4_dinov3", "vit_base_patch14_dinov3"]
    for backbone in backbones:
        try:
            model = TiledFiLMDINO(backbone)
            result = model.load_state_dict(state_dict, strict=False)
            missing = [k for k in result.missing_keys if not k.startswith("backbone.pos_embed")]
            if len(missing) == 0:
                model = maybe_parallelize(model, INF_CFG.DEVICE, min_batch=4)
                model.eval()
                return model
        except Exception:
            continue
    return None


def get_tta_transforms_mvp(img_size):
    norm = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
    return [
        A.Compose([A.Resize(img_size, img_size), *norm]),
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(img_size, img_size), *norm]),
        A.Compose([A.VerticalFlip(p=1.0), A.Resize(img_size, img_size), *norm]),
        A.Compose([A.RandomRotate90(p=1.0), A.Resize(img_size, img_size), *norm]),
    ]


class BiomassDataset(Dataset):
    def __init__(self, df, transform, img_dir):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir
        self.paths = self.df["image_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = os.path.basename(self.paths[idx])
        full_path = os.path.join(self.img_dir, filename)
        img = cv2.imread(full_path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        mid = w // 2
        left_img = img[:, :mid]
        right_img = img[:, mid:]
        left_tensor = self.transform(image=left_img)["image"]
        right_tensor = self.transform(image=right_img)["image"]
        return left_tensor, right_tensor


@torch.no_grad()
def predict_one_view_mvp(models, loader):
    preds = []
    use_amp = INF_CFG.DEVICE.type == "cuda"
    for left_imgs, right_imgs in tqdm(loader, desc="Infer", leave=False):
        left_imgs = left_imgs.to(INF_CFG.DEVICE, non_blocking=True)
        right_imgs = right_imgs.to(INF_CFG.DEVICE, non_blocking=True)
        batch_preds = []
        with torch.amp.autocast("cuda", enabled=use_amp):
            for model in models:
                total, gdm, green = model(left_imgs, right_imgs)
                dead = torch.clamp(total - gdm, min=0.0)
                clover = torch.clamp(gdm - green, min=0.0)
                pred = torch.cat([green, dead, clover, gdm, total], dim=1)
                batch_preds.append(pred.clamp(0.05, 400.0).cpu())
        preds.append(torch.stack(batch_preds).mean(dim=0).numpy())
    return np.concatenate(preds)


def run_inference_for_ckpts(checkpoint_paths, df, img_dir):
    models = []
    for ckpt_path in checkpoint_paths:
        model = load_model(ckpt_path)
        if model is not None:
            models.append(model)
    if not models:
        raise ValueError(f"No models loaded from {checkpoint_paths}")
    print(f"[MVP] Loaded {len(models)} checkpoint(s) from {checkpoint_paths}")
    input_size = getattr(unwrap_model(models[0]), "input_size", 518)
    tta_preds = []
    for transform in get_tta_transforms_mvp(input_size):
        ds = BiomassDataset(df, transform, img_dir)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
        tta_preds.append(predict_one_view_mvp(models, dl))
    return np.mean(tta_preds, axis=0)


def create_submission_mvp(final_preds, test_df, unique_df):
    cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    wide = pd.DataFrame({"image_path": unique_df["image_path"]})
    for i, col in enumerate(cols):
        wide[col] = np.clip(final_preds[:, i], 0.05, 400.0)
    wide["GDM_g"] = wide["Dry_Green_g"] + wide["Dry_Clover_g"]
    wide["Dry_Total_g"] = wide["GDM_g"] + wide["Dry_Dead_g"]
    wide = post_process_biomass(wide)
    long_df = wide.melt(id_vars="image_path", value_vars=cols, var_name="target_name", value_name="target")
    sub = test_df[["sample_id", "image_path", "target_name"]].merge(long_df, on=["image_path", "target_name"], how="left")[["sample_id", "target"]]
    sub.to_csv("submission2.csv", index=False)
    return sub


# In[7]:


def _build_single_tta(name: str, img_size: int, base_ops):
    name = (name or "identity").strip().lower()
    ops = []
    if name in ("identity", "original", "base"):
        pass
    elif name in ("hflip", "horizontal_flip"):
        ops.append(A.HorizontalFlip(p=1.0))
    elif name in ("vflip", "vertical_flip"):
        ops.append(A.VerticalFlip(p=1.0))
    elif name == "hvflip":
        ops.extend([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])
    elif name == "transpose":
        ops.append(A.Transpose(p=1.0))
    elif name in ("rot90", "rotate90"):
        ops.append(A.Rotate(limit=(90, 90), border_mode=cv2.BORDER_REFLECT101, p=1.0))
    elif name in ("rot180", "rotate180"):
        ops.append(A.Rotate(limit=(180, 180), border_mode=cv2.BORDER_REFLECT101, p=1.0))
    elif name in ("rot270", "rotate270"):
        ops.append(A.Rotate(limit=(270, 270), border_mode=cv2.BORDER_REFLECT101, p=1.0))
    elif name == "brightness":
        ops.append(A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=1.0))
    elif name == "blur":
        ops.append(A.GaussianBlur(blur_limit=(3, 5), p=1.0))
    else:
        return None
    ops.append(A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA))
    ops.extend(base_ops)
    return A.Compose(ops)

def get_tta_transforms(img_size: int, variant_names=None):
    base_ops = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
    names = variant_names or INF_CFG.TTA_TRANSFORMS
    transforms = []
    for name in names:
        aug = _build_single_tta(name, img_size, base_ops)
        if aug is not None:
            transforms.append(aug)
    if not transforms:
        transforms.append(A.Compose([A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA), *base_ops]))
    return transforms



# In[ ]:


def apply_tta_with_stacking(models, base_df, image_dir, transforms):
    preds_per_aug = []
    for transform in transforms:
        ds = TestBiomassDataset(base_df, transform, image_dir)
        dl = DataLoader(ds, batch_size=INF_CFG.BATCH_SIZE, shuffle=False, num_workers=INF_CFG.NUM_WORKERS, pin_memory=True)
        preds_per_aug.append(predict_one_view(models, dl))
    stacked = np.stack(preds_per_aug, axis=0)  # (T, N, C)
    mean_pred = stacked.mean(axis=0)
    if INF_CFG.TTA_STACK and stacked.shape[0] > 1:
        from sklearn.linear_model import Ridge
        n_transforms, n_samples, n_targets = stacked.shape
        aug_features = stacked.transpose(1, 0, 2).reshape(n_samples, n_transforms * n_targets)
        meta = Ridge(alpha=1e-3)
        meta.fit(aug_features, mean_pred)
        meta_pred = meta.predict(aug_features).reshape(n_samples, n_targets)
        return INF_CFG.TTA_STACK_WEIGHT * meta_pred + (1 - INF_CFG.TTA_STACK_WEIGHT) * mean_pred
    return mean_pred


# In[8]:


def get_tta_transforms_mvp(img_size, variant_names=None):
    return get_tta_transforms(img_size, variant_names)


# In[9]:


def run_mvp_inference():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = "/kaggle/input/csiro-biomass/test.csv"
    test_img_dir = "/kaggle/input/csiro-biomass/test"
    model_dir = "/kaggle/input/csiro-mvp-models"
    model_paths = [os.path.join(model_dir, f"model{i}.pth") for i in range(1, 11)]
    existing_models = [path for path in model_paths if os.path.exists(path)]
    CKPTS_A = existing_models[:5]; CKPTS_B = existing_models[5:]
    W_A, W_B = 0.95, 0.075
    test_df = pd.read_csv(test_csv)
    unique_df = test_df.drop_duplicates("image_path").reset_index(drop=True)
    pred_a = run_inference_for_ckpts(CKPTS_A, unique_df, test_img_dir)
    pred_b = run_inference_for_ckpts(CKPTS_B, unique_df, test_img_dir)
    final_preds = W_A * pred_a + W_B * pred_b
    submission = create_submission_mvp(final_preds, test_df, unique_df)
    print(f"[MVP] Saved {len(submission)} rows to submission2.csv")
    return submission


# In[ ]:


def strip_embedding_columns(df, prefixes=("emb", "dino_emb")):
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    if not cols:
        return df
    return df.drop(columns=cols, errors='ignore')


def rename_embedding_columns(df, base_prefix="dino_emb"):
    emb_cols = [c for c in df.columns if c.startswith(base_prefix) or c.startswith('emb')]
    if not emb_cols:
        return df, []
    rename_map = {}
    for col in emb_cols:
        cleaned = col.replace('_x','').replace('_y','')
        cleaned = cleaned.replace('emb', base_prefix, 1) if cleaned.startswith('emb') else cleaned
        rename_map[col] = cleaned
    df = df.rename(columns=rename_map)
    final_cols = [c for c in df.columns if c.startswith(base_prefix)]
    return df, final_cols


def preview_embedding_columns(df, label="DF"):
    emb_cols = [c for c in df.columns if c.startswith('emb') or c.startswith('dino_emb')]
    print(f"[{label}] embedding columns: {len(emb_cols)}")
    if emb_cols:
        print(emb_cols[:20])


# In[ ]:


# Run this cell after assigning EMBED_DEBUG_DF to any dataframe to preview its embedding columns.
EMBED_DEBUG_DF = globals().get('EMBED_DEBUG_DF')
if EMBED_DEBUG_DF is not None:
    preview_embedding_columns(EMBED_DEBUG_DF, 'EMBED_DEBUG_DF')
else:
    print('[Embedding Debug] Set EMBED_DEBUG_DF to a dataframe (e.g., dino_train_df) and re-run this cell.')


# In[10]:


def main():
    seeding(42)

    print("SigLIP/Ensemble Model ===")
    test_df = pd.read_csv(cfg.DATA_PATH/'test.csv')
    test_df = pivot_table(df=test_df)
    test_df['image_path'] = test_df['image_path'].apply(lambda p: str(cfg.DATA_PATH / p))
    train_df = pd.read_csv("/kaggle/input/csiro-datasplit/csiro_data_split.csv")

    base_train_df = train_df.copy()
    siglip_pred = None
    pillar_paths = {}

    sem_train_full = None
    sem_test_full = None

    if cfg.run_siglip_stack:
        train_siglip_df = assign_folds(base_train_df.copy(), cfg.stack_kfolds, cfg.seed)
        train_siglip_df_legacy = assign_folds(base_train_df.copy(), cfg.legacy_kfolds, cfg.seed)
        siglip_path = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1"
        base_test_siglip_df = compute_embeddings(model_path=siglip_path, df=test_df, patch_size=520)

        train_siglip_df, test_siglip_df = append_cluster_features(
            train_siglip_df,
            base_test_siglip_df.copy(),
            embed_prefix='emb',
            cluster_sizes=cfg.siglip_cluster_sizes,
            random_state=cfg.seed,
        )
        train_siglip_df_legacy, test_siglip_df_legacy = append_cluster_features(
            train_siglip_df_legacy,
            base_test_siglip_df.copy(),
            embed_prefix='emb',
            cluster_sizes=cfg.siglip_cluster_sizes,
            random_state=cfg.seed,
        )

        flush()

        sem_train_full = None
        sem_test_full = None
        if cfg.use_semantic_features:
            X_all_emb = np.vstack([train_siglip_df.filter(like="emb").values, test_siglip_df.filter(like="emb").values])
            try:
                all_semantic_scores = generate_semantic_features(X_all_emb, model_path=siglip_path)
                n_train = len(train_siglip_df)
                sem_train_full = all_semantic_scores[:n_train]
                sem_test_full = all_semantic_scores[n_train:]
            except Exception as e:
                sem_train_full = None
                sem_test_full = None
        else:
            print("[FastMode] Skipping semantic feature generation for SigLIP pillar.")

        if cfg.fast_mode:
            feat_engine = SupervisedEmbeddingEngine(n_pca=0.95, n_pls=5, n_gmm=3)
        else:
            feat_engine = SupervisedEmbeddingEngine(n_pca=0.88, n_pls=10, n_gmm=5)

        ensemble_preds = []
        ensemble_weights = []
        stack_train_features = []
        stack_test_features = []
        model_tags = []

        def register_model(name, oof_preds, test_preds, processed_cv):
            weight = max(processed_cv, 1e-6)
            ensemble_preds.append(test_preds)
            ensemble_weights.append(weight)
            stack_train_features.append(oof_preds)
            stack_test_features.append(test_preds)
            model_tags.append(name)
            print(f"[Stack] {name}: processed CV {processed_cv:.6f}")
        def run_siglip_model(name, estimator):
            if cfg.fast_mode and cfg.siglip_fast_models and name not in cfg.siglip_fast_models:
                print(f"[FastMode] Skipping SigLIP model {name}")
                return
            oof_preds, pred_test = cross_validate(
                estimator,
                train_siglip_df,
                test_siglip_df,
                feature_engine=feat_engine,
                semantic_train=sem_train_full,
                semantic_test=sem_test_full,
                n_splits_override=cfg.stack_kfolds,
            )
            _, proc_score = compare_results(oof_preds, train_siglip_df, return_scores=True)
            register_model(name, oof_preds, pred_test, proc_score)

        run_siglip_model("cat_seed0", CatBoostRegressor(verbose=0, random_seed=cfg.seed))
        run_siglip_model("svr_seed0", SVR(kernel='rbf', C=0.7, epsilon=0.02, tol=1e-4, max_iter=100000, cache_size=2000))
        run_siglip_model("xgb_seed0", XGBRegressor(n_estimators=700, learning_rate=0.02, max_depth=7, subsample=0.9, colsample_bytree=0.95, reg_lambda=1.0, reg_alpha=0.0, tree_method='hist', random_state=cfg.seed))
        run_siglip_model("bayes_ridge", BayesianRidge())
        run_siglip_model("elastic_net", ElasticNet(alpha=0.08, l1_ratio=0.4, max_iter=2000))
        run_siglip_model("extra_trees", ExtraTreesRegressor(n_estimators=600, max_depth=None, min_samples_split=4, min_samples_leaf=2, random_state=cfg.seed))
        run_siglip_model("xgb_seed1", XGBRegressor(n_estimators=700, learning_rate=0.02, max_depth=7, subsample=0.9, colsample_bytree=0.95, reg_lambda=1.0, reg_alpha=0.0, tree_method='hist', random_state=cfg.seed + 1881))
        run_siglip_model("svr_seed1", SVR(kernel='rbf', C=0.7, epsilon=0.02, tol=1e-4, max_iter=100000, cache_size=2000))
        run_siglip_model("cat_seed1", CatBoostRegressor(verbose=0, random_seed=cfg.seed + 404))
        run_siglip_model("ridge_seed0", Ridge(alpha=0.5))
        run_siglip_model("ridge_seed1", Ridge(alpha=0.4))
        run_siglip_model("elastic_net_seed1", ElasticNet(alpha=0.06, l1_ratio=0.35, max_iter=2500))
        weights = np.array(ensemble_weights, dtype=np.float64)
        weights /= weights.sum()
        pred_stack = np.stack(ensemble_preds, axis=0)
        weighted_pred = np.tensordot(weights, pred_stack, axes=(0, 0))
        print(f"Ensemble weights: {[round(w, 4) for w in weights.tolist()]}")

        if stack_train_features:
            stack_train_matrix = np.concatenate(stack_train_features, axis=1)
            stack_test_matrix = np.concatenate(stack_test_features, axis=1)
            stacker = Ridge(alpha=0.3)
            stacker.fit(stack_train_matrix, train_siglip_df[TARGET_NAMES].values)
            stack_train_pred = stacker.predict(stack_train_matrix)
            stack_cv = competition_metric(train_siglip_df[TARGET_NAMES].values, stack_train_pred)
            print(f"[Stack] Ridge layer CV: {stack_cv:.6f}")
            stack_test_pred = stacker.predict(stack_test_matrix)
            pred_advanced = 0.6 * stack_test_pred + 0.4 * weighted_pred
        else:
            pred_advanced = weighted_pred

        mix = cfg.siglip_hybrid_weight
        if mix < 1.0 and cfg.enable_legacy_siglip:
            legacy_models = [
                ("gb", GradientBoostingRegressor()),
                ("hgb", HistGradientBoostingRegressor()),
                ("cat", CatBoostRegressor(verbose=0)),
                ("lgbm", LGBMRegressor(verbose=-1)),
            ]
            legacy_preds = []
            for name, model in legacy_models:
                oof_leg, pred_leg = cross_validate(
                    model,
                    train_siglip_df_legacy,
                    test_siglip_df_legacy,
                    feature_engine=feat_engine,
                    semantic_train=sem_train_full,
                    semantic_test=sem_test_full,
                    n_splits_override=cfg.legacy_kfolds,
                )
                _, proc_leg = compare_results(oof_leg, train_siglip_df_legacy, return_scores=True)
                print(f"[Legacy] {name}: processed CV {proc_leg:.6f}")
                legacy_preds.append(pred_leg)
            pred_legacy = np.mean(legacy_preds, axis=0)
            pred_test = mix * pred_advanced + (1 - mix) * pred_legacy
            print(f"[SigLIP] Hybrid advanced/legacy weight: {mix:.2f}/{1 - mix:.2f}")
        else:
            pred_test = pred_advanced

        siglip_pred = pred_test
        test_df[TARGET_NAMES] = pred_test
        test_df = post_process_biomass(test_df)
        sub_df = melt_table(test_df)
        sub_df[['sample_id', 'target']].to_csv("submission1.csv", index=False)
        pillar_paths["siglip"] = Path("submission1.csv")
    else:
        print("[Config] Skipping SigLIP stack.")
        pillar_paths["siglip"] = None


    dinov2_pred = None
    if cfg.run_dinov2:
        print("Dinov2-L Pillar ===")
        dinov2_path = "/kaggle/input/dinov2/pytorch/large/1"
        dino_train_df = assign_folds(base_train_df.copy(), cfg.dino_kfolds, cfg.seed)
        preview_embedding_columns(dino_train_df, "Dinov2 train (pre-strip)")
        dino_train_df = strip_embedding_columns(dino_train_df, prefixes=("emb", "dino_emb"))
        preview_embedding_columns(dino_train_df, "Dinov2 train (post-strip)")
        dino_train_df["image_path"] = dino_train_df["image_path"].apply(ensure_abs_path)
        dino_test_meta = pd.DataFrame({"image_path": test_df["image_path"].apply(ensure_abs_path)})
        dino_test_meta = strip_embedding_columns(dino_test_meta, prefixes=("emb", "dino_emb"))
        dino_train_df = compute_embeddings(dinov2_path, dino_train_df, patch_size=518)
        dino_test_df = compute_embeddings(dinov2_path, dino_test_meta.copy(), patch_size=518)
        dino_train_df, dino_emb_cols = rename_embedding_columns(dino_train_df, base_prefix="dino_emb")
        dino_test_df, _ = rename_embedding_columns(dino_test_df, base_prefix="dino_emb")
        preview_embedding_columns(dino_train_df, "Dinov2 train (post-rename)")
        preview_embedding_columns(dino_test_df, "Dinov2 test (post-rename)")
        dino_train_df, dino_test_df = append_cluster_features(
            dino_train_df,
            dino_test_df,
            embed_prefix="dino_emb",
            cluster_sizes=(16, 32, 48, 64, 80),
            random_state=cfg.seed,
        )
        feat_engine_dino = SupervisedEmbeddingEngine(n_pca=0.995, n_pls=10, n_gmm=4)

        dino_preds = []
        dino_oof = []
        dino_weights = []
        dino_tags = []

        def register_dino(name, oof_preds, test_preds, processed_cv):
            dino_preds.append(test_preds)
            dino_oof.append(oof_preds)
            dino_weights.append(max(processed_cv, 1e-6))
            dino_tags.append(name)
            print(f"[Dinov2] {name}: processed CV {processed_cv:.6f}")

        oof_dino_cat, pred_dino_cat = cross_validate(
            CatBoostRegressor(verbose=0, random_seed=cfg.seed, iterations=900, depth=7, learning_rate=0.028),
            dino_train_df,
            dino_test_df,
            feature_engine=feat_engine_dino,
            semantic_train=sem_train_full,
            semantic_test=sem_test_full,
            n_splits_override=cfg.dino_kfolds,
        )
        _, proc_dino_cat = compare_results(oof_dino_cat, dino_train_df, return_scores=True)
        register_dino("dinov2_cat", oof_dino_cat, pred_dino_cat, proc_dino_cat)

        oof_dino_xgb, pred_dino_xgb = cross_validate(
            XGBRegressor(
                n_estimators=650, learning_rate=0.025, max_depth=7, subsample=0.9, colsample_bytree=0.95,
                reg_lambda=1.3, reg_alpha=0.1, tree_method='hist', random_state=cfg.seed
            ),
            dino_train_df,
            dino_test_df,
            feature_engine=feat_engine_dino,
            semantic_train=sem_train_full,
            semantic_test=sem_test_full,
            n_splits_override=cfg.dino_kfolds,
        )
        _, proc_dino_xgb = compare_results(oof_dino_xgb, dino_train_df, return_scores=True)
        register_dino("dinov2_xgb", oof_dino_xgb, pred_dino_xgb, proc_dino_xgb)

        oof_dino_lgbm, pred_dino_lgbm = cross_validate(
            LGBMRegressor(
                n_estimators=900, learning_rate=0.03, num_leaves=192, subsample=0.85, colsample_bytree=0.9,
                reg_lambda=0.9, reg_alpha=0.1, random_state=cfg.seed, verbose=-1
            ),
            dino_train_df,
            dino_test_df,
            feature_engine=feat_engine_dino,
            semantic_train=sem_train_full,
            semantic_test=sem_test_full,
            n_splits_override=cfg.dino_kfolds,
        )
        _, proc_dino_lgbm = compare_results(oof_dino_lgbm, dino_train_df, return_scores=True)
        register_dino("dinov2_lgbm", oof_dino_lgbm, pred_dino_lgbm, proc_dino_lgbm)

        oof_dino_lasso, pred_dino_lasso = cross_validate(
            Lasso(alpha=0.0005, max_iter=5000),
            dino_train_df,
            dino_test_df,
            feature_engine=feat_engine_dino,
            semantic_train=sem_train_full,
            semantic_test=sem_test_full,
            n_splits_override=cfg.dino_kfolds,
        )
        _, proc_dino_lasso = compare_results(oof_dino_lasso, dino_train_df, return_scores=True)
        register_dino("dinov2_lasso", oof_dino_lasso, pred_dino_lasso, proc_dino_lasso)

        dino_weights = np.array(dino_weights, dtype=np.float64)
        dino_weights /= dino_weights.sum()
        dino_stack = np.stack(dino_preds, axis=0)
        dino_weighted_pred = np.tensordot(dino_weights, dino_stack, axes=(0, 0))
        dino_oof_stack = np.concatenate(dino_oof, axis=1)
        dino_test_stack = np.concatenate(dino_preds, axis=1)
        dino_meta = Ridge(alpha=0.25)
        dino_meta.fit(dino_oof_stack, dino_train_df[TARGET_NAMES].values)
        dino_meta_oof = dino_meta.predict(dino_oof_stack)
        dino_meta_pred = dino_meta.predict(dino_test_stack)
        meta_cv = competition_metric(dino_train_df[TARGET_NAMES].values, dino_meta_oof)
        print(f"[Dinov2] Ridge meta CV: {meta_cv:.6f}")

        dino_weighted_oof = np.tensordot(dino_weights, np.stack(dino_oof, axis=0), axes=(0, 0))
        dinov2_pred = 0.7 * dino_meta_pred + 0.3 * dino_weighted_pred
        dinov2_cv = competition_metric(dino_train_df[TARGET_NAMES].values, 0.7 * dino_meta_oof + 0.3 * dino_weighted_oof)
        print(f"[Dinov2] Blended CV: {dinov2_cv:.6f}")

        dinov2_out = dino_test_df[['image_path']].copy()
        dinov2_out[TARGET_NAMES] = dinov2_pred
        dinov2_out = post_process_biomass(dinov2_out)
        dinov2_sub = melt_table(dinov2_out)
        dinov2_sub[["sample_id", "target"]].to_csv("submission4.csv", index=False)
        pillar_paths["dinov2"] = Path("submission4.csv")
    else:
        print("[Config] Skipping Dinov2 pillar.")
        pillar_paths["dinov2"] = None

    if cfg.run_dino1:
        print("DINO Model 1 ===")
        run_dino_inference()
        pillar_paths["dino"] = Path("submission3.csv") if Path("submission3.csv").exists() else None
    else:
        print("[Config] Skipping DINO Model 1 pillar.")
        pillar_paths["dino"] = None

    if cfg.run_mvp:
        print("DINO Model 2 ===")
        run_mvp_inference()
        pillar_paths["mvp"] = Path("submission2.csv") if Path("submission2.csv").exists() else None
    else:
        print("[Config] Skipping MVP pillar.")
        pillar_paths["mvp"] = None

    blend_candidates = []
    default_weights = {
        "siglip": 0.50,
        "dino": 0.25,
        "mvp": 0.15,
        "dinov2": 0.10,
    }
    submission_map = {
        "siglip": Path("submission1.csv"),
        "dino": Path("submission3.csv"),
        "mvp": Path("submission2.csv"),
        "dinov2": Path("submission4.csv"),
    }
    for key, path in submission_map.items():
        if path.exists():
            df = pd.read_csv(path)
            blend_candidates.append((default_weights[key], df))
        else:
            print(f"[Blend] Missing {path}, skipping.")

    if not blend_candidates:
        raise RuntimeError("No pillar submissions available for blending.")

    merged = None
    total_w = sum(w for w, _ in blend_candidates)
    for idx, (w, df) in enumerate(blend_candidates, start=1):
        col = f"target_{idx}"
        df = df.rename(columns={"target": col})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="sample_id")
    weight_cols = [c for c in merged.columns if c.startswith("target_")]
    weights = np.array([w for w, _ in blend_candidates], dtype=np.float64)
    weights /= total_w
    merged["target"] = sum(merged[col] * weights[i] for i, col in enumerate(weight_cols))
    print(f"[Blend] Final weights (normalized): {', '.join(f'{w:.2f}' for w in weights.tolist())}")

    final_submission = merged[['sample_id', 'target']]
    final_submission.to_csv('submission.csv', index=False)

    pillar_dir = Path("pillar_outputs")
    pillar_dir.mkdir(exist_ok=True)
    pillar_sources = {
        pillar_dir / "submission_siglip.csv": submission_map["siglip"],
        pillar_dir / "submission_dino.csv": submission_map["dino"],
        pillar_dir / "submission_mvp.csv": submission_map["mvp"],
        pillar_dir / "submission_dinov2.csv": submission_map["dinov2"],
    }
    for dst, src in pillar_sources.items():
        if src.exists():
            shutil.copy(src, dst)
            print(f"[Pillar Export] Copied {src} -> {dst}")
        else:
            print(f"[Pillar Export] Missing {src}, skipping export.")

if __name__ == "__main__":
    main()
