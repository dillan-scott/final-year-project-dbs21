from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DAY_FILES_DEFAULT = [
    ('Monday-WorkingHours.csv',    0),
    ('Tuesday-WorkingHours.csv',   1),
    ('Wednesday-WorkingHours.csv', 2),
    ('Thursday-WorkingHours.csv',  3),
    ('Friday-WorkingHours.csv',    4),
]
DAY_NAMES_DEFAULT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']


@dataclass
class LoadedData:
    """Output of load_cicids2017: standardised features + binary/multi-class labels + day index."""
    X: np.ndarray
    y_all: np.ndarray
    y_str: np.ndarray
    day_idx: np.ndarray
    le_full: LabelEncoder
    input_dim: int
    benign_id: int

@dataclass
class TemporalSplit:
    """Output of prepare_temporal_split: training set, fixed eval set, day-sorted stream."""
    X_tr: np.ndarray
    y_tr: np.ndarray
    y_tr_mc: np.ndarray
    y_tr_bin: np.ndarray
    le_known: LabelEncoder
    n_known: int

    X_eval: torch.Tensor
    y_eval_full: np.ndarray
    y_eval_bin: np.ndarray
    novel_mask_eval: np.ndarray
    per_class_masks: dict[str, np.ndarray]

    X_stream: np.ndarray
    y_stream: np.ndarray
    y_stream_bin: np.ndarray
    d_stream: np.ndarray

    novel_pool: np.ndarray
    novel_ids: np.ndarray


def load_cicids2017(
    data_dir: str,
    day_files: list[tuple[str, int]] | None = None,
) -> LoadedData:
    """Load CICIDS2017 (Engelen-corrected) with per-row day index preserved.
      - merges DoS variants into 'DoS'
      - merges FTP/SSH-Patator into 'Patator'
      - merges Web Attack variants into 'Web Attack'
      - relabels '... - Attempted' as 'BENIGN'
      - drops Flow ID/IP/Port/Timestamp metadata
      - fits StandardScaler on full data
    """
    day_files = day_files or DAY_FILES_DEFAULT
    frames = []
    for fname, didx in day_files:
        df_day = pd.read_csv(os.path.join(data_dir, fname))
        df_day['day_idx'] = didx
        frames.append(df_day)
    df = pd.concat(frames, ignore_index=True)

    df.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp'],
            inplace=True, errors='ignore')
    df['Label'] = df['Label'].apply(lambda x: 'BENIGN' if x.endswith('- Attempted') else x)
    df['Label'] = df['Label'].replace({
        'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS', 'DoS Slowhttptest': 'DoS',
        'Web Attack - Brute Force': 'Web Attack',
        'Web Attack - XSS': 'Web Attack',
        'Web Attack - Sql Injection': 'Web Attack',
        'FTP-Patator': 'Patator', 'SSH-Patator': 'Patator',
    })

    day_idx_col = df['day_idx'].values
    X_raw = df[[c for c in df.columns if c not in ('Label', 'day_idx')]].copy()
    y_str = df['Label'].values
    X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_raw.dropna(axis=1, how='all', inplace=True)
    X_raw.fillna(X_raw.mean(), inplace=True)

    X = StandardScaler().fit_transform(X_raw).astype(np.float32)
    le_full = LabelEncoder()
    y_all = le_full.fit_transform(y_str)
    return LoadedData(
        X=X, y_all=y_all, y_str=y_str, day_idx=day_idx_col,
        le_full=le_full, input_dim=X.shape[1],
        benign_id=int(le_full.transform(['BENIGN'])[0]),
    )


def prepare_temporal_split(
    data: LoadedData,
    novel_classes: list[str],
    per_class_metric_names: list[str] | None = None,
    test_size: float = 0.25,
    eval_n: int = 5_000,
    random_seed: int = 42,
) -> TemporalSplit:
    """
    Build initial-training set + balanced fixed eval set + day-sorted streaming set.
    """
    rng = np.random.default_rng(random_seed)

    novel_ids = np.array([data.le_full.transform([c])[0] for c in novel_classes])
    known_mask = ~np.isin(data.y_all, novel_ids)
    X_kn = data.X[known_mask]
    y_kn = data.y_all[known_mask]
    d_kn = data.day_idx[known_mask]
    orig_idx_kn = np.where(known_mask)[0]

    X_tr, X_ho, y_tr, y_ho, d_tr, d_ho, oi_tr, oi_ho = train_test_split(
        X_kn, y_kn, d_kn, orig_idx_kn,
        test_size=test_size, random_state=random_seed, stratify=y_kn,
    )
    le_known = LabelEncoder()
    y_tr_mc = le_known.fit_transform(y_tr).astype(np.int64)
    y_tr_bin = (y_tr != data.benign_id).astype(np.int64)
    n_known = len(le_known.classes_)

    novel_ho_mask = np.isin(data.y_all, novel_ids)
    X_nov_all = data.X[novel_ho_mask]
    y_nov_all = data.y_all[novel_ho_mask]
    d_nov_all = data.day_idx[novel_ho_mask]
    oi_nov_all = np.where(novel_ho_mask)[0]

    n_eval = min(eval_n, len(X_ho))
    n_eval_nov = min(eval_n, len(X_nov_all))
    eval_kn_idx = rng.choice(len(X_ho), n_eval, replace=False)
    eval_nov_idx = rng.choice(len(X_nov_all), n_eval_nov, replace=False)

    X_eval = torch.from_numpy(np.vstack([X_nov_all[eval_nov_idx], X_ho[eval_kn_idx]]))
    y_eval_full = np.concatenate([y_nov_all[eval_nov_idx], y_ho[eval_kn_idx]])
    y_eval_bin = np.concatenate([
        np.ones(n_eval_nov, dtype=int),
        (y_ho[eval_kn_idx] != data.benign_id).astype(int),
    ])
    novel_mask_eval = np.concatenate([np.ones(n_eval_nov, bool), np.zeros(n_eval, bool)])

    per_class_masks: dict[str, np.ndarray] = {}
    for cname in (per_class_metric_names or []):
        cid = data.le_full.transform([cname])[0]
        per_class_masks[cname.lower().replace(' ', '_')] = (y_eval_full == cid)

    eval_ho_set = set(eval_kn_idx.tolist())
    eval_nov_set = set(eval_nov_idx.tolist())
    stream_kn_mask = np.array([i not in eval_ho_set for i in range(len(X_ho))])
    stream_nov_mask = np.array([i not in eval_nov_set for i in range(len(X_nov_all))])

    X_stream = np.vstack([X_ho[stream_kn_mask], X_nov_all[stream_nov_mask]])
    y_stream = np.concatenate([y_ho[stream_kn_mask], y_nov_all[stream_nov_mask]])
    d_stream = np.concatenate([d_ho[stream_kn_mask], d_nov_all[stream_nov_mask]])
    oi_stream = np.concatenate([oi_ho[stream_kn_mask], oi_nov_all[stream_nov_mask]])
    sort_order = np.lexsort((oi_stream, d_stream))
    X_stream = X_stream[sort_order]
    y_stream = y_stream[sort_order]
    d_stream = d_stream[sort_order]
    y_stream_bin = (y_stream != data.benign_id).astype(np.int64)

    novel_pool = data.X[novel_ho_mask]

    return TemporalSplit(
        X_tr=X_tr, y_tr=y_tr, y_tr_mc=y_tr_mc, y_tr_bin=y_tr_bin,
        le_known=le_known, n_known=n_known,
        X_eval=X_eval, y_eval_full=y_eval_full, y_eval_bin=y_eval_bin,
        novel_mask_eval=novel_mask_eval, per_class_masks=per_class_masks,
        X_stream=X_stream, y_stream=y_stream, y_stream_bin=y_stream_bin, d_stream=d_stream,
        novel_pool=novel_pool, novel_ids=novel_ids,
    )
