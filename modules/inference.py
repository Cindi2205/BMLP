from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config.domain_config import CLASS_LABELS, CLUSTER_COLUMN, CLUSTER_LABELS
from modules.preprocessing import get_feature_columns, prepare_features


def _predict_array(model: Any, X: np.ndarray) -> np.ndarray | None:
    """Menjalankan prediksi aman untuk model Keras maupun estimator standar."""
    if model is None or not hasattr(model, "predict"):
        return None
    try:
        return np.asarray(model.predict(X))
    except Exception:
        return None


def run_clustering_and_pca(df: pd.DataFrame, models: dict[str, Any]) -> dict[str, Any]:
    """Menghasilkan label cluster, koordinat 2D PCA, dan profil rata-rata tiap cluster."""
    feature_cols = get_feature_columns(df)
    X_df = prepare_features(df, feature_cols)
    X = X_df.to_numpy()

    result_df = df.copy()

    # Tentukan label cluster (dari dataset jika sudah ada, atau dari model)
    if CLUSTER_COLUMN in result_df.columns:
        clusters = result_df[CLUSTER_COLUMN].to_numpy()
    else:
        raw_pred = _predict_array(models.get("clustering"), X)
        if raw_pred is not None:
            clusters = np.argmax(raw_pred, axis=1) if raw_pred.ndim > 1 else raw_pred.ravel()
        else:
            clusters = np.zeros(len(result_df), dtype=int)
        result_df[CLUSTER_COLUMN] = clusters

    result_df["Cluster_Label"] = [
        CLUSTER_LABELS.get(c, f"Cluster {c}") for c in result_df[CLUSTER_COLUMN]
    ]

    # Hitung komponen PCA 2D untuk visualisasi sebaran
    pca_pred = _predict_array(models.get("pca"), X)
    if pca_pred is not None and pca_pred.ndim == 2 and pca_pred.shape[1] >= 2:
        coords = pca_pred[:, :2]
    elif X.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        coords = np.column_stack([X[:, 0], np.zeros(len(X))])

    result_df["PC1"] = coords[:, 0]
    result_df["PC2"] = coords[:, 1]

    cluster_profile = (
        result_df.groupby("Cluster_Label")[feature_cols]
        .mean()
        .round(3)
        .reset_index()
    )

    return {
        "data": result_df,
        "profile": cluster_profile,
        "feature_cols": feature_cols,
    }


def run_classification(
    input_df: pd.DataFrame,
    models: dict[str, Any],
    model_key: str = "classification_tuned",
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Menjalankan prediksi klasifikasi dan mengembalikan kelas beserta probabilitas."""
    X_df = prepare_features(input_df, feature_cols)
    X = X_df.to_numpy()

    model = models.get(model_key)
    raw_pred = _predict_array(model, X)

    if raw_pred is None:
        return {
            "status": "unavailable",
            "predictions": [],
            "probabilities": None,
        }

    if raw_pred.ndim > 1 and raw_pred.shape[1] > 1:
        pred_ids = np.argmax(raw_pred, axis=1)
        probs = np.max(raw_pred, axis=1)
    else:
        flat = raw_pred.ravel()
        is_prob = np.all((flat >= 0.0) & (flat <= 1.0)) and not np.all(np.isin(flat, [0, 1]))
        pred_ids = (flat >= 0.5).astype(int) if is_prob else flat.astype(int)
        probs = flat if is_prob else None

    labels = [CLASS_LABELS.get(int(idx), f"Kelas {int(idx)}") for idx in pred_ids]

    return {
        "status": "ok",
        "predictions": labels,
        "probabilities": probs.tolist() if probs is not None else None,
    }