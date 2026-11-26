from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "assets" / "data"
MODELS_DIR = BASE_DIR / "assets" / "models"

APP_CONFIG = {
    "page_title": "BMLP Machine Learning Portal",
    "layout": "wide",
    "header_title": "Machine Learning & Data Segmentation Portal",
    "header_subtitle": "Eksplorasi clustering, PCA, dan prediksi klasifikasi berbasis artefak model BMLP.",
}

DATA_PATHS = {
    "data_clustering": DATA_DIR / "data_clustering.csv",
    "data_clustering_inverse": DATA_DIR / "data_clustering_inverse.csv",
}

MODEL_PATHS = {
    "decision_tree": MODELS_DIR / "decision_tree_model.h5",
    "clustering": MODELS_DIR / "model_clustering.h5",
    "pca": MODELS_DIR / "PCA_model_clustering.h5",
    "classification_tuned": MODELS_DIR / "tuning_classification.h5",
}

CLUSTER_COLUMN = "Cluster"
TARGET_COLUMN = "Target"
FEATURE_COLUMNS: list[str] = []
CLUSTER_LABELS: dict[int, str] = {}
CLASS_LABELS: dict[int, str] = {}