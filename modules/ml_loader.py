from pathlib import Path
import pickle
import h5py
import joblib
import pandas as pd
import streamlit as st

from config.domain_config import DATA_PATHS, MODEL_PATHS


def _load_single_h5(model_path: str | Path, model_key: str = "model"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"[{model_key}] Model file not found: {path}")

    try:
        import keras
        return keras.models.load_model(path, compile=False)
    except Exception:
        pass

    try:
        return h5py.File(path, "r")
    except Exception:
        pass

    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        pass

    try:
        return joblib.load(path)
    except Exception as exc:
        raise OSError(f"[{model_key}] Failed to load model artifact '{path}': {exc}") from exc


@st.cache_data
def load_datasets() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for key, file_path in DATA_PATHS.items():
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"[{key}] Dataset file not found: {path}")
        datasets[key] = pd.read_csv(path)
    return datasets


@st.cache_resource
def load_models() -> dict[str, object]:
    models: dict[str, object] = {}
    for key, file_path in MODEL_PATHS.items():
        models[key] = _load_single_h5(file_path, model_key=key)
    return models