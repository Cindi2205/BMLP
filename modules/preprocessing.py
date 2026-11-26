import numpy as np
import pandas as pd

from config.domain_config import CLUSTER_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN

COLUMN_ALIASES: dict[str, str] = {
    "transactionamount": "TransactionAmount",
    "transaction_amount": "TransactionAmount",
    "amount": "TransactionAmount",
    "amt": "TransactionAmount",
    "customerage": "CustomerAge",
    "customer_age": "CustomerAge",
    "age": "CustomerAge",
    "transactionduration": "TransactionDuration",
    "transaction_duration": "TransactionDuration",
    "duration": "TransactionDuration",
    "loginattempts": "LoginAttempts",
    "login_attempts": "LoginAttempts",
    "attempts": "LoginAttempts",
    "accountbalance": "AccountBalance",
    "account_balance": "AccountBalance",
    "balance": "AccountBalance",
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Memetakan variasi nama kolom umum ke nama kolom standar fitur."""
    target_set = set(FEATURE_COLUMNS or [])
    rename_map: dict[str, str] = {}

    for col in df.columns:
        if col in target_set:
            rename_map[col] = col
            continue
        key = str(col).strip().lower().replace("-", "_").replace(" ", "_")
        if key in COLUMN_ALIASES:
            mapped = COLUMN_ALIASES[key]
            # Hindari menimpa kolom yang sudah ada namanya persis
            if mapped not in df.columns and mapped not in rename_map.values():
                rename_map[col] = mapped

    return df.rename(columns=rename_map)


def clean_dataframe(df: pd.DataFrame, drop_dups: bool = False) -> pd.DataFrame:
    """Membersihkan nilai kosong, (opsional) duplikat, dan merapikan tipe data."""
    cleaned = normalize_column_names(df.copy())
    if drop_dups:
        cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.reset_index(drop=True)

    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            series_num = pd.to_numeric(cleaned[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median_val = series_num.median()
            fill_num = 0.0 if pd.isna(median_val) else float(median_val)
            cleaned[col] = series_num.fillna(fill_num)
        else:
            mode_series = cleaned[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_val)

    return cleaned


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Menentukan daftar kolom fitur dari konfigurasi atau deteksi kolom numerik."""
    normalized_df = normalize_column_names(df)
    if FEATURE_COLUMNS:
        return list(FEATURE_COLUMNS)

    excluded = {TARGET_COLUMN, CLUSTER_COLUMN}
    return [
        col for col in normalized_df.select_dtypes(include=[np.number]).columns
        if col not in excluded
    ]


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    drop_dups: bool = False,
) -> pd.DataFrame:
    """Menyelaraskan urutan kolom, mengisi kolom kurang, dan menyiapkan float32."""
    cleaned = clean_dataframe(df, drop_dups=drop_dups)
    cols = feature_cols if feature_cols else get_feature_columns(cleaned)

    aligned = pd.DataFrame(index=cleaned.index)
    for col in cols:
        if col in cleaned.columns:
            s = pd.to_numeric(cleaned[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = s.median()
            fill_v = 0.0 if pd.isna(med) else float(med)
            aligned[col] = s.fillna(fill_v)
        else:
            aligned[col] = 0.0

    return aligned.astype(np.float32)