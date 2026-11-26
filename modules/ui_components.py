from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


def render_header(title: str, subtitle: str) -> None:
    """Menampilkan judul utama dan deskripsi singkat aplikasi."""
    st.title(title)
    st.caption(subtitle)
    st.divider()


def render_kpi_metrics(metrics: dict[str, Any]) -> None:
    """Menampilkan kartu metrik ringkas secara berdampingan."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label=label, value=value)


def render_data_table(df: pd.DataFrame, max_rows: int = 20) -> None:
    """Menampilkan tabel data dengan batas baris agar tetap ringan."""
    if df is None or df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return
    st.dataframe(df.head(max_rows), use_container_width=True)


def render_pca_scatter(df: pd.DataFrame, color_col: str = "Cluster_Label") -> None:
    """Menampilkan visualisasi sebaran PCA 2D menggunakan Plotly."""
    if df is None or df.empty or not {"PC1", "PC2"}.issubset(df.columns):
        st.info("Koordinat PCA belum tersedia untuk divisualisasikan.")
        return
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color=color_col if color_col in df.columns else None,
        title="Sebaran Data pada Ruang PCA (2D)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cluster_distribution(df: pd.DataFrame, cluster_col: str = "Cluster_Label") -> None:
    """Menampilkan jumlah anggota tiap cluster dalam bentuk bar chart."""
    if df is None or df.empty or cluster_col not in df.columns:
        return
    counts = df[cluster_col].value_counts().reset_index()
    counts.columns = [cluster_col, "Jumlah Data"]
    fig = px.bar(
        counts,
        x=cluster_col,
        y="Jumlah Data",
        color=cluster_col,
        title="Distribusi Jumlah Data per Cluster",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)