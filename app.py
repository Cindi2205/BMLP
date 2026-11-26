import pandas as pd
import streamlit as st

from config.domain_config import APP_CONFIG, CLUSTER_COLUMN
from modules.inference import run_classification, run_clustering_and_pca
from modules.ml_loader import load_datasets, load_models
from modules.preprocessing import clean_dataframe, get_feature_columns
from modules.ui_components import (
    render_cluster_distribution,
    render_data_table,
    render_header,
    render_kpi_metrics,
    render_pca_scatter,
)

st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    layout=APP_CONFIG["layout"],
)

render_header(APP_CONFIG["header_title"], APP_CONFIG["header_subtitle"])

datasets = load_datasets()
models = load_models()

if not datasets:
    st.error("Dataset tidak ditemukan di folder assets/data.")
    st.stop()

dataset_choice = st.sidebar.selectbox(
    "Pilih Dataset Referensi",
    options=list(datasets.keys()),
)

page = st.sidebar.radio(
    "Navigasi Halaman",
    options=["Overview & Data", "Clustering & PCA", "Prediksi Klasifikasi"],
)

raw_df = datasets[dataset_choice]
clean_df = clean_dataframe(raw_df)
feature_cols = get_feature_columns(clean_df)

if page == "Overview & Data":
    cluster_count = (
        clean_df[CLUSTER_COLUMN].nunique()
        if CLUSTER_COLUMN in clean_df.columns
        else 0
    )
    render_kpi_metrics({
        "Total Baris": len(clean_df),
        "Total Kolom": len(clean_df.columns),
        "Jumlah Fitur Numerik": len(feature_cols),
        "Jumlah Cluster": cluster_count,
    })
    st.subheader("Pratinjau Dataset Bersih")
    render_data_table(clean_df)

    st.divider()
    st.subheader("Penjelasan Alur & Kode Notebook")
    st.caption(
        "Bagian ini menjelaskan maksud tahapan dan kode di notebook agar pengguna mudah memahami arti angka, kategori, dan hasil pemodelan."
    )

    tab_prep, tab_cluster, tab_clf = st.tabs([
        "1. Persiapan Data & Skala",
        "2. Clustering & PCA",
        "3. Klasifikasi & Makna Hasil",
    ])

    with tab_prep:
        st.markdown(
            """
            * **Pembersihan data (`dropna` / pembersihan nilai kosong & outlier):** Merapikan data mentah agar baris kosong atau nilai ekstrem tidak mengganggu perhitungan pola.
            * **Pengubahan teks menjadi kode angka (`LabelEncoder` / `map`):** Kategori seperti jenis transaksi (`TransactionType`), lokasi (`Location`), kanal (`Channel`), profesi (`CustomerOccupation`), atau kelompok umur (`AgeGroup`) diubah menjadi angka agar bisa diproses model. Angka ini berfungsi sebagai kode penanda kategori, bukan ukuran besar-kecil atau skor risiko.
            * **Penyamakan skala angka (`StandardScaler`):** Nominal transaksi (`TransactionAmount`), usia (`CustomerAge`), durasi (`TransactionDuration`), dan saldo (`AccountBalance`) disamakan skalanya agar satu kolom tidak mendominasi kolom lain.
              * **Angka `0`:** Mendekati rata-rata data acuan.
              * **Angka minus:** Lebih kecil dari rata-rata data acuan (bukan berarti saldo minus atau utang).
              * **Angka positif lebih besar:** Lebih besar dari rata-rata data acuan.
            * **`LoginAttempts`:** Jumlah percobaan masuk/login sebelum transaksi; nilai yang lebih tinggi perlu diperhatikan karena bisa menandakan aktivitas yang tidak biasa.
            """
        )
        with st.expander("Contoh kode persiapan data di notebook"):
            st.code(
                """
df = df.dropna()
encoder = LabelEncoder()
scaler = StandardScaler()
                """,
                language="python",
            )

    with tab_cluster:
        st.markdown(
            """
            * **Pengelompokan pola serupa (`KMeans`):** Mengelompokkan transaksi dengan karakteristik yang mirip ke dalam beberapa segmen (*cluster*) agar pola umum dan kelompok transaksi berisiko tinggi lebih mudah dibedakan.
            * **Peta sebaran 2 dimensi (`PCA`):** Merangkum banyak kolom angka menjadi komponen utama (`PC1` dan `PC2`) agar sebaran data dan titik yang menyimpang bisa dilihat secara visual di layar.
            * **Pengembalian skala (`inverse` / `data_clustering_inverse`):** Mengembalikan nilai yang sudah di-*scale* ke bentuk skala yang lebih mudah dibaca secara bisnis saat menafsirkan karakteristik transaksi.
            """
        )
        with st.expander("Contoh kode clustering & PCA di notebook"):
            st.code(
                """
pca = PCA(n_components=2, random_state=42)
kmeans = KMeans(n_clusters=k, random_state=42)
                """,
                language="python",
            )

    with tab_clf:
        st.markdown(
            """
            * **Pembagian data latih & uji (`train_test_split`):** Memisahkan data menjadi bagian latihan (*train*) dan pengujian (*test*) agar model diuji pada data baru dan tidak sekadar menghafal data latihan.
            * **Pelatihan model (`DecisionTreeClassifier` & `RandomForestClassifier`):** Melatih model pohon keputusan dan hutan acak untuk mengenali pola transaksi normal maupun yang menyimpang.
            * **Arti hasil prediksi (`Kelas 0` vs `Kelas 1`):**
              * **`Kelas 0`:** Transaksi terprediksi normal / wajar sesuai pola umum data acuan.
              * **`Kelas 1`:** Transaksi terprediksi anomali / berpotensi menyimpang sehingga layak ditinjau lebih lanjut.
            * **Catatan penting:** Hasil prediksi berfungsi sebagai penyaring awal untuk membantu analisis lanjutan.
            """
        )
        with st.expander("Contoh kode klasifikasi di notebook"):
            st.code(
                """
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
                """,
                language="python",
            )

elif page == "Clustering & PCA":
    result = run_clustering_and_pca(clean_df, models)
    clustered_df = result["data"]
    profile_df = result["profile"]

    col_left, col_right = st.columns(2)
    with col_left:
        render_pca_scatter(clustered_df)
    with col_right:
        render_cluster_distribution(clustered_df)

    st.subheader("Profil Rata-Rata per Cluster")
    render_data_table(profile_df, max_rows=50)

    st.subheader("Data Hasil Segmentasi")
    render_data_table(clustered_df)

elif page == "Prediksi Klasifikasi":
    st.subheader("Uji Prediksi Data")
    model_choice = st.selectbox(
        "Pilih Model Klasifikasi",
        options=["classification_tuned", "decision_tree"],
    )

    uploaded_file = st.file_uploader("Unggah file CSV (opsional)", type=["csv"])
    if uploaded_file is not None:
        sample_df = pd.read_csv(uploaded_file)
    else:
        sample_df = clean_df[feature_cols].head(5).copy() if feature_cols else clean_df.head(5).copy()

    st.caption("Data input yang akan diprediksi:")
    render_data_table(sample_df, max_rows=10)

    if st.button("Jalankan Prediksi"):
        outcome = run_classification(
            sample_df,
            models=models,
            model_key=model_choice,
            feature_cols=feature_cols,
        )
        if outcome["status"] != "ok":
            st.warning("Model klasifikasi belum dapat menghasilkan prediksi pada environment saat ini.")
        else:
            output_df = sample_df.copy()
            output_df["Prediksi"] = outcome["predictions"]
            if outcome["probabilities"] is not None:
                output_df["Probabilitas"] = [round(p, 4) for p in outcome["probabilities"]]
            st.subheader("Hasil Prediksi")
            render_data_table(output_df, max_rows=20)