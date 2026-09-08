"""
Aplikasi Streamlit — Segmentasi & Rekomendasi Pemain Football Manager 2023
============================================================================
Menggunakan model hasil training pada notebook `Analisis_Segmentasi_Pemain_FM2023.ipynb`:
    - segmentation_model.pkl  -> PlayerSegmentationModel (scaler + KMeans + metadata)
    - kmeans_model.pkl        -> komponen individual KMeans (fallback)
    - scaler_clustering.pkl   -> komponen individual StandardScaler untuk clustering (fallback)
    - knn_model.pkl           -> NearestNeighbors untuk sistem "pemain mirip"
    - scaler_knn.pkl          -> StandardScaler khusus fitur KNN
    - metadata.pkl            -> dict berisi cluster_features, cluster_names, ROLE_WEIGHTS, dll.

Jalankan dengan:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Wajib di-import agar joblib.load() bisa unpickle PlayerSegmentationModel
from segmentation_model import PlayerSegmentationModel  # noqa: F401

# ----------------------------------------------------------------------------
# Konfigurasi halaman & path
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="FM2023 Player Segmentation & Recommendation",
    page_icon="⚽",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "fm2023.csv")

MODEL_PATHS = {
    "segmentation_model": os.path.join(APP_DIR, "segmentation_model.pkl"),
    "kmeans_model": os.path.join(APP_DIR, "kmeans_model.pkl"),
    "scaler_clustering": os.path.join(APP_DIR, "scaler_clustering.pkl"),
    "knn_model": os.path.join(APP_DIR, "knn_model.pkl"),
    "scaler_knn": os.path.join(APP_DIR, "scaler_knn.pkl"),
    "metadata": os.path.join(APP_DIR, "metadata.pkl"),
}

ATTRIBUTE_COLS = [
    "Acc", "Agi", "Ant", "Bal", "Bra", "Cmp", "Dec", "Dri", "Fin",
    "Fir", "Fla", "Hea", "Lon", "Mar", "Pas", "Pac", "Pos", "Sta",
    "Str", "Tck", "Tec", "Tea", "Vis", "Wor", "Agg", "Cnt", "Cor",
    "Cro", "Det", "Fre", "Jum", "Pen", "Thr",
]


# ============================================================================
# 1. LOAD MODEL (di-cache sebagai resource, tidak di-hash ulang tiap rerun)
# ============================================================================
@st.cache_resource(show_spinner="Memuat model machine learning...")
def load_models():
    segmentation_model = joblib.load(MODEL_PATHS["segmentation_model"])
    knn_model = joblib.load(MODEL_PATHS["knn_model"])
    scaler_knn = joblib.load(MODEL_PATHS["scaler_knn"])
    metadata = joblib.load(MODEL_PATHS["metadata"])
    return segmentation_model, knn_model, scaler_knn, metadata


# ============================================================================
# 2. PIPELINE DATA — mereplikasi persis notebook (cleaning & feature engineering)
# ============================================================================
def parse_single_currency(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().upper()
    val = (
        val.replace("€", "").replace("£", "").replace("$", "")
        .replace("\xa0", "").replace("P/M", "").replace("PW", "")
        .replace("/MONTH", "").replace("/WEEK", "").strip()
    )
    if not val or val in ["NAN", "NONE", "NOT FOR SALE", "N/A", "-", "FREE", "LOAN"]:
        return np.nan

    multiplier = 1
    if val.endswith("M"):
        multiplier = 1_000_000
        val = val[:-1]
    elif val.endswith("K"):
        multiplier = 1_000
        val = val[:-1]
    elif val.endswith("B"):
        multiplier = 1_000_000_000
        val = val[:-1]

    if "," in val:
        parts = val.split(",")
        if len(parts) == 2:
            after_comma = parts[1]
            if multiplier > 1 or len(after_comma) < 3:
                val = val.replace(",", ".")
            else:
                val = val.replace(",", "")
    try:
        return float(val.strip()) * multiplier
    except ValueError:
        return np.nan


def parse_currency_range(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if " - " in val_str or "-" in val_str:
        parts = val_str.split("-")
        if len(parts) == 2:
            try:
                min_val = parse_single_currency(parts[0])
                max_val = parse_single_currency(parts[1])
                if min_val is not None and max_val is not None:
                    return (min_val + max_val) / 2
            except Exception:
                pass
    return parse_single_currency(val_str)


def clean_fm_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    if "UID" in df.columns:
        df = df.drop_duplicates(subset="UID", keep="first")

    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    for col in ["Transfer Value", "Wage"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_currency_range)

    for col in ATTRIBUTE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, physical_attrs, technical_attrs, mental_attrs) -> pd.DataFrame:
    df = df.copy()
    phys = [c for c in physical_attrs if c in df.columns]
    tech = [c for c in technical_attrs if c in df.columns]
    ment = [c for c in mental_attrs if c in df.columns]

    if phys:
        df["Physical Score"] = df[phys].mean(axis=1)
    if tech:
        df["Technical Score"] = df[tech].mean(axis=1)
    if ment:
        df["Mental Score"] = df[ment].mean(axis=1)

    all_attrs = list(set(phys + tech + ment))
    all_attrs = [c for c in all_attrs if c in df.columns]
    if all_attrs:
        df["Overall Ability"] = df[all_attrs].mean(axis=1)

    def age_score(age):
        if pd.isna(age):
            return 50
        if age <= 18:
            return 80
        if age <= 21:
            return 100
        if age <= 25:
            return 90
        if age <= 28:
            return 75
        if age <= 30:
            return 60
        if age <= 33:
            return 40
        return 20

    df["Age Score"] = df["Age"].apply(age_score)

    if "Transfer Value" in df.columns:
        df["Value Efficiency"] = df["Overall Ability"] / np.log10(df["Transfer Value"].fillna(0) + 1)
    if "Wage" in df.columns:
        df["Wage Efficiency"] = df["Overall Ability"] / np.log10(df["Wage"].fillna(0) + 1)

    return df


def compute_role_score(df: pd.DataFrame, role: str, role_weights: dict) -> pd.Series:
    weights = role_weights[role]
    available = {k: v for k, v in weights.items() if k in df.columns}
    if not available:
        return pd.Series(np.nan, index=df.index)
    numerator = sum(df[attr] * w for attr, w in available.items())
    denominator = sum(available.values())
    return (numerator / denominator) * 5


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(50, index=s.index)
    return (s - mn) / (mx - mn) * 100


def recommend_players(
    df: pd.DataFrame,
    role: str,
    position: str = None,
    max_age: int = 30,
    max_transfer_value: float = np.inf,
    max_wage: float = np.inf,
    top_n: int = 10,
    weights: tuple = (0.6, 0.2, 0.2),
) -> pd.DataFrame:
    role_col = f"Role Score - {role}"
    filt = df.copy()
    if position:
        filt = filt[filt["Position"].str.contains(position, case=False, na=False)]
    if max_age:
        filt = filt[filt["Age"] <= max_age]
    if max_transfer_value < np.inf:
        filt = filt[(filt["Transfer Value"].isna()) | (filt["Transfer Value"] <= max_transfer_value)]
    if max_wage < np.inf:
        filt = filt[(filt["Wage"].isna()) | (filt["Wage"] <= max_wage)]

    filt = filt.dropna(subset=[role_col])
    if filt.empty:
        return pd.DataFrame()

    filt = filt.copy()
    filt["Role_norm"] = normalize_series(filt[role_col])
    filt["Age_norm"] = normalize_series(filt["Age Score"])
    filt["VE_norm"] = normalize_series(filt["Value Efficiency"])

    w_role, w_age, w_ve = weights
    filt["Final Score"] = (
        w_role * filt["Role_norm"] + w_age * filt["Age_norm"] + w_ve * filt["VE_norm"]
    )

    result = filt.sort_values("Final Score", ascending=False).head(top_n)
    out_cols = [
        "Name", "Age", "Position", "Club", "Best Role", role_col,
        "Transfer Value", "Wage", "Final Score", "Cluster Name",
    ]
    out_cols = [c for c in out_cols if c in result.columns]
    return result[out_cols].reset_index(drop=True)


# ============================================================================
# 3. LOAD & PROSES DATASET (di-cache — hanya dijalankan sekali)
# ============================================================================
@st.cache_data(show_spinner="Memproses dataset FM2023 (± 189 ribu baris, hanya dijalankan sekali)...")
def load_and_process_data():
    _, _, _, metadata = load_models()
    physical_attrs = metadata["physical_attrs"]
    technical_attrs = metadata["technical_attrs"]
    mental_attrs = metadata["mental_attrs"]
    role_weights = metadata["role_weights"]

    df_raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df_clean = clean_fm_dataset(df_raw)
    df_fe = engineer_features(df_clean, physical_attrs, technical_attrs, mental_attrs)

    for role in role_weights:
        df_fe[f"Role Score - {role}"] = compute_role_score(df_fe, role, role_weights)

    def get_best_role_score(row):
        role = row.get("Best Role")
        if pd.isna(role) or role not in role_weights:
            return np.nan
        return row.get(f"Role Score - {role}", np.nan)

    df_fe["Best Role Score"] = df_fe.apply(get_best_role_score, axis=1)

    # --- Segmentasi (clustering) memakai model yang sudah dilatih (bukan fit ulang) ---
    seg_model, knn_model, scaler_knn, _ = load_models()
    cluster_result = seg_model.predict(df_fe)
    df_fe["Cluster"] = np.nan
    df_fe["Cluster Name"] = pd.Series(np.nan, index=df_fe.index, dtype=object)
    df_fe.loc[cluster_result.index, "Cluster"] = cluster_result["Cluster"]
    df_fe.loc[cluster_result.index, "Cluster Name"] = cluster_result["Cluster Name"]

    # --- Data untuk KNN (player similarity), memakai scaler_knn yang sudah dilatih ---
    knn_features = metadata["knn_features"]
    df_knn = df_fe.dropna(subset=knn_features).copy()
    X_knn = scaler_knn.transform(df_knn[knn_features])

    return df_fe, df_knn, X_knn


def find_similar_players(player_name: str, df_knn, X_knn, knn_model, knn_features, n: int = 10):
    matches = df_knn[df_knn["Name"].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        return None, pd.DataFrame()

    target_idx = matches.index[0]
    target_player = df_knn.loc[target_idx]
    pos_in_matrix = df_knn.index.get_loc(target_idx)
    target_vector = X_knn[pos_in_matrix].reshape(1, -1)

    n_query = min(n + 1, X_knn.shape[0])
    distances, indices = knn_model.kneighbors(target_vector, n_neighbors=n_query)

    results = []
    for dist, neighbor_idx in zip(distances[0], indices[0]):
        if neighbor_idx == pos_in_matrix:
            continue
        neighbor_data = df_knn.iloc[neighbor_idx]
        similarity_pct = round(100 / (1 + dist), 2)
        results.append({
            "Name": neighbor_data["Name"],
            "Age": neighbor_data.get("Age"),
            "Position": neighbor_data.get("Position"),
            "Club": neighbor_data.get("Club"),
            "Best Role": neighbor_data.get("Best Role"),
            "Cluster Name": neighbor_data.get("Cluster Name"),
            "Distance": round(dist, 4),
            "Similarity %": similarity_pct,
        })
        if len(results) >= n:
            break
    return target_player, pd.DataFrame(results)


# ============================================================================
# 4. UI
# ============================================================================
def format_currency(v):
    if pd.isna(v):
        return "-"
    return f"€{v:,.0f}"


def main():
    seg_model, knn_model, scaler_knn, metadata = load_models()
    df_fe, df_knn, X_knn = load_and_process_data()
    knn_features = metadata["knn_features"]
    cluster_names = metadata["cluster_names"]
    role_weights = metadata["role_weights"]

    st.sidebar.title("⚽ FM2023 Player Analytics")
    st.sidebar.caption("Segmentasi & Rekomendasi Pemain berbasis Machine Learning")
    page = st.sidebar.radio(
        "Navigasi",
        [
            "🏠 Beranda",
            "🧩 Segmentasi Pemain (Clustering)",
            "🔍 Cari Pemain Mirip (KNN)",
            "⭐ Rekomendasi Pemain per Role",
            "📊 Eksplorasi Cluster",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Pemain (bersih)", f"{len(df_fe):,}")
    st.sidebar.metric("Jumlah Cluster", f"{len(cluster_names)}")

    # ------------------------------------------------------------------
    if page == "🏠 Beranda":
        st.title("⚽ Segmentasi & Sistem Rekomendasi Pemain FM2023")
        st.markdown(
            "Aplikasi ini menampilkan hasil model **K-Means Clustering** (segmentasi gaya "
            "bermain), **K-Nearest Neighbors** (pencarian pemain mirip), dan **rekomendasi "
            "pemain berbasis role tertimbang**, yang dilatih pada dataset ekspor "
            "Football Manager 2023."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pemain", f"{len(df_fe):,}")
        col2.metric("Rata-rata Umur", f"{df_fe['Age'].mean():.1f} th")
        col3.metric("Jumlah Klub", f"{df_fe['Club'].nunique():,}")
        col4.metric("Jumlah Cluster", f"{len(cluster_names)}")

        st.subheader("Distribusi Pemain per Cluster")
        cluster_counts = df_fe["Cluster Name"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster Name", "Jumlah Pemain"]
        fig = px.pie(cluster_counts, names="Cluster Name", values="Jumlah Pemain", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Contoh Data Pemain (setelah cleaning & feature engineering)")
        show_cols = ["Name", "Age", "Position", "Club", "Best Role",
                     "Overall Ability", "Cluster Name", "Transfer Value", "Wage"]
        show_cols = [c for c in show_cols if c in df_fe.columns]
        st.dataframe(df_fe[show_cols].head(20), use_container_width=True)

    # ------------------------------------------------------------------
    elif page == "🧩 Segmentasi Pemain (Clustering)":
        st.title("🧩 Segmentasi Pemain (K-Means Clustering)")
        st.write(
            "Prediksi cluster gaya bermain seorang pemain menggunakan model "
            "`PlayerSegmentationModel` (`segmentation_model.pkl`)."
        )

        mode = st.radio("Pilih mode", ["Pilih pemain dari dataset", "Input atribut manual"], horizontal=True)

        cluster_features = seg_model.cluster_features

        if mode == "Pilih pemain dari dataset":
            name_query = st.text_input("Cari nama pemain", value="Erling Haaland")
            candidates = df_fe[df_fe["Name"].str.contains(name_query, case=False, na=False)] if name_query else df_fe.head(0)
            if candidates.empty:
                st.warning("Pemain tidak ditemukan. Coba nama lain.")
            else:
                selected_name = st.selectbox("Pilih pemain", candidates["Name"].head(50).tolist())
                row = candidates[candidates["Name"] == selected_name].iloc[0]

                c1, c2, c3 = st.columns(3)
                c1.metric("Posisi", str(row.get("Position", "-")))
                c2.metric("Klub", str(row.get("Club", "-")))
                c3.metric("Umur", f"{row.get('Age', '-')}")

                if pd.isna(row.get("Cluster")):
                    st.warning("Pemain ini memiliki data atribut tidak lengkap sehingga tidak bisa di-cluster.")
                else:
                    st.success(f"**Cluster: {row['Cluster Name']}**")

                    centroid = seg_model.cluster_profile_summary()
                    player_vals = [row[f] for f in cluster_features]
                    centroid_vals = centroid.loc[row["Cluster Name"], cluster_features].values

                    radar_fig = go.Figure()
                    radar_fig.add_trace(go.Scatterpolar(r=player_vals, theta=cluster_features,
                                                         fill="toself", name=selected_name))
                    radar_fig.add_trace(go.Scatterpolar(r=centroid_vals, theta=cluster_features,
                                                         fill="toself", name=f"Rata-rata {row['Cluster Name']}"))
                    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                                             title="Profil Atribut Pemain vs Rata-rata Cluster")
                    st.plotly_chart(radar_fig, use_container_width=True)

        else:
            st.write("Masukkan atribut pemain (skala 1–20 khas FM) untuk memprediksi cluster-nya.")
            union_attrs = []
            for lst in [metadata["physical_attrs"], metadata["technical_attrs"],
                        metadata["mental_attrs"], cluster_features]:
                for a in lst:
                    if a not in union_attrs and a in ATTRIBUTE_COLS:
                        union_attrs.append(a)

            input_vals = {}
            n_cols = 4
            cols = st.columns(n_cols)
            for i, attr in enumerate(union_attrs):
                with cols[i % n_cols]:
                    input_vals[attr] = st.slider(attr, 1, 20, 10, key=f"seg_{attr}")

            if st.button("Prediksi Cluster", type="primary"):
                input_df = pd.DataFrame([input_vals])
                pred = seg_model.predict(input_df)
                if pred.empty:
                    st.error("Data tidak cukup untuk memprediksi cluster.")
                else:
                    cname = pred.iloc[0]["Cluster Name"]
                    st.success(f"**Hasil Prediksi Cluster: {cname}**")
                    centroid = seg_model.cluster_profile_summary()
                    st.write("Perbandingan dengan rata-rata (centroid) cluster tersebut:")
                    comp = pd.DataFrame({
                        "Atribut": cluster_features,
                        "Input Anda": [pred.iloc[0][f] for f in cluster_features],
                        f"Rata-rata {cname}": centroid.loc[cname, cluster_features].values,
                    })
                    st.dataframe(comp, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    elif page == "🔍 Cari Pemain Mirip (KNN)":
        st.title("🔍 Cari Pemain Mirip (K-Nearest Neighbors)")
        st.write("Menemukan pemain dengan profil atribut paling mirip menggunakan `knn_model.pkl`.")

        col_a, col_b = st.columns([3, 1])
        with col_a:
            player_name = st.text_input("Nama pemain", value="Kevin De Bruyne")
        with col_b:
            n_similar = st.slider("Jumlah pemain mirip", 3, 20, 10)

        if st.button("Cari Pemain Mirip", type="primary"):
            target_player, results = find_similar_players(
                player_name, df_knn, X_knn, knn_model, knn_features, n=n_similar
            )
            if target_player is None:
                st.error(f"Pemain '{player_name}' tidak ditemukan di dataset (atau atributnya tidak lengkap).")
            else:
                st.info(
                    f"🎯 Pemain acuan: **{target_player['Name']}** — "
                    f"{target_player.get('Position', 'N/A')} | {target_player.get('Club', 'N/A')} | "
                    f"Cluster: {target_player.get('Cluster Name', 'N/A')}"
                )
                st.dataframe(results, use_container_width=True, hide_index=True)

                fig = px.bar(results.sort_values("Similarity %"), x="Similarity %", y="Name",
                             orientation="h", color="Similarity %", color_continuous_scale="Blues",
                             title=f"Tingkat Kemiripan dengan {target_player['Name']}")
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    elif page == "⭐ Rekomendasi Pemain per Role":
        st.title("⭐ Rekomendasi Pemain Berdasarkan Role")
        st.write(
            "Skor akhir merupakan kombinasi tertimbang dari **Role Score**, **Age Score** "
            "(potensi berdasarkan umur), dan **Value Efficiency** (kualitas relatif terhadap harga)."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            role = st.selectbox("Role", list(role_weights.keys()))
            position_filter = st.text_input("Filter posisi (opsional, contoh: 'M', 'D', 'ST')", value="")
        with c2:
            max_age = st.slider("Umur maksimal", 15, 40, 26)
            max_tv_m = st.number_input("Transfer Value maksimal (juta €)", min_value=0.0, value=10.0, step=0.5)
        with c3:
            max_wage_k = st.number_input("Wage maksimal (ribu €/minggu)", min_value=0.0, value=0.0, step=10.0,
                                          help="0 = tidak dibatasi")
            top_n = st.slider("Jumlah rekomendasi", 5, 50, 10)

        st.markdown("**Bobot komponen skor akhir**")
        w1, w2, w3 = st.columns(3)
        w_role = w1.slider("Bobot Role Score", 0.0, 1.0, 0.6, 0.05)
        w_age = w2.slider("Bobot Age Score", 0.0, 1.0, 0.2, 0.05)
        w_ve = w3.slider("Bobot Value Efficiency", 0.0, 1.0, 0.2, 0.05)
        total_w = w_role + w_age + w_ve
        if total_w == 0:
            total_w = 1
        weights = (w_role / total_w, w_age / total_w, w_ve / total_w)

        if st.button("Cari Rekomendasi", type="primary"):
            max_tv = max_tv_m * 1_000_000 if max_tv_m > 0 else np.inf
            max_wage = max_wage_k * 1_000 if max_wage_k > 0 else np.inf
            rec = recommend_players(
                df_fe, role=role,
                position=position_filter if position_filter else None,
                max_age=max_age, max_transfer_value=max_tv, max_wage=max_wage,
                top_n=top_n, weights=weights,
            )
            if rec.empty:
                st.warning("⚠️ Tidak ada pemain yang lolos filter. Coba longgarkan kriteria.")
            else:
                display_rec = rec.copy()
                for col in ["Transfer Value", "Wage"]:
                    if col in display_rec.columns:
                        display_rec[col] = display_rec[col].apply(format_currency)
                st.dataframe(display_rec, use_container_width=True, hide_index=True)

                fig = px.bar(rec.sort_values("Final Score"), x="Final Score", y="Name",
                             orientation="h", color="Final Score", color_continuous_scale="Greens",
                             title=f"Top Rekomendasi untuk Role: {role}")
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    elif page == "📊 Eksplorasi Cluster":
        st.title("📊 Eksplorasi Profil Cluster")

        cluster_features = seg_model.cluster_features
        df_cluster = df_fe.dropna(subset=["Cluster"]).copy()

        centroid = seg_model.cluster_profile_summary()
        st.subheader("Profil Rata-rata (Centroid) Tiap Cluster")
        st.dataframe(centroid.style.background_gradient(cmap="YlGnBu", axis=1).format("{:.1f}"),
                     use_container_width=True)

        st.subheader("Heatmap Profil Cluster")
        fig_heat = px.imshow(
            centroid.T, text_auto=".1f", aspect="auto", color_continuous_scale="YlGnBu",
            labels=dict(x="Cluster", y="Atribut", color="Nilai rata-rata"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Distribusi Posisi per Cluster")
        if "Position" in df_cluster.columns:
            pos_ct = pd.crosstab(df_cluster["Cluster Name"], df_cluster["Position"].str[:4])
            top_positions = df_cluster["Position"].str[:4].value_counts().head(10).index
            pos_ct = pos_ct[[c for c in top_positions if c in pos_ct.columns]]
            fig_pos = px.bar(pos_ct, barmode="stack", title="Distribusi Posisi (Top 10) per Cluster")
            st.plotly_chart(fig_pos, use_container_width=True)

        st.subheader("Top 3 Best Role per Cluster")
        if "Best Role" in df_cluster.columns:
            top_roles = (
                df_cluster.groupby("Cluster Name")["Best Role"]
                .apply(lambda s: s.value_counts(normalize=True).head(3) * 100)
                .round(1)
            )
            st.dataframe(top_roles.rename("Persentase (%)"), use_container_width=True)

        st.subheader("Jelajahi Pemain per Cluster")
        selected_cluster = st.selectbox("Pilih cluster", sorted(df_cluster["Cluster Name"].unique()))
        cols_show = ["Name", "Age", "Position", "Club", "Best Role", "Overall Ability", "Transfer Value"]
        cols_show = [c for c in cols_show if c in df_cluster.columns]
        st.dataframe(
            df_cluster[df_cluster["Cluster Name"] == selected_cluster][cols_show]
            .sort_values("Overall Ability", ascending=False)
            .head(50),
            use_container_width=True, hide_index=True,
        )


if __name__ == "__main__":
    main()
