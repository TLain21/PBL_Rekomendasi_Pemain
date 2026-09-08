"""
Modul deployment untuk segmentasi pemain FM2023.
Didefinisikan sebagai file terpisah (bukan didefinisikan langsung di dalam
notebook) agar kelas ini bisa di-import dengan aman saat model .pkl dimuat
ulang (baik di notebook lain maupun di server/aplikasi deployment) — kelas
yang hanya didefinisikan inline di sebuah notebook TIDAK bisa di-unpickle
di proses/file lain.
"""
import pandas as pd


class PlayerSegmentationModel:
    """
    Pipeline deployment untuk segmentasi pemain FM2023.

    Menerima dataframe pemain mentah (kolom atribut 1-20 khas FM) dan mengembalikan
    dataframe yang sama ditambah kolom 'Cluster' dan 'Cluster Name'.
    """

    def __init__(self, scaler, kmeans, cluster_features,
                 physical_attrs, technical_attrs, mental_attrs, cluster_names):
        self.scaler = scaler
        self.kmeans = kmeans
        self.cluster_features = cluster_features
        self.physical_attrs = physical_attrs
        self.technical_attrs = technical_attrs
        self.mental_attrs = mental_attrs
        self.cluster_names = cluster_names

    def _engineer(self, df):
        df = df.copy()
        phys = [c for c in self.physical_attrs if c in df.columns]
        tech = [c for c in self.technical_attrs if c in df.columns]
        ment = [c for c in self.mental_attrs if c in df.columns]
        if phys:
            df['Physical Score'] = df[phys].mean(axis=1)
        if tech:
            df['Technical Score'] = df[tech].mean(axis=1)
        if ment:
            df['Mental Score'] = df[ment].mean(axis=1)
        return df

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Prediksi cluster & nama cluster untuk data pemain baru."""
        df = self._engineer(df_raw)
        missing = [c for c in self.cluster_features if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom berikut dibutuhkan tapi tidak ada: {missing}")

        df_valid = df.dropna(subset=self.cluster_features).copy()
        X_new = self.scaler.transform(df_valid[self.cluster_features])
        df_valid['Cluster'] = self.kmeans.predict(X_new)
        df_valid['Cluster Name'] = df_valid['Cluster'].map(self.cluster_names)
        return df_valid

    def cluster_profile_summary(self) -> pd.DataFrame:
        """Ringkasan pusat (centroid) tiap cluster dalam skala asli fitur."""
        centers_scaled = self.kmeans.cluster_centers_
        centers_original = self.scaler.inverse_transform(centers_scaled)
        return pd.DataFrame(centers_original, columns=self.cluster_features,
                            index=[self.cluster_names.get(i, i) for i in range(len(centers_original))])