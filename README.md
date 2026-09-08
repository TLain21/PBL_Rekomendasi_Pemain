# FM2023 Player Segmentation & Recommendation — Streamlit App

## Isi folder
- `app.py` — aplikasi Streamlit utama
- `requirements.txt` — dependency Python
- `segmentation_model.py` — definisi class `PlayerSegmentationModel` (wajib ada, dipakai untuk unpickle `segmentation_model.pkl`)
- `segmentation_model.pkl`, `kmeans_model.pkl`, `scaler_clustering.pkl`, `knn_model.pkl`, `scaler_knn.pkl`, `metadata.pkl` — model hasil training
- `fm2023.csv` — dataset mentah (dipakai app untuk membangun tabel pemain, fitur KNN, dan skor role secara runtime)

## Cara menjalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Push seluruh isi folder ini (termasuk file .pkl dan fm2023.csv) ke sebuah repo GitHub.
   - Total ukuran ±109 MB, masih di bawah limit repo GitHub (soft limit 1GB, warning di atas 50MB/file — fm2023.csv ±63MB dan knn_model.pkl ±49MB, sebaiknya gunakan **Git LFS** untuk kedua file ini agar aman.
2. Buka https://share.streamlit.io → New app → pilih repo tsb → main file path: `app.py`.
3. Deploy. Proses pertama kali membuka app akan memproses ±189 ribu baris data (cleaning + feature engineering + assign cluster), memakan waktu ±10-20 detik, tapi hasilnya di-cache (`st.cache_data`/`st.cache_resource`) sehingga hanya berjalan sekali per sesi server.

## Catatan penting
- `app.py` MEREPLIKASI persis pipeline cleaning & feature engineering dari notebook
  (`clean_fm_dataset`, `engineer_features`, `compute_role_score`, dst.) sehingga urutan baris
  dataset hasil olahan identik dengan yang dipakai saat melatih `knn_model.pkl` — ini penting
  karena `knn_model.pkl` hanya menyimpan struktur pohon jarak (tanpa nama pemain), jadi index
  hasil pencarian tetangga harus dipetakan balik ke dataframe yang disusun dengan urutan yang sama.
- Model clustering (`segmentation_model.pkl`) dan KNN (`knn_model.pkl`, `scaler_knn.pkl`) dipakai
  langsung dalam mode **predict/transform** (tidak di-fit ulang) agar konsisten dengan hasil training
  di notebook.
