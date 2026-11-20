# MusicType

## Struktur folder

Simpan file-file ini dalam satu folder proyek:

- `01_eksperimen_musictype.ipynb` — eksperimen, evaluasi, dan training model
- `features.py` — ekstraksi fitur MFCC bersama untuk training dan aplikasi
- `app.py` — aplikasi Streamlit
- `requirements.txt` — daftar library Python
- `IRMAS-TrainingData/` — folder dataset training lokal

Di dalam `IRMAS-TrainingData`, pastikan ada subfolder `pia`, `gac`, `vio`, `sax`, `flu`, dan `voi`.

## Menjalankan di VS Code (Windows)

Buka terminal di folder proyek, lalu jalankan:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Buka notebook `01_eksperimen_musictype.ipynb`, pilih kernel `.venv`, lalu jalankan semua sel dari atas ke bawah. Pastikan `model.joblib`, `results_summary.csv`, dan `confusion_matrix.png` berhasil dibuat.

Setelah training selesai, jalankan aplikasi:

```bash
streamlit run app.py
```

## Jika folder dataset berbeda lokasi

Ubah nilai `ROOT` di sel 2 notebook, misalnya:

```python
ROOT = Path(r"D:/Dataset/IRMAS-TrainingData")
```

`ROOT` harus menunjuk ke folder yang langsung berisi subfolder kelas (`pia`, `gac`, dan seterusnya).

## Catatan

- Notebook memakai enam kelas dan maksimal 450 file per kelas.
- Setiap file diubah menjadi 40 fitur: 20 MFCC mean dan 20 MFCC standard deviation.
- Model terbaik dipilih berdasarkan rata-rata F1 macro 5-fold cross validation.
- Aplikasi membagi audio menjadi segmen 3 detik dan membatasi analisis sampai 60 detik.
- Confidence yang ditampilkan adalah probabilitas rata-rata hasil kalibrasi model, bukan jaminan mutlak bahwa prediksi benar.
- Hasil cross validation dapat terpengaruh data leakage jika klip dari rekaman sumber yang sama tersebar ke fold berbeda. Periksa pola nama file sebelum menerapkan group cross validation.
