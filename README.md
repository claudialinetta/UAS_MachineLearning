# MusicType: Pengenalan Instrumen Musik dari Audio

Projek UAS Machine Learning kelompok 10

| NIM | Nama |
|---|---|
| 535250075 | William Richie Heriyanto |
| 535250083 | Sheireen Sadeli |
| 535250098 | Claudia Linetta Widjaja |

## Tentang projek

MusicType adalah aplikasi web untuk mengenali instrumen musik dari sebuah file audio. Pengguna mengunggah lagu atau potongan rekaman, lalu aplikasi menampilkan instrumen apa saja yang terdengar, misalnya piano, gitar, biola, saxophone, drum, dan vokal. Aplikasi ini juga menampilkan visualisasi audio dan perbandingan performa beberapa algoritma.

Ada dua pendekatan yang kami coba:

1. **MFCC + algoritma klasik.** Fitur MFCC (20 koefisien, diambil mean dan standar deviasinya) dipakai untuk melatih Naive Bayes, SVM, KNN, Decision Tree, dan Random Forest. Pendekatan ini hanya menebak satu instrumen dominan per klip.
2. **PANNs (CNN14).** Model pretrained dari AudioSet ini dipakai untuk mendeteksi beberapa instrumen sekaligus dalam satu lagu. Embedding-nya juga kami coba dengan algoritma klasik yang sama sebagai pembanding.

Kami memakai PANNs untuk deteksi multi-instrumen karena lagu biasanya berisi banyak instrumen, sedangkan model MFCC hanya bisa memberi satu label.

## Dataset

[IRMAS](https://zenodo.org/records/1290750) (Instrument Recognition in Musical Audio Signals). Klip pelatihan berdurasi 3 detik dengan label instrumen dominan, lisensi CC BY-NC-SA 3.0.

Kami memakai seluruh 11 kelas: cello, clarinet, flute, gitar akustik, gitar elektrik, organ, piano, saxophone, trumpet, biola, dan vokal. Jumlah klip per kelas tidak sama (sekitar 390 sampai 780 klip), jadi selain akurasi kami juga memakai F1 macro. Datasetnya tidak kami sertakan di repo karena ukurannya besar (sekitar 3 GB), jadi perlu diunduh sendiri dari link di atas.

## Isi repo

| File | Keterangan |
|---|---|
| `01_eksperimen_musictype.ipynb` | Ekstraksi MFCC, eksperimen 5-fold cross validation, confusion matrix, dan penyimpanan model |
| `02_panns_multilabel.ipynb` | Eksperimen embedding PANNs dan evaluasi deteksi multi-instrumen |
| `features.py` | Fungsi ekstraksi fitur MFCC |
| `panns_utils.py` | Fungsi untuk deteksi instrumen dengan PANNs |
| `app.py` | Aplikasi Streamlit |
| `model.joblib` | Model MFCC terbaik hasil training |
| `results_*.csv`, `confusion_matrix.png` | Hasil eksperimen |
| `fitur_mfcc.csv` | Fitur MFCC hasil ekstraksi |
| `requirements.txt` | Daftar library |

## Cara menjalankan

```
pip install -r requirements.txt
streamlit run app.py
```

Saat pertama kali dijalankan, PANNs akan mengunduh checkpoint sekitar 300 MB, jadi perlu koneksi internet. Di Windows, unduhan ini kadang gagal karena memakai `wget`. Kalau terjadi, jalankan lewat Google Colab atau WSL.

Untuk mengulang eksperimen, unduh IRMAS lalu jalankan notebook 01 dan 02 secara berurutan. Kalau memakai Colab, upload `features.py` dan `panns_utils.py` ke sesi Colab terlebih dahulu.

## Hasil

HASIL EKSPERIMEN KLASIFIKASI EMOSI SUARA

Tabel Ringkasan Hasil (MFCC)

| Algoritma       | Fitur | Akurasi | F1 (macro) |
|----------------|--------|----------|------------|
| Naive Bayes    | MFCC   | 0.3664   | 0.3430     |
| SVM (RBF)      | MFCC   | 0.5040   | 0.4930     |
| KNN (k=5)      | MFCC   | 0.4058   | 0.3862     |
| Decision Tree  | MFCC   | 0.2987   | 0.2902     |
| Random Forest  | MFCC   | 0.4953   | 0.4732     |
| (terbaik)      | Embedding PANNs | [isi] | [isi] |


Kesimpulan

Dengan fitur MFCC, algoritma terbaik adalah SVM (RBF) karena memperoleh nilai akurasi tertinggi (0.5040) dan F1 macro tertinggi (0.4930) dibandingkan algoritma lainnya.

## Catatan dan keterbatasan

- Klip pelatihan IRMAS dipotong dari rekaman sumber yang sama, jadi skor cross validation kemungkinan lebih tinggi dari performa sebenarnya. Karena itu kami juga mengevaluasi model pada data uji resmi IRMAS yang berasal dari lagu berbeda.
- Deteksi multi-instrumen bisa meleset pada instrumen yang suaranya tertutup instrumen lain atau yang mirip, seperti saxophone dan klarinet.
- Label uji IRMAS tidak mencakup drum, jadi deteksi drum tampil di aplikasi tetapi tidak terukur di evaluasi.
- Analisis satu lagu di CPU memakan waktu sekitar 1 sampai 2 menit.
