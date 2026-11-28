# MusicType: Pengenalan Instrumen Musik dari Audio

Projek UAS Machine Learning kelompok 10

| NIM | Nama |
|---|---|
| 535250075 | William Richie Heriyanto |
| 535250083 | Sheireen Sadeli |
| 535250098 | Claudia Linetta Widjaja |

## Tentang projek

MusicType adalah aplikasi web (Streamlit) untuk mengenali instrumen musik dari sebuah file audio. Pengguna mengunggah lagu atau potongan rekaman, lalu aplikasi menampilkan instrumen dominan beserta tingkat keyakinannya, visualisasi audio, dan perbandingan performa beberapa model.

Audio dipotong menjadi segmen berdurasi 3 detik (maksimal 60 detik dianalisis). Setiap segmen diklasifikasikan, lalu probabilitasnya dirata-ratakan untuk menentukan instrumen dominan. Prediksi segmen dengan keyakinan di bawah batas yang bisa diatur (bawaan 40%) ditandai "Tidak yakin".

Ada tiga pendekatan yang bisa dipilih di aplikasi:

1. **SVM.** Memakai 166 fitur audio klasik (MFCC dan fitur spektral lain) dari `features.py`. SVM memakai kernel RBF dengan seleksi fitur `SelectKBest` (k=120) dan C=100, serta dikalibrasi agar menghasilkan probabilitas. Nilai k dan C dipilih lewat penyetelan dengan 5-fold cross validation.
2. **CNN.** EfficientNetB0 pretrained (ImageNet) yang dilatih pada spektrogram 3 channel: mel harmonik, mel perkusif (hasil pemisahan HPSS), dan delta harmonik. Pelatihan dilakukan dua tahap: melatih head saja, lalu fine-tuning 100 layer terakhir.
3. **Ensemble SVM + CNN.** Probabilitas SVM dan CNN dirata-ratakan berbobot (late fusion). Bobot CNN (`w_cnn` = 0.5) dipilih di data validasi, bukan di data test. Tidak ada pelatihan ulang, hanya penggabungan keluaran dua model. Alasannya, SVM dan CNN melihat audio dengan cara berbeda sehingga kesalahan keduanya tidak selalu sama.

## Dataset

[IRMAS](https://zenodo.org/records/1290750) (Instrument Recognition in Musical Audio Signals). Klip pelatihan berdurasi 3 detik dengan label instrumen dominan, lisensi CC BY-NC-SA 3.0.

Kami memakai seluruh 11 kelas: cello, clarinet, flute, gitar akustik, gitar elektrik, organ, piano, saxophone, trumpet, biola, dan vokal. Total 6.705 klip, dengan jumlah per kelas tidak sama (388 sampai 760 klip), jadi selain akurasi kami juga memakai F1 macro. Datasetnya tidak disertakan di repo karena ukurannya besar (sekitar 3 GB), jadi perlu diunduh sendiri dari link di atas.

### Pembagian data

Data dibagi **60% train, 20% validasi, 20% test (1.340 klip)** dengan `StratifiedGroupKFold` berdasarkan grup lagu. Hasilnya 4.031 klip latih, 1.334 validasi, dan 1.340 uji dari 2.273 grup rekaman. Klip yang berasal dari rekaman yang sama selalu berada di split yang sama, sehingga tidak ada kebocoran data antar split. Data validasi dipakai untuk early stopping CNN, memilih antara CNN v1 dan v2, dan memilih bobot ensemble, sedangkan data test hanya untuk evaluasi akhir.

## Isi repo
 
| Lokasi | Keterangan |
|---|---|
| `app.py` | Aplikasi Streamlit |
| `features.py` | Fungsi ekstraksi fitur audio untuk SVM dan pemuatan segmen audio |
| `requirements.txt` | Daftar library |
| `01_eksperimen_musictype.ipynb` | Eksperimen awal: ekstraksi 166 fitur, perbandingan lima algoritma klasik dengan 5-fold cross validation, dan penyetelan SVM (GridSearchCV, SelectKBest, perbandingan kernel) |
| `02_eksperimen_cnn.ipynb` | Eksperimen awal CNN |
| `03_klasifikasi_instrumen_colab.ipynb` | Split 60/20/20, CNN v1 dan v2, SVM, algoritma klasik, dan ensemble. Menghasilkan semua file di `models/` dan `results/` |
| `models/` | Model yang dipakai aplikasi: `svm_calibrated.joblib`, `ensemble_config.joblib`, `best_irmas_hpss_efficientnet_v2.keras`, `cnn_hpss_preprocessing.joblib` |
| `results/` | Hasil evaluasi di data test: `hasil_evaluasi_test.csv`, laporan klasifikasi per kelas, dan confusion matrix SVM, CNN, serta ensemble |
| `archive/` | File hasil eksperimen awal yang sudah tidak dipakai aplikasi |
 
## Cara menjalankan
 
```
pip install -r requirements.txt
streamlit run app.py
```
 
Aplikasi membutuhkan semua file di folder `models/` dan `results/`.
 
## Hasil
 
Evaluasi pada data test (20% dari split berbasis grup lagu, 1.340 klip). Hasil per kelas ada di folder `results/`.
 
| Model | Akurasi | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| Naive Bayes | 0.3515 | 0.3870 | 0.3653 | 0.3389 |
| Decision Tree | 0.3134 | 0.3050 | 0.3070 | 0.3050 |
| KNN (k=5) | 0.4313 | 0.4256 | 0.4285 | 0.4139 |
| Random Forest | 0.4925 | 0.5165 | 0.4865 | 0.4812 |
| CNN v2 (model CNN final) | 0.5336 | 0.5636 | 0.5246 | 0.5076 |
| CNN v1 (baseline) | 0.5448 | 0.5553 | 0.5376 | 0.5277 |
| SVM (RBF, terkalibrasi) | 0.5731 | 0.5703 | 0.5643 | 0.5637 |
| **Ensemble SVM + CNN** | **0.6239** | **0.6271** | **0.6149** | **0.6124** |
 
Precision, recall, dan F1 memakai rata-rata macro (semua kelas dihitung sama rata).
 
## Kesimpulan
 
- Di antara algoritma klasik, SVM (RBF) paling baik, diikuti Random Forest. Naive Bayes dan Decision Tree paling lemah.
- SVM sedikit mengungguli CNN pada data test ini. Peringkat CNN v1 dan v2 berganti antar pelatihan (v2 lebih baik di pelatihan pertama, v1 sedikit lebih baik di pelatihan akhir), jadi keduanya dianggap setara. CNN v2 dipakai sebagai model CNN final berdasarkan perbandingan di data validasi pada pelatihan pertama.
- **Ensemble SVM + CNN memberi hasil terbaik**, sekitar 5 poin akurasi di atas SVM dan sekitar 9 poin di atas CNN. Ini menunjukkan bahwa fitur audio klasik dan spektrogram saling melengkapi.

## Catatan dan keterbatasan
 
- Hasil berasal dari satu kali pembagian data dan satu kali pelatihan CNN, jadi selisih kecil antar model tidak dapat dianggap pasti. Dua kali pelatihan CNN v1 dengan konfigurasi yang sama pernah menghasilkan akurasi test 0,5097 dan 0,5448.
- Nilai k = 120 dan C = 100 dipilih lewat cross validation pada seluruh data (termasuk rekaman yang kemudian menjadi data test), dan hanya SVM yang disetel. Perbandingan karenanya cenderung menguntungkan SVM.
- Kalibrasi SVM memakai `cv=5` biasa yang belum memperhatikan grup lagu, sehingga probabilitasnya bisa sedikit terlalu percaya diri.
- IRMAS TestingData resmi belum dievaluasi.
- Akurasi terbaik sekitar 62%. Saxophone, flute, dan biola masih sering salah dikenali (recall ensemble untuk ketiganya di bawah 50%).
- Aplikasi menampilkan satu instrumen dominan per segmen, jadi bukan deteksi semua instrumen yang berbunyi bersamaan (belum multi-label).
- Analisis satu lagu di CPU membutuhkan waktu beberapa saat karena CNN dijalankan di setiap segmen.
