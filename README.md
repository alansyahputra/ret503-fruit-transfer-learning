# Fruit Image Classification using Transfer Learning

## Deskripsi Proyek

Proyek ini merupakan implementasi klasifikasi citra buah menggunakan metode *Transfer Learning* dengan arsitektur EfficientNet-B0.

Tujuan proyek ini adalah membandingkan performa model berdasarkan tiga pendekatan pelatihan, yaitu Feature Extraction, Partial Fine-Tuning, dan Training from Scratch.

Eksperimen dilakukan menggunakan Google Colab dengan framework PyTorch. Evaluasi model menggunakan nilai akurasi pada data validasi dan data pengujian (*test*).

## Teknologi yang Digunakan

- Python
- PyTorch
- Torchvision
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
- EfficientNet-B0

## Struktur Folder

```text
ret503-fruit-transfer-learning/
├── dataset_raw/
├── docs/
├── results/
│   ├── perbandingan_akurasi_efficientnet_b0.png
│   ├── tabel_efficientnet_b0.csv
│   └── tabel_efficientnet_b0.md
├── scripts/
│   ├── make_crops.py
│   ├── report.py
│   └── train.py
└── README.md
```

**Keterangan folder:**

- `dataset_raw/`: Menyimpan dataset asli.
- `docs/`: Menyimpan dokumentasi proyek.
- `results/`: Menyimpan hasil eksperimen, grafik, dan tabel evaluasi.
- `scripts/`: Berisi program untuk pemrosesan data, pelatihan model, dan pembuatan laporan.
- `README.md`: Dokumentasi utama proyek.

## Metode Eksperimen

Tiga pendekatan pelatihan yang digunakan adalah:

### 1. Feature Extraction

Model EfficientNet-B0 menggunakan bobot pretrained. Lapisan ekstraksi fitur dipertahankan, sedangkan bagian klasifikasi dilatih untuk mengenali kelas buah pada dataset.

### 2. Partial Fine-Tuning

Sebagian lapisan model pretrained dilatih kembali agar fitur yang telah dipelajari dapat disesuaikan dengan dataset yang digunakan.

### 3. Training from Scratch

Model dilatih tanpa menggunakan bobot pretrained sebagai pembanding terhadap pendekatan transfer learning.

## Hasil Eksperimen

Eksperimen awal menggunakan EfficientNet-B0 menghasilkan nilai akurasi berikut:

| Metode | Seed | Akurasi Validasi | Akurasi Test |
|---|---:|---:|---:|
| Feature Extraction | 0 | 96,1% | 89,8% |
| Feature Extraction | 1 | 93,2% | 89,8% |
| Feature Extraction | 2 | 94,2% | 87,8% |
| Partial Fine-Tuning | 0 | 95,1% | 92,9% |
| Partial Fine-Tuning | 1 | 94,2% | 94,9% |
| Partial Fine-Tuning | 2 | 93,2% | 91,8% |
| Training from Scratch | 0 | 81,6% | 75,5% |
| Training from Scratch | 1 | 81,6% | 73,5% |
| Training from Scratch | 2 | 77,7% | 75,5% |

*Catatan: Hasil di atas merupakan hasil eksperimen yang tercatat pada proses pengujian. Nilai tersebut bukan jaminan performa pada dataset atau kondisi pengujian yang berbeda.*

### Analisis Hasil

Berdasarkan hasil pengujian, Partial Fine-Tuning memperoleh akurasi test tertinggi sebesar 94,9% pada seed 1.

Feature Extraction juga memberikan hasil yang baik, dengan akurasi test berkisar antara 87,8% hingga 89,8%.

Sementara itu, Training from Scratch menghasilkan akurasi test yang lebih rendah, yaitu antara 73,5% hingga 75,5%.

Hasil ini menunjukkan bahwa penggunaan bobot pretrained dapat membantu meningkatkan performa klasifikasi pada eksperimen ini. Namun, hasil tersebut tetap bergantung pada dataset, konfigurasi pelatihan, dan proses evaluasi yang digunakan.

## Cara Menjalankan Proyek

### 1. Persiapan Environment

Proyek dapat dijalankan menggunakan Google Colab atau lingkungan Python yang telah dilengkapi dengan dependensi yang diperlukan.

Pastikan PyTorch, Torchvision, Pandas, Matplotlib, dan pustaka pendukung lainnya telah tersedia.

### 2. Persiapan Dataset

Siapkan dataset citra buah dengan struktur folder dan format anotasi yang sesuai dengan implementasi program pada `scripts/train.py`.

Pastikan lokasi dataset sesuai dengan konfigurasi pada program sebelum memulai pelatihan.

### 3. Menjalankan Pelatihan

Jalankan perintah berikut dari direktori utama proyek:

```bash
python scripts/train.py --modes feature partial scratch --seeds 0 1 2
```

Perintah tersebut menjalankan tiga pendekatan pelatihan dengan tiga variasi seed.

Untuk menjalankan Feature Extraction saja:

```bash
python scripts/train.py --modes feature --seeds 0
```

### 4. Membuat Laporan Hasil

Jalankan perintah berikut untuk membuat laporan eksperimen:

```bash
python scripts/report.py --model efficientnet_b0
```

Pastikan dataset, dependensi, dan file yang dibutuhkan telah tersedia sebelum menjalankan program.

## Hasil dan Dokumentasi

Hasil eksperimen disimpan dalam folder `results/`, meliputi:

- Grafik perbandingan akurasi.
- Tabel hasil pengujian dalam format CSV.
- Tabel ringkasan hasil eksperimen dalam format Markdown.

File tersebut dapat digunakan untuk menganalisis performa masing-masing metode pelatihan.

## Kesimpulan

Berdasarkan eksperimen yang dilakukan, Partial Fine-Tuning memperoleh akurasi test tertinggi sebesar 94,9%, sedangkan Feature Extraction memberikan performa yang relatif konsisten.

Training from Scratch memperoleh hasil yang lebih rendah dibandingkan kedua pendekatan transfer learning tersebut.

Dengan demikian, transfer learning menggunakan EfficientNet-B0 menjadi pendekatan yang menjanjikan untuk tugas klasifikasi citra buah pada eksperimen ini.

## Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan meliputi:

- Menambahkan evaluasi menggunakan dataset pengujian yang lebih beragam.
- Membandingkan arsitektur model lainnya.
- Melakukan analisis confusion matrix untuk mengetahui kesalahan klasifikasi setiap kelas.
- Mengembangkan aplikasi sederhana untuk melakukan prediksi kelas buah dari gambar yang diunggah.

## Author

**Alan Syahputra**

GitHub: [@alansyahputra](https://github.com/alansyahputra)

---

*Project: Fruit Image Classification using Transfer Learning with EfficientNet-B0*
