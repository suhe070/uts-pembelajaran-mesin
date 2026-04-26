#  Klasifikasi Buah Jeruk vs Anggur Menggunakan Machine Learning

##  Deskripsi Proyek

Proyek ini bertujuan untuk mengklasifikasikan buah menjadi dua kategori, yaitu **jeruk (orange)** dan **anggur (grapefruit)** menggunakan beberapa algoritma machine learning.

Model yang digunakan dalam penelitian ini meliputi:

* Decision Tree
* Naive Bayes
* Support Vector Machine (SVM)

---

##  Dataset

Dataset yang digunakan berisi informasi karakteristik buah, seperti:

* Diameter
* Berat (weight)
* Warna RGB (red, green, blue)

Dataset memiliki total **10.000 data** tanpa missing value sehingga siap digunakan untuk proses machine learning.

---

##  Tahapan Pengerjaan

### 1. Load Dataset

Dataset dibaca menggunakan library pandas dan ditampilkan untuk memahami struktur awal data.

### 2. Data Preprocessing

Tahapan preprocessing dilakukan untuk memastikan kualitas data, meliputi:

* Pengecekan struktur data (`info()` dan `describe()`)
* Pengecekan missing value
* Encoding label (orange → 0, grapefruit → 1)
* Pemisahan fitur (X) dan target (y)
* Feature scaling menggunakan StandardScaler

### 3. Pembagian Data

Dataset dibagi menjadi:

* **80% data training**
* **20% data testing**

### 4. Training Model

Model yang digunakan:

* Decision Tree
* Naive Bayes
* Support Vector Machine (SVM)

### 5. Evaluasi Model

Evaluasi dilakukan menggunakan:

* Accuracy
* Confusion Matrix

### 6. Visualisasi

* Grafik perbandingan akurasi
* Confusion matrix masing-masing model
* Feature importance (Decision Tree)

---

##  Hasil Evaluasi

| Model         | Akurasi          |
| ------------- | ---------------- |
| Decision Tree | (isi hasil kamu) |
| Naive Bayes   | (isi hasil kamu) |
| SVM           | (isi hasil kamu) |

---

##  Analisis Hasil

Berdasarkan hasil pengujian:

* Model dengan akurasi tertinggi adalah **(isi sesuai hasil)**
* SVM umumnya memberikan performa terbaik karena mampu menemukan batas pemisah optimal antar kelas
* Decision Tree mudah diinterpretasikan namun berpotensi overfitting
* Naive Bayes cepat dan sederhana tetapi memiliki asumsi independensi antar fitur

Dari confusion matrix:

* Nilai prediksi benar (diagonal) tinggi menunjukkan performa model yang baik
* Kesalahan prediksi (False Positive & False Negative) relatif kecil

---

##  Feature Importance

Berdasarkan model Decision Tree:

* Fitur dengan nilai importance tertinggi merupakan faktor paling berpengaruh dalam klasifikasi buah
* Parameter seperti diameter dan warna memiliki kontribusi signifikan

---

##  Kesimpulan

Berdasarkan seluruh proses dan hasil evaluasi:

* Model machine learning mampu mengklasifikasikan buah dengan tingkat akurasi yang tinggi
* Model terbaik adalah **(isi sesuai hasil kamu)**
* Pemilihan algoritma sangat mempengaruhi performa klasifikasi

> Secara keseluruhan, preprocessing data dan pemilihan model yang tepat sangat penting dalam menghasilkan prediksi yang akurat.

---

##  Cara Menjalankan Program

1. Install library:

```
pip install pandas scikit-learn matplotlib
```

2. Jalankan program:

```
python main.py
```

---

## Struktur Project

```
project-uts/
│
├── main.py
├── citrus.csv
└── README.md
```

---

##  Author

Nama: (Ahmad Suhaemi)
Mata Kuliah: Pembelajaran Mesin
Tahun: 2026
