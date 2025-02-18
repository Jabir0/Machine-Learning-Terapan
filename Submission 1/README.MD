
---

## Laporan Proyek Machine Learning - Jabir Muktabir

---

## Domain Proyek
Pada era persaingan ketat di industri ponsel, perusahaan baru sulit bersaing dengan merek ternama seperti Apple dan Samsung. Estimasi harga yang tepat menjadi faktor kunci dalam menentukan daya saing produk. Bob, seorang pengusaha yang baru memulai bisnis ponsel, ingin mengatasi tantangan ini dengan menemukan hubungan antara fitur ponsel (seperti RAM, memori internal, dll.) dan rentang harga jual.

**Mengapa masalah ini penting?**  
- Estimasi harga yang akurat dapat membantu dalam **strategi pemasaran** dan **penentuan segmentasi pasar** yang tepat.  
- Meminimalkan **kerugian finansial** akibat salah penetapan harga.  

**Referensi**:  
- [K-BEST SELECTION UNTUK MENINGKATKAN KINERJA ARTIFICIAL NEURAL NETWORK DALAM MEMPREDIKSI RANGE HARGA PONSEL](https://ejournal.nusamandiri.ac.id/index.php/inti/article/view/5554)  
- [ANALISIS PENGARUH CITRA MERK, KUALITAS PRODUK, HARGA DAN PROMOSI TERHADAP KEPUTUSAN PEMBELIAN SMARTPHONE SAMSUNG DI MAKASSAR](https://ojs.unsulbar.ac.id/index.php/mandar/article/view/864)

---

## Business Understanding

### Problem Statements
1. **Bagaimana menentukan rentang harga ponsel berdasarkan fitur-fitur teknis?**  
2. **Algoritma machine learning apa yang paling efektif untuk mengklasifikasikan rentang harga ponsel?**  
3. **Bagaimana performa model dalam hal akurasi, presisi, recall, dan F1-Score pada data ini?**  

---

### Goals
- **Menentukan hubungan antara fitur teknis ponsel dan rentang harga.**  
- **Membangun model klasifikasi yang akurat untuk mengelompokkan harga ponsel.**  
- **Mengevaluasi performa model menggunakan metrik akurasi, presisi, recall, dan F1-Score.**  

---

### Solution Statements
- Menggunakan **tiga algoritma klasifikasi**:  
  1. **CatBoost**: Algoritma boosting yang kuat dalam menangani fitur kategori dan **overfitting yang rendah**.  
  2. **AdaBoost**: Algoritma boosting adaptif yang cocok untuk **imbalance data**.  
  3. **Random Forest**: Algoritma ensemble dengan **interpretabilitas yang baik** dan **stabil terhadap outlier**.  
- **Hyperparameter tuning** pada masing-masing model untuk **memaksimalkan performa**.  
- **Metrik evaluasi yang digunakan**: Akurasi, Presisi, Recall, dan F1-Score.

---

## Data Understanding
Dataset yang digunakan adalah **Mobile Price Classification Dataset**, yang berisi informasi mengenai spesifikasi teknis ponsel dan rentang harga.  
Dataset dapat diakses di: [Mobile Price Classification Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

### Variabel-variabel pada dataset:
- **battery_power** = Total energi yang dapat disimpan oleh baterai dalam satu kali pengisian daya, diukur dalam mAh  
- **blue** = Apakah memiliki fitur Bluetooth atau tidak  
- **clock_speed** = Kecepatan prosesor dalam mengeksekusi instruksi  
- **dual_sim** = Apakah mendukung dual SIM atau tidak  
- **fc** = Megapiksel kamera depan  
- **four_g** = Apakah mendukung 4G atau tidak  
- **int_memory** = Memori internal dalam Gigabyte  
- **m_dep** = Ketebalan ponsel dalam cm  
- **mobile_wt** = Berat ponsel  
- **n_cores** = Jumlah inti (core) pada prosesor  
- **pc** = Megapiksel kamera utama  
- **px_height** = Resolusi tinggi piksel layar  
- **px_width** = Resolusi lebar piksel layar  
- **ram** = Random Access Memory dalam Megabyte  
- **sc_h** = Tinggi layar ponsel dalam cm  
- **sc_w** = Lebar layar ponsel dalam cm  
- **talk_time** = Waktu terlama yang bisa dicapai dengan satu kali pengisian baterai saat digunakan untuk panggilan  
- **three_g** = Apakah mendukung 3G atau tidak  
- **touch_screen** = Apakah memiliki layar sentuh atau tidak  
- **wifi** = Apakah memiliki fitur WiFi atau tidak  
- **price_range** = Variabel target dengan nilai: 0 (biaya rendah), 1 (biaya menengah), 2 (biaya tinggi), dan 3 (biaya sangat tinggi)  

---

### Exploratory Data Analysis (EDA)
Beberapa teknik EDA yang digunakan: 
- **Metode IQR** : Menggunakan Boxplot untuk memahami distribusi dan keberadaan outlier pada fitur numerik

<div style="text-align: center;">
  <h4>Visualisasi Menggunakan Boxplot</h4>
  <img src="C:\SEMESTER 5_Bangkit\Machine Learning Terapan\images\boxplot.png" alt="Visualisasi Menggunakan Boxplot" style="width: 100%; margin-bottom: 10px;"/>
</div>

- **Distribusi numerik**: Menggunakan histogram untuk melihat distribusi fitur numerik. 
- **Korelasi antar fitur**: Menggunakan correlation matrix untuk menganalisis korelasi antar fitur.  
- **Distribusi kategori**: Menggunakan bar plot untuk melihat distribusi fitur kategori.  

Visualisasi sudah diperindah menggunakan palet warna yang konsisten, transparansi yang lembut, dan **grid horizontal** untuk pembacaan yang lebih mudah.

---

## Data Preparation
Teknik data preparation yang diterapkan:  
- **Handling missing values**: Tidak ada nilai yang hilang dalam dataset.  
- **Normalisasi fitur numerik**: Menggunakan **Min-Max Scaling** untuk fitur    
- **Split data**: Membagi dataset menjadi **90% data latih** dan **10% data uji** dengan `train_test_split`.  

**Alasan**:  
- **Normalisasi** membantu **mempercepat konvergensi model** dan **mengurangi bias skala** pada algoritma berbasis jarak.  

---

## Modeling
Menggunakan tiga model klasifikasi:
1. **CatBoost**:  
   - **Kelebihan**: Handling fitur kategori secara otomatis, overfitting rendah.  
   - **Hyperparameter tuning**: `learning_rate`, `depth`, `iterations`.  
2. **AdaBoost**:  
   - **Kelebihan**: Meningkatkan performa model lemah, stabil terhadap noise.  
   - **Hyperparameter tuning**: `n_estimators`, `learning_rate`.  
3. **Random Forest**:  
   - **Kelebihan**: Robust terhadap outlier, overfitting lebih rendah pada dataset besar.  
   - **Hyperparameter tuning**: `n_estimators`, `max_depth`, `min_samples_split`.  

---

## Evaluation
### Metrik Evaluasi
- **Akurasi**: Persentase prediksi yang benar dari keseluruhan prediksi.  
  \[
  Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
  \]
- **Precision**: Persentase prediksi positif yang benar dari semua prediksi positif.  
  \[
  Precision = \frac{TP}{TP + FP}
  \]
- **Recall (Sensitivity)**: Persentase data positif yang terdeteksi dengan benar.  
  \[
  Recall = \frac{TP}{TP + FN}
  \]
- **F1-Score**: Harmonic mean dari Precision dan Recall.  
  \[
  F1\text{-}Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}
  \]

### Hasil Evaluasi
- **CatBoost**: Akurasi = 88%, Precision = 87%, Recall = 89%, F1-Score = 88%.  
- **AdaBoost**: Akurasi = 82%, Precision = 80%, Recall = 81%, F1-Score = 80%.  
- **Random Forest**: Akurasi = 89%, Precision = 87%, Recall = 90%, F1-Score = 88%.  

**Model Terbaik**: **Random Forest** dipilih karena memiliki akurasi dan recall tertinggi, serta stabilitas yang baik terhadap noise.

---

**---Ini adalah bagian akhir laporan---**

---