# Laporan Proyek Machine Learning - Jabir Muktabir

## Project Overview

Sistem rekomendasi telah menjadi bagian penting dalam berbagai industri, terutama dalam bidang e-commerce dan hiburan. Dengan semakin meningkatnya jumlah buku yang tersedia secara daring, pengguna sering kali kesulitan menemukan buku yang sesuai dengan preferensi mereka. Oleh karena itu, Sistem rekomendasi buku menjadi solusi untuk membantu pengguna menemukan buku yang relevan berdasarkan data historis dan perilaku mereka **[1]**.


Referensi :
- [**[1]**. T. Monisya Afriyanti and E. Retnoningsih, “Sistem Rekomendasi Buku Perpustakaan Menggunakan Algoritma Frequent Pattern Growth Library Book Recommendation System using Frequent Pattern Growth Algorithm,” Techno.COM, vol. 21, no. 2, pp. 292–310, May 2022.](https://core.ac.uk/download/pdf/521875503.pdf)

## Business Understanding

### Problem Statements
1. Bagaimana cara memberikan rekomendasi buku yang relevan bagi pengguna berdasarkan rating yang diberikan?
2. Bagaimana meningkatkan pengalaman pengguna dengan memberikan rekomendasi yang dipersonalisasi?

### Goals
1. Mengembangkan sistem rekomendasi yang dapat menyarankan buku berdasarkan rating dan preferensi pengguna.
2. Menganalisis dan membandingkan beberapa metode rekomendasi untuk menemukan pendekatan terbaik.

### **Solution Approach**  

#### **Solution Statements**  
Untuk membangun sistem rekomendasi buku yang efektif, pendekatan berikut digunakan:  

1. **Content-Based Filtering dengan TF-IDF**  
   - Model ini menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengekstrak fitur dari metadata buku (judul, penulis, dan penerbit).  
   - Kesamaan antar buku dihitung menggunakan **Cosine Similarity** untuk merekomendasikan buku yang paling mirip dengan buku yang telah dinilai pengguna.  

2. **Collaborative Filtering dengan Recommender Net**  
   - Menggunakan model **Neural Collaborative Filtering (NCF)** berbasis **Recommender Net**, yang memanfaatkan **embedding layer** untuk pengguna dan buku.  
   - Model ini dilatih untuk memprediksi rating buku berdasarkan interaksi pengguna dan menghasilkan rekomendasi personal berdasarkan pola preferensi pengguna lainnya.  

3. **Evaluasi Model dengan Metrik yang Sesuai**  
   - **Content-Based Filtering** dievaluasi menggunakan **Cosine Similarity** untuk mengukur kesamaan antara buku-buku rekomendasi.  
   - **Collaborative Filtering** dievaluasi menggunakan **Root Mean Squared Error (RMSE)** untuk mengukur akurasi prediksi rating yang diberikan oleh model.  
---

## **Data Understanding** 

Dataset yang digunakan dalam proyek ini berasal dari [Kaggle - Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset), yang dikumpulkan dari komunitas Book-Crossing pada tahun 2004. Dataset ini terdiri dari tiga bagian utama: data pengguna, data buku, dan data rating buku.

### **1. Users.csv**  
Berisi informasi tentang pengguna yang memberikan rating buku. Dataset ini memiliki **278.858 entri** dengan tiga kolom utama:  
- **`User-ID`**: ID unik pengguna (tidak memiliki nilai yang hilang).  
- **`Location`**: Lokasi pengguna dalam format teks.  
- **`Age`**: Usia pengguna (banyak nilai yang kosong, hanya tersedia untuk **168.096 entri**).  

### **2. Books.csv**  
Berisi informasi tentang buku yang tersedia dalam dataset, dengan **271.360 entri** dan delapan kolom utama:  
- **`ISBN`**: Identifikasi unik untuk setiap buku (tidak memiliki nilai yang hilang).  
- **`Book-Title`**: Judul buku.  
- **`Book-Author`**: Nama penulis buku (**2 nilai kosong**).  
- **`Year-Of-Publication`**: Tahun buku diterbitkan.  
- **`Publisher`**: Nama penerbit (**2 nilai kosong**).  
- **`Image-URL-S`**, **`Image-URL-M`**, **`Image-URL-L`**: URL gambar sampul buku dalam berbagai ukuran (**1 nilai kosong untuk `Image-URL-L`**).  

### **3. Ratings.csv**  
Berisi data rating buku yang diberikan oleh pengguna dengan total **1.149.780 entri**, yang terdiri dari tiga kolom utama:  
- **`User-ID`**: ID pengguna yang memberikan rating.  
- **`ISBN`**: Identifikasi unik buku yang diberi rating.  
- **`Book-Rating`**: Skor rating buku dalam skala **1-10** (rating eksplisit) dan **0** untuk indikasi bahwa pengguna memiliki buku tersebut tetapi tidak memberikan rating eksplisit (rating implisit).  

---

![univariate_analysis](https://github.com/user-attachments/assets/f0187177-48be-4730-a119-ecb7319d0029)

## **Insight dari Data**  
1. Dari total **278.858 pengguna**, hanya **1.149.780 entri** yang berisi rating buku, menunjukkan bahwa tidak semua pengguna memberikan rating.  
2. Dari **271.360 buku** yang terdaftar dalam dataset, sebanyak **340.556 buku** telah menerima rating. Perbedaan jumlah ini kemungkinan disebabkan oleh adanya buku yang tercatat dalam data rating tetapi tidak terdaftar dalam data buku.  
3. Terdapat total **102.024 penulis** yang terdaftar dalam dataset.  
4. Skala rating buku berkisar antara **1 hingga 10**.  
5. Terdapat perbedaan jumlah buku dan ISBN, yang kemungkinan disebabkan oleh **keberadaan beberapa versi berbeda dari buku yang sama** (misalnya edisi berbeda, cetakan ulang, atau format yang berbeda seperti paperback dan hardcover).  

---

## **Data Preparation**  

Sebelum data digunakan untuk membangun model rekomendasi, dilakukan beberapa langkah **persiapan data** untuk memastikan kualitas dan efisiensi pemrosesan.  

### **1. Data Merging**  
Langkah pertama adalah **menggabungkan (merging) dataset** untuk menghubungkan data rating dengan informasi buku. Penggabungan dilakukan antara dataset **ratings** dan **books** berdasarkan **ISBN** agar setiap rating memiliki informasi tambahan seperti **judul buku, penulis, dan penerbit**.  

Kode untuk melakukan merging:  
```python
data_rating_buku = pd.merge(
    ratings,
    books,
    on = 'ISBN',
    how = 'left'
)
```
Metode **left join** digunakan agar semua entri rating tetap dipertahankan meskipun beberapa ISBN mungkin tidak ditemukan dalam dataset buku.  

---

### **2. Handling Missing Values**  
Dataset yang digunakan memiliki beberapa kolom dengan **nilai yang hilang (missing values)**. Untuk memastikan kualitas data, dilakukan **penghapusan** entri yang memiliki nilai kosong pada kolom **judul buku (Book-Title)** dan **penulis (Book-Author)**.  

```python
data_rating_buku = data_rating_buku.dropna()
```

---

### **3. Handling Duplicate Data**  
Beberapa data memiliki **duplikasi (duplicate entries)** yang dapat memengaruhi akurasi model rekomendasi. Oleh karena itu, dilakukan **penghapusan duplikasi** berdasarkan **ISBN** dan **judul buku (Book-Title)** untuk memastikan setiap buku hanya direpresentasikan satu kali.  

```python
# Menghitung jumlah duplikasi
duplicate_count_isbn = data_rating_buku['ISBN'].duplicated().sum()
duplicate_count_title = data_rating_buku['Book-Title'].duplicated().sum()

print(f"Jumlah duplikasi di kolom ISBN: {duplicate_count_isbn}")
print(f"Jumlah duplikasi di kolom Judul Buku: {duplicate_count_title}")

# Menghapus data yang duplikat
data_rating_buku = data_rating_buku.drop_duplicates(subset='ISBN')
data_rating_buku = data_rating_buku.drop_duplicates(subset='Book-Title')
```

---

### **4. Data Selection**  
Dataset ini memiliki **241.065 entri**, yang cukup besar untuk diproses. Untuk efisiensi dan keterbatasan penyimpanan di **Kaggle**, hanya digunakan **30.000 entri teratas**, yang dipilih berdasarkan jumlah rating yang diberikan oleh pengguna.  

```python
data_rating_buku = data_rating_buku.head(30000)
```

---

## **Data Preparation - Collaborative Filtering**  

Pendekatan pertama dalam sistem rekomendasi ini menggunakan **Collaborative Filtering**. Sebelum membangun model, dilakukan beberapa proses tambahan:

### **5. Encoding dan Mapping**  
Agar model dapat memproses data dengan lebih efisien, dilakukan **pengonversian data ke format numerik**.  
- **User-ID** dan **Book-Title** dikonversi ke indeks numerik menggunakan dictionary mapping.  

```python
# Mendapatkan daftar user ID yang unik
user_ids = data_rating_buku['User-ID'].unique().tolist()

# Membuat encoding user ID
encode_user_id1 = {user_id: index for index, user_id in enumerate(user_ids)}
encoded_user_id2 = {index: user_id for index, user_id in enumerate(user_ids)}

# Mendapatkan daftar judul buku yang unik
titles = data_rating_buku['Book-Title'].unique().tolist()

# Membuat encoding untuk judul buku
encode_title1 = {title: index for index, title in enumerate(titles)}
encoded_title2 = {index: title for index, title in enumerate(titles)}

# Mengubah kolom user dan books dengan peta encoding
data_rating_buku['user'] = data_rating_buku['User-ID'].apply(lambda x: encode_user_id1[x])
data_rating_buku['books'] = data_rating_buku['Book-Title'].apply(lambda x: encode_title1[x])
```

---

### **6. Pengambilan Sampel Acak**  
Agar distribusi data lebih merata, dilakukan pemilihan sampel data secara acak dengan **random state** untuk replikasi hasil yang konsisten.  

```python
data_rating_buku = data_rating_buku.sample(frac=1, random_state=123)
```

---

### **7. Pembagian Data Latih dan Uji**  
Data dibagi menjadi **train set (80%)** dan **test set (20%)** untuk mengevaluasi performa model.  

```python
# Menyiapkan fitur dan label
x = data_rating_buku[['user', 'books']].values
y = data_rating_buku['Book-Rating'].apply(lambda rating: (rating - min_rating) / (max_rating - min_rating))

# Membagi data menjadi training dan validation set
split_index = int(0.8 * len(data_rating_buku))
x_train, x_val = x[:split_index], x[split_index:]
y_train, y_val = y[:split_index], y[split_index:]
```

---

## **Data Preparation - Content-Based Filtering**  

Pendekatan kedua menggunakan **Content-Based Filtering**, yang memanfaatkan **teknik TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengekstrak fitur dari teks judul buku.  

### **8. Ekstraksi Fitur dengan TF-IDF**  
Ekstraksi fitur dengan **TF-IDF** dilakukan untuk mengukur **pentingnya sebuah kata** dalam judul buku relatif terhadap kumpulan dokumen lainnya.  

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TfidfVectorizer
vectorizer = TfidfVectorizer()

# Melatih Vectorizer pada kolom 'Titles'
vectorizer.fit(data_rating_buku['Book-Title'])

# Mendapatkan fitur nama yang dihasilkan
feature_names = vectorizer.get_feature_names_out()
print(feature_names)

# Menstranformasi data 'Titles' menggunakan TF-IDF
tfidf_matrix = vectorizer.fit_transform(data_rating_buku['Book-Title'])

# Mendapatkan bentuk dari tfidf_matrix
print(tfidf_matrix.shape)
```
Pada tahap ini:
- **TfidfVectorizer** digunakan untuk mengubah teks judul buku menjadi vektor numerik.  
- Setiap kata dalam judul buku diberi bobot berdasarkan frekuensinya dalam dataset.  
- Matriks hasil transformasi akan digunakan sebagai fitur dalam sistem rekomendasi berbasis konten.  


---
## **Modeling**  

Pada tahap ini, model sistem rekomendasi dikembangkan untuk memberikan rekomendasi buku kepada pengguna berdasarkan pola interaksi mereka dengan buku. Model akan menyajikan **Top-N Recommendation** sebagai output utama.  

Sistem rekomendasi yang digunakan terdiri dari dua pendekatan utama:  

### **1. Content-Based Filtering**  
Pendekatan ini merekomendasikan buku berdasarkan kemiripan fitur buku, seperti **judul**, **penulis**, dan **penerbit**. Model menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengekstrak fitur dari metadata buku dan mengukur kesamaan antar buku menggunakan **Cosine Similarity**.  

#### **Langkah-langkah:**
1. **Ekstraksi Fitur dengan TF-IDF:**  
   - Representasi teks buku dikonversi menjadi vektor numerik menggunakan TF-IDF.  
2. **Pengukuran Kesamaan dengan Cosine Similarity:**  
   - Menghitung kemiripan antara buku berdasarkan vektor TF-IDF.  
3. **Pembuatan Rekomendasi:**  
   - Jika pengguna menyukai buku tertentu, model akan merekomendasikan buku lain yang memiliki kesamaan tinggi dengan buku tersebut. 

Dengan hasil rekomendasinya sebagai berikut :

![tf_idf_rekomendasi](https://github.com/user-attachments/assets/a86b7b4f-8866-4e54-8925-b72eb841bd3b)


#### **Kelebihan & Kekurangan Content-Based Filtering**  
✅ **Kelebihan:**  
- Mampu memberikan rekomendasi meskipun tidak banyak interaksi pengguna.  
- Tidak mengalami masalah *cold start* untuk buku baru karena hanya bergantung pada fitur buku.  

❌ **Kekurangan:**  
- Memerlukan representasi fitur yang baik untuk hasil optimal.  
- Rekomendasi cenderung terbatas pada buku yang mirip dengan yang telah dinilai pengguna (*serendipity* rendah).  

---

### **2. Collaborative Filtering (Recommender Net)**  
Pendekatan ini merekomendasikan buku berdasarkan pola kesukaan pengguna lain yang memiliki preferensi serupa. Model yang digunakan adalah **Recommender Net**, sebuah model berbasis **Neural Collaborative Filtering** (NCF) yang terdiri dari embedding layer untuk pengguna dan buku.  

#### **Langkah-langkah:**  
1. **Encoding User dan ISBN ke dalam Representasi Numerik.**  
2. **Embedding Layer:**  
   - Mentransformasikan data pengguna dan buku ke dalam representasi **latent factor**.  
3. **Fully Connected Layers:**  
   - Menggunakan beberapa lapisan jaringan saraf untuk menangkap interaksi kompleks antara pengguna dan buku.  
4. **Prediksi Rating:**  
   - Model akan memprediksi rating yang mungkin diberikan pengguna untuk buku tertentu.  

Dengan hasil rekomendasinya sebagai berikut :

![recomendernet_rekomendasi](https://github.com/user-attachments/assets/068106e5-059f-45e1-a6eb-200757e0c4ff)


#### **Kelebihan & Kekurangan Collaborative Filtering**  
✅ **Kelebihan:**  
- Dapat menangkap pola kesukaan pengguna tanpa memerlukan metadata buku.  
- Mampu memberikan rekomendasi lebih bervariasi dibandingkan Content-Based Filtering.  

❌ **Kekurangan:**  
- Membutuhkan data rating yang cukup besar agar model dapat bekerja dengan baik.  
- Mengalami masalah *cold start* untuk pengguna baru yang belum memiliki interaksi.  

---

## **Evaluation**  

Evaluasi dilakukan untuk mengukur performa sistem rekomendasi yang telah dibangun menggunakan metrik yang sesuai dengan masing-masing pendekatan.

### **1. Evaluasi Content-Based Filtering**  
Model Content-Based Filtering dievaluasi menggunakan **Cosine Similarity**, yang mengukur sejauh mana dua buku memiliki kesamaan berdasarkan fitur TF-IDF.  

#### **Formula Cosine Similarity:**  

![cosine_similarity](https://github.com/user-attachments/assets/3075e829-fc23-4a83-867d-02161db0711a)

di mana:  
- \( A \) dan \( B \) adalah vektor representasi TF-IDF dari dua buku yang dibandingkan.  
- Nilai berkisar antara **0 hingga 1** (semakin mendekati 1, semakin mirip).  

#### **Precision@K untuk Evaluasi**  
Selain Cosine Similarity, evaluasi juga menggunakan **Precision@K** untuk mengukur relevansi rekomendasi. Precision@K menghitung persentase buku yang relevan di antara K rekomendasi teratas. 

##### **Formula Precision@K:**  

![presisi_rekomendasi](https://github.com/user-attachments/assets/3359975a-2bca-4b32-853f-7eb957e4463f)


#### **Interpretasi Hasil:**  
- Jika nilai cosine similarity tinggi, rekomendasi yang dihasilkan dianggap relevan.  
- Precision@K membantu mengukur seberapa baik sistem dalam memberikan rekomendasi yang benar di antara K rekomendasi teratas.  
- Model diuji dengan membandingkan hasil rekomendasi terhadap buku-buku yang telah dinilai pengguna sebelumnya atau menggunakan similarity matching untuk menangani variasi dalam judul buku.

#### **Visualisasi Kinerja Model Content-Based Filtering menggunakan presisi@k:**

![Visualisasi Kinerja Model Content-Based Filtering menggunakan presisi@k](https://github.com/user-attachments/assets/3a6dae12-b9b3-4519-9184-84b405a711c3)

Precision@K untuk Content-based Filtering pada percobaan ini sering bernilai 0 karena pencocokan judul untuk sistem rekomendasi dilakukan secara persis atau terlalu ketat harus sama dengan judul dari buku yang ingin dicari rekomendasinya, sehingga variasi kecil dalam judul membuat rekomendasi dianggap tidak relevan. Selain itu, jika buku target tidak ada dalam dataset evaluasi, precision otomatis menjadi 0%. Solusinya, saya menggunakan similarity matching seperti substring matching atau fuzzy matching untuk mempertimbangkan kemiripan judul dengan ambang batas 70%.

```python
from rapidfuzz import fuzz

def is_relevant(recommended_title, target_title, threshold=70):
    """
    Cek apakah buku direkomendasikan cukup mirip dengan target title.
    """
    return fuzz.partial_ratio(recommended_title.lower(), target_title.lower()) >= threshold
```

Dari hasil evaluasi Content-Based Filtering, model berhasil merekomendasikan 2 buku yang relevan dari 5 buku yang ditampilkan sebagai rekomendasi. Dengan nilai presisi 2/5 = 0.4 atau 40%

---

### **2. Evaluasi Collaborative Filtering (Recommender Net)**  
Model Collaborative Filtering dievaluasi menggunakan **Root Mean Squared Error (RMSE)** untuk mengukur perbedaan antara rating yang diprediksi dan rating yang sebenarnya diberikan pengguna.  

#### **Formula RMSE:**  

![rmse](https://github.com/user-attachments/assets/6a5136f1-211e-465a-9721-45c8ae2f25f5)


di mana:  
- \( y_i \) adalah rating aktual dari pengguna untuk buku tertentu.  
- \( y^_i \) adalah rating yang diprediksi oleh model.  
- Semakin kecil nilai RMSE, semakin akurat prediksi model.  

#### **Visualisasi Kinerja Model dalam Metode Evaluasi RMSE:**  

![rmse_rekomender_net](https://github.com/user-attachments/assets/815acac0-9c40-4f1c-9c98-15b9777dcb57)



- Jika RMSE rendah, berarti prediksi model mendekati rating sebenarnya yang diberikan pengguna.  

Berdasarkan grafik performa model dengan evaluasi RMSE, model menunjukkan performa yang baik, ditandai dengan rendahnya nilai RMSE. Interpretasi hasil juga dapat dilihat pada bagian **Modeling**, di mana model **Recommender Net** yang telah dilatih mampu menghasilkan rekomendasi buku yang bervariasi dan sesuai dengan preferensi pengguna.


---

