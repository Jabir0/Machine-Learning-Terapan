# Laporan Proyek Machine Learning - Jabir Muktabir

## Project Overview

Sistem rekomendasi telah menjadi bagian penting dalam berbagai industri, terutama dalam bidang e-commerce dan hiburan. Dengan semakin meningkatnya jumlah buku yang tersedia secara daring, pengguna sering kali kesulitan menemukan buku yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi buku menjadi solusi untuk membantu pengguna menemukan buku yang relevan berdasarkan data historis dan perilaku mereka.

Dataset yang digunakan dalam proyek ini berasal dari [Kaggle - Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset), yang dikumpulkan dari komunitas Book-Crossing pada tahun 2004. Dataset ini terdiri dari tiga bagian utama: data pengguna, data buku, dan data rating buku.

Referensi terkait:
- [T. Monisya Afriyanti and E. Retnoningsih, “Sistem Rekomendasi Buku Perpustakaan Menggunakan Algoritma Frequent Pattern Growth Library Book Recommendation System using Frequent Pattern Growth Algorithm,” Techno.COM, vol. 21, no. 2, pp. 292–310, May 2022.](https://core.ac.uk/download/pdf/521875503.pdf)

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

Dataset yang digunakan dalam proyek ini terdiri dari tiga file utama:  

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
# **Data Preparation**  

Sebelum data digunakan untuk membangun model rekomendasi, dilakukan beberapa langkah **persiapan data** untuk memastikan kualitas dan efisiensi pemrosesan.  

### **1. Data Merging**  
Untuk menghubungkan data rating dengan informasi buku, dilakukan **penggabungan (merging) data** antara dataset **ratings** dan **books** berdasarkan ISBN. Hal ini bertujuan untuk menambahkan informasi tambahan seperti judul buku, penulis, dan penerbit ke dalam dataset rating.  

Kode yang digunakan untuk melakukan **data merging**:  
```python
data_rating_buku = pd.merge(
    ratings,
    books,
    on = 'ISBN',
    how = 'left'
)
data_rating_buku
```
Metode **left join** digunakan agar setiap entri rating tetap dipertahankan, meskipun beberapa ISBN mungkin tidak ditemukan dalam dataset buku.  

### **2. Missing Values Handling**  
Beberapa kolom dalam dataset memiliki **nilai yang hilang (missing values)**. Untuk mengatasi hal ini, dilakukan **penghapusan (drop)** entri yang memiliki nilai kosong, terutama pada kolom-kolom penting seperti **judul buku (Book-Title)** dan **penulis (Book-Author)**.  

```python
data_rating_buku = data_rating_buku.dropna()
```

### **3. Duplicates Data Handling**  
Data yang memiliki **duplikasi (duplicate entries)** dapat memengaruhi hasil model. Oleh karena itu, dilakukan **penghapusan duplikasi** berdasarkan kombinasi **User-ID dan ISBN** untuk memastikan bahwa setiap pengguna hanya memberikan satu rating untuk satu buku.  

```python
#Menghitung jumlah data yang duplikat
duplicate_count_isbn = data_rating_buku['ISBN'].duplicated().sum()
duplicate_count_title = data_rating_buku['Book-Title'].duplicated().sum()

print(f"Number of duplicates in the 'ISBN' column : {duplicate_count_isbn}")
print(f"Number of duplicated in the 'Book-Title' column : {duplicate_count_title}")

#Menghapus data yang duplikat
data_rating_buku = data_rating_buku.drop_duplicates(subset='ISBN')
data_rating_buku = data_rating_buku.drop_duplicates(subset='Book-Title')
```

### **4. Data Selection**  
Dataset ini memiliki **241.065 entri**, yang cukup besar untuk diproses. Demi efisiensi serta mempertimbangkan keterbatasan penyimpanan di **Kaggle**, hanya digunakan **30.000 entri teratas** berdasarkan jumlah rating yang diberikan oleh pengguna.  

```python
data_rating_buku = data_rating_buku.head(30000)
```

---

## **Data Preparation - Collaborative Filtering**  

Sistem rekomendasi yang akan dibangun menggunakan pendekatan **Collaborative Filtering**. Sebelum memasuki tahap pelatihan model, dilakukan beberapa proses tambahan:  

1. **Encoding dan Mapping**  
   - Mengonversi data menjadi format numerik agar dapat diproses oleh model.  
   - ISBN dan User-ID dikonversi ke dalam indeks numerik untuk mempermudah pemrosesan.  

   ```python
   # Mendapatkan daftar user ID yang unik
    user_ids = df['User-ID'].unique().tolist()

    # Membuat encoding user ID
    encode_user_id1 = {user_id: index for index, user_id in enumerate(user_ids)}
    encoded_user_id2 = {index: user_id for index, user_id in enumerate(user_ids)}

    # Mendapatkan daftar judul buku yang unik
    titles = df['Book-Title'].unique().tolist()

    # Membuat encoding untuk judul buku
    encode_title1 = {title: index for index, title in enumerate(titles)}
    encoded_title2 = {index: title for index, title in enumerate(titles)}

    #Mengubah kolom user dan books dengan menggunakan peta encode
    df['user'] = df['User-ID'].apply(lambda x : encode_user_id1[x])
    df['books'] = df['Book-Title'].apply(lambda x : encode_title1[x])
   ```

2. **Pengambilan Sampel Acak**  
   - Untuk memastikan distribusi data yang lebih merata, dilakukan pemilihan sampel data secara acak.  

   ```python
   df = df.sample(frac = 1, random_state = 123)
   ```

3. **Pembagian Data Latih dan Uji**  
   - Data dibagi menjadi **train set (80%)** dan **test set (20%)** untuk mengevaluasi performa model.  

   ```python
   #Menampung nilai dari kolom 'user' dan 'books' dalan variabel x
    x = df[['user','books']].values
    #Normalisasi nilai 'Book-Rating'
    y = df['Book-Rating'].apply(lambda rating: (rating - min_rating) / (max_rating - min_rating))

    #Membagi data menjadi training dan validation set
    split_index = int(0.8 * len(df))
    x_train, x_val = x[:split_index], x[split_index:]
    y_train,y_val = y[:split_index], y[split_index:]
   ```

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

#### **Interpretasi Hasil:**  
- Jika nilai cosine similarity tinggi, rekomendasi yang dihasilkan dianggap relevan.  
- Model diuji dengan membandingkan hasil rekomendasi terhadap buku-buku yang telah dinilai pengguna sebelumnya.  

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
![rekomendernet_collaborative_evaluation](https://github.com/user-attachments/assets/6d3b4f3a-ad2d-4234-aa89-baeab8a84005)

- Jika RMSE rendah, berarti prediksi model mendekati rating sebenarnya yang diberikan pengguna.  

Berdasarkan grafik performa model dengan evaluasi RMSE, model menunjukkan performa yang baik, ditandai dengan rendahnya nilai RMSE. Interpretasi hasil juga dapat dilihat pada bagian **Modeling**, di mana model **Recommender Net** yang telah dilatih mampu menghasilkan rekomendasi buku yang bervariasi dan sesuai dengan preferensi pengguna.


---

