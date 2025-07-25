# 🏁 Tugas Akhir (TA) - Final Project

**Nama Mahasiswa**: I Gusti Ngurah Ervan Juli Ardana
**NRP**: 5025211205  
**Judul TA**: KLASIFIKASI PENYAKIT KRONIS BERBASIS MULTIMODAL DEEP LEARNING PADA DATA SAMPEL BPJS KESEHATAN 
**Dosen Pembimbing**: Dini Adni Navastara, S.Kom., M.Sc.  
**Dosen Ko-pembimbing**: Prof. Dr. Ir. Diana Purwitasari, S.Kom., M.Sc.

---

## 📺 Demo Aplikasi  

[![Demo Aplikasi](https://github.com/user-attachments/assets/2ac0b3e3-4440-4cf3-a569-4a79397bb8a9)](https://youtu.be/k0ok6r54A_c)  
*Klik gambar di atas untuk menonton demo*

---

*Konten selanjutnya hanya merupakan contoh awalan yang baik. Anda dapat berimprovisasi bila diperlukan.*

## 🛠 Panduan Instalasi & Menjalankan Software  

### Prasyarat

| Library                             | Keterangan                                                                             |
| ---------------------------------- | -------------------------------------------------------------------------------------- |
| `streamlit`                        | Membangun antarmuka aplikasi berbasis web                                             |
| `pandas`                           | Manipulasi dan analisis data tabular (CSV, DataFrame)                                 |
| `numpy`                            | Operasi numerik, vektor, dan array                                                    |
| `tensorflow`                       | Framework Deep Learning; memuat model `.h5`, `Layer`, `keras.preprocessing`           |
| `scikit-learn` (`sklearn`)         | Normalisasi fitur numerik dengan `StandardScaler`                                     |
| `networkx`                         | Membangun dan menganalisis struktur graph                                             |
| `pyvis`                            | Visualisasi graph interaktif di Streamlit dengan tampilan HTML                        |
| `pickle`                           | Memuat objek Python yang telah disimpan (misalnya tokenizer)                          |
| `tempfile`                         | Membuat file HTML sementara untuk visualisasi Pyvis                                   |
| `streamlit.components.v1`         | Menampilkan konten HTML kustom (grafik jaringan dari Pyvis)                           |
| `IPython.display`                 | Untuk membersihkan output interaktif (opsional, digunakan di notebook)                |
| `scipy`                            | Paket pendukung untuk analisis ilmiah, meskipun tidak secara eksplisit dipakai        |
| `re`  | cleaning teks dengan ekspresi reguler   |

### Langkah-langkah  
1. **Clone Repository**  
   ```bash
   git clone https://github.com/Informatics-ITS/ta-NgurahErvan.git
   ```
2. **Instalasi Dependensi**
   ```bash
   cd [folder-proyek]
   pip install -r requirements.txt
   !wget --no-check-certificate http://nlp.stanford.edu/data/glove.6B.zip
   !unzip glove.6B.zip  
   ```
3. **Jalankan Aplikasi**
   ```bash
   streamlit run app.py
   ```
5. Buka browser dan kunjungi: `http://localhost:8501` (sesuaikan dengan port proyek Anda)

---
## ✅ Validasi

- Project ini tidak dapat digunakan langsung, karena terdapat data confidential

---

## ⁉️ Pertanyaan?

Hubungi:
- Penulis: ngurahervan23@gmail.com
- Pembimbing Utama: dini_navastara@if.its.ac.id
