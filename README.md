# 🌴 Sawit Ripeness Classifier (UAP Pembelajaran Mesin)

> Sistem klasifikasi tingkat kematangan tandan sawit (5 kelas) berbasis citra menggunakan CNN baseline dan Transfer Learning (MobileNetV2 & EfficientNetB0).  
> Mendukung prediksi **single/multi-image** dan **ZIP batch** via Streamlit.

---

## 📌 Table of Contents
1. [Deskripsi Project](#-deskripsi-project)  
   - [Latar Belakang](#-latar-belakang)  
   - [Tujuan](#-tujuan)  
2. [Dataset](#-dataset)  
3. [Eksperimen & Metodologi](#-eksperimen--metodologi)  
   - [EDA Singkat](#-eda-singkat)  
   - [Preprocessing](#-preprocessing)  
   - [Augmentasi](#-augmentasi)  
   - [Pemodelan](#-pemodelan)  
4. [Hasil & Analisis](#-hasil--analisis)  
   - [Ringkasan Performa](#-ringkasan-performa)  
   - [Confusion Matrix & Error Analysis](#-confusion-matrix--error-analysis)  
5. [Cara Menjalankan (VSCode / Lokal)](#-cara-menjalankan-vscode--lokal)  
6. [Demo Streamlit](#-demo-streamlit)  
7. [Struktur Folder](#-struktur-folder)  
8. [Keterbatasan & Rencana Perbaikan](#-keterbatasan--rencana-perbaikan)  
9. [Biodata / Kontributor](#-kontributor)

---

## 🧾 Deskripsi Project

Project ini dibuat untuk memenuhi **UAP Mata Kuliah Pembelajaran Mesin**.  
Tujuan utamanya adalah membangun sistem klasifikasi tingkat kematangan tandan sawit berbasis citra dan menyajikannya dalam bentuk aplikasi **Streamlit** yang mudah digunakan user.

### 🔍 Latar Belakang
Kematangan tandan sawit berpengaruh langsung terhadap kualitas hasil panen. Penilaian manual sering terpengaruh pencahayaan, sudut pengambilan gambar, dan subjektivitas pengamat. Oleh karena itu, project ini mengembangkan model klasifikasi citra untuk membantu prediksi kematangan secara lebih konsisten.

### 🎯 Tujuan
1. Membangun baseline model **CNN dari nol** sebagai pembanding.  
2. Menerapkan **Transfer Learning** untuk meningkatkan akurasi dan stabilitas pelatihan.  
3. Menyediakan aplikasi Streamlit yang mendukung:
   - Upload **1 gambar**
   - Upload **multi gambar**
   - Upload **ZIP batch**
   - Menampilkan **Top-3 + Confidence + Insight low-confidence**

---

## 🗂️ Dataset

Dataset: `dataset_sawit_UAP` (https://drive.google.com/drive/folders/1-nIuz8GupNU95R9naIz6s1i0DnDBahlP?usp=sharing)  
Jumlah kelas: **5**
- `decayed`
- `fully_ripe`
- `immature`
- `over_ripe`
- `partially_ripe`

Distribusi dataset (setelah augmentasi offline): **5058 gambar**.
Pembagian Data (Stratified Split)
Dataset dibagi secara **stratified** (proporsi tiap kelas tetap terjaga) dengan rasio:
- **Train: 70%** → **3541 gambar**
- **Validation: 15%** → **759 gambar**
- **Test: 15%** → **758 gambar**

Pembagian stratified digunakan untuk menjaga distribusi kelas tetap seimbang pada setiap subset.


**Catatan penting:** sebagian kelas memiliki kemiripan visual tinggi (mis. `partially_ripe` vs `fully_ripe`) sehingga bisa menurunkan confidence.

---

## 🧪 Eksperimen & Metodologi

### 📊 EDA Singkat
- Cek distribusi kelas
- Visualisasi contoh per kelas
- Cek variasi pencahayaan / background

### 🧼 Preprocessing
- Resize: **160×160**
- Normalisasi: (mis. `x/255.0`)
- Format RGB

### 🧩 Augmentasi
Augmentasi ringan untuk meningkatkan generalisasi tanpa membebani training:
- RandomFlip (horizontal)
- RandomRotation kecil
- RandomZoom kecil
- RandomContrast kecil

### 🧠 Pemodelan
Model yang diuji:
1. **Base CNN (Non-pretrained)** — baseline
2. **MobileNetV2 (Pretrained - Freeze + Head)** — efisien
3. **EfficientNetB0 (Pretrained - Fine-tune)** — akurasi terbaik

---

## 🏆 Hasil & Analisis

### 📌 Ringkasan Performa
| Model | Test Accuracy | Catatan |
|------|--------------:|--------|
| Base CNN | 0.61 | baseline, generalisasi terbatas |
| MobileNetV2 (Freeze) | 0.758 | stabil, ringan |
| EfficientNetB0 (Fine-tune) | 0.821 | terbaik, gap train-val kecil |


### 🧩 Confusion Matrix & Error Analysis
- Error dominan terjadi pada pasangan kelas yang mirip:
  - `partially_ripe` ↔ `fully_ripe`
- Faktor penyebab umum:
  - pencahayaan ekstrem
  - background dominan
  - objek tandan terlalu kecil di frame
  - blur

**Low confidence insight (di aplikasi):**
- menampilkan Top-3 probabilitas
- menampilkan margin Top1–Top2
- memberi rekomendasi foto ulang bila confidence rendah

---

## 💻 Cara Menjalankan (VSCode / Lokal)

### 1) Install dependensi
Jika pakai PDM:
```
pdm install
```
### 2) Jalankan Streamlit
```
pdm run streamlit run src/app.py
```
---

## 🖥️ Demo Streamlit

Aplikasi Streamlit disiapkan sebagai antarmuka untuk memprediksi tingkat kematangan tandan sawit berbasis citra.

### ✨ Fitur Utama
- **Pilih model**: Base CNN / MobileNetV2 / EfficientNetB0  
- **Mode input**:
  - Upload **single** gambar
  - Upload **multi-image** (lebih dari 1 gambar)
  - Upload **ZIP batch** (banyak gambar sekaligus)
- Menampilkan **Top-3 candidates** (3 kelas teratas beserta probabilitas)
- Menampilkan **Confidence + Margin (Top1–Top2)** untuk mendeteksi ambiguitas prediksi
- Peringatan **LOW confidence** + **insight & saran foto ulang**
- **Download hasil prediksi (CSV)** untuk multi-image dan ZIP

---

## 🧱 Struktur Folder

```bash
DEMO_UAP_ML/
├─ src/
│  └─ app.py
├─ sawit_models/
│  ├─ class_names.json
│  ├─ model_base_cnn.keras
│  ├─ model_mobilenetv2.keras
│  └─ model_efficientnetb0_ft.keras
├─ results/
│  ├─ figures/         # plot acc/loss, confusion matrix
│  ├─ reports/         # classification_report per model (txt)
│  ├─ tables/          # summary_metrics.csv, dll
│  └─ demo_outputs/    # output prediksi multi/zip (csv)
├─ docs/               # (opsional) screenshot UI Streamlit + hero image
├─ notebooks/          # (opsional) training & evaluasi di Colab
├─ pyproject.toml
├─ pdm.lock
└─ README.md
```

---

## ⚠️ Keterbatasan & Rencana Perbaikan

### Keterbatasan
- **Overlap visual antar kelas:** pasangan kelas seperti `partially_ripe` vs `fully_ripe` sering mirip karena perbedaan tingkat kematangan bersifat gradual.
- **Sensitif terhadap kondisi foto lapangan:** pencahayaan ekstrem, blur, background ramai, atau objek tandan terlalu kecil dapat menurunkan confidence.
- **Ketidakseimbangan data (minoritas):** beberapa kelas memiliki jumlah data lebih sedikit (mis. `decayed`) sehingga model berpotensi bias, meskipun sudah dibantu augmentasi / class weight.

### Rencana Perbaikan
- Menambah data untuk kelas yang lebih sedikit (contoh: `decayed`) dan memperkaya variasi kondisi lapangan.
- Menambahkan augmentasi yang lebih robust terhadap pencahayaan (brightness/contrast yang terkontrol) tanpa membuat data menjadi tidak realistis.
- Menambahkan interpretabilitas (opsional) seperti **Grad-CAM** untuk melihat area citra yang paling berpengaruh terhadap prediksi.
- (Opsional) Kalibrasi confidence (mis. **temperature scaling**) agar confidence lebih representatif pada data baru.

---

## 👥 Kontributor

| Nama | NIM | Prodi | Tahun |
|------|-----|------|------|
| **Muhammad Wildan Nabila** | **202210370311252** | **Informatika, Universitas Muhammadiyah Malang** | **2025** |
