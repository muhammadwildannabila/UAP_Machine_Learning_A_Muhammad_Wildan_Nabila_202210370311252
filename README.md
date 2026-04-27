<div align="center">

# 🌴 Sawit Ripeness Classifier  
### Deep Learning for Oil Palm Maturity Classification

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-CNN-orange?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer-Vision-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge)

### 🔵 Academic Project (Coursework / Assignment) 2025

</div>

---

## 🖥️ Application Preview

![Dashboard](assets/dashboard-preview.png)

> Interactive dashboard for oil palm ripeness classification with multi-input support and confidence-based insights.

---

## 🧠 Project Overview

This project develops a **deep learning-based image classification system** to identify the ripeness level of oil palm fruit bunches.

The system integrates:
- Convolutional Neural Networks (CNN)
- Transfer Learning (MobileNetV2 & EfficientNetB0)
- Interactive deployment using Streamlit

The objective is to improve **consistency, scalability, and objectivity** in ripeness assessment compared to manual observation.

---

## 🎯 Project Objectives

- Build a **baseline CNN model** for benchmarking  
- Apply **transfer learning** to improve performance  
- Develop an **interactive prediction system**  
- Provide **confidence-based insights** for decision support  

---

## 🗂️ Dataset

- Source: [Google Drive Dataset](https://drive.google.com/drive/folders/1-nIuz8GupNU95R9naIz6s1i0DnDBahlP?usp=sharing)  
- Total Classes: **5**

### 🖼️ Sample Data per Class

| Class | Example |
|------|--------|
| **immature** | ![](assets/immature.jpg) |
| **partially_ripe** | ![](assets/partiallyripe.jpg) |
| **fully_ripe** | ![](assets/fullyripe.jpg) |
| **over_ripe** | ![](assets/overripe.jpg) |
| **decayed** | ![](assets/decayed.jpg) |

---

### 📊 Data Characteristics

- Total images: **5058**
- Stratified split:
  - Train: 70%
  - Validation: 15%
  - Test: 15%

### ⚠️ Notes

- Some classes exhibit **high visual similarity**, especially:
  - *partially_ripe* vs *fully_ripe*
- Variability in:
  - Lighting conditions  
  - Background complexity  
  - Object scale  

These factors contribute to **classification ambiguity and model uncertainty**.

---

## 🧪 Methodology

### 🔹 Exploratory Data Analysis
- Class distribution analysis  
- Sample visualization  
- Lighting & background variability check  

### 🔹 Preprocessing
- Resize: **160×160**
- Normalization: `x / 255`
- RGB conversion  

### 🔹 Data Augmentation
- Horizontal flip  
- Small rotation  
- Zoom & contrast adjustment  

### 🔹 Modeling Strategy
- **Baseline CNN** (non-pretrained)  
- **MobileNetV2** (transfer learning - frozen layers)  
- **EfficientNetB0** (fine-tuning)  

---

## 🏆 Results & Analysis

### 📌 Model Performance

| Model | Accuracy | Insight |
|------|---------:|--------|
| Base CNN | 0.61 | Limited generalization |
| MobileNetV2 | 0.758 | Stable & efficient |
| EfficientNetB0 | **0.821** | Best performance |

---

### 🧩 Error Analysis

Main challenges:
- `partially_ripe` vs `fully_ripe` confusion  
- Lighting variability  
- Background noise  
- Small object representation  

### 🔍 Confidence Insight (App Feature)
- Top-3 prediction output  
- Confidence margin (Top1 vs Top2)  
- Low-confidence detection + recommendation  

---

## 📈 Key Contributions

- Comparative study: **CNN vs Transfer Learning**  
- Fine-tuning EfficientNet for improved accuracy  
- Deployment into **Streamlit-based application**  
- Integration of **confidence-aware prediction system**  
- Application of AI in **agriculture domain**

---

## 💻 How to Run

```bash
pdm install
pdm run python -m streamlit run src/app.py
```
---

---

## 🖥️ Application Features

- Multi-model selection (CNN, MobileNetV2, EfficientNetB0)  
- Input support:
  - Single image  
  - Multiple images  
  - ZIP batch upload  
- Top-3 prediction display with probability scores  
- Confidence & ambiguity detection (Top1–Top2 margin)  
- CSV export for batch prediction results  

---

## 🔗 Live Demo

👉 https://uapmachinelearningamuhammadwildannabila202210370311252-3dgw4zg.streamlit.app/

---

## 🧱 Project Structure

```bash
DEMO_UAP_ML/
├─ src/
│  └─ app.py
├─ sawit_models/
├─ results/
├─ notebooks/
├─ requirements.txt
└─ README.md
```
---
---

## ⚠️ Limitations

- Visual overlap between similar classes (e.g., *partially_ripe* vs *fully_ripe*)  
- Sensitivity to real-world image conditions (lighting, blur, background noise)  
- Class imbalance in certain categories  

---

## 🚀 Future Improvements

- Data enrichment for minority classes  
- More robust augmentation strategies  
- Model interpretability (Grad-CAM)  
- Confidence calibration for improved reliability  

---

## 🎯 Project Positioning

This project demonstrates competencies in:

- Deep Learning (CNN & Transfer Learning)  
- Computer Vision (Image Classification)  
- Model Evaluation & Error Analysis  
- Model Deployment (Streamlit)  
- Applied AI in Agriculture  

---

## 👨‍💻 Author

**Muhammad Wildan Nabila**  
Informatics — Universitas Muhammadiyah Malang  
2025  

---

## 🚀 Closing

> Transforming agricultural image data into actionable insights through deep learning and intelligent systems.
