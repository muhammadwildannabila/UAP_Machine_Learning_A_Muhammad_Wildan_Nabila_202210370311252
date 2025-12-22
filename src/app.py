import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================
# CONFIG
# =====================
st.set_page_config(page_title="Sawit UAP - Ripeness Classifier", layout="wide")

APP_TITLE = "🌴 Sawit Ripeness Classifier (UAP)"
APP_DESC = (
    "Dashboard klasifikasi kematangan tandan sawit (5 kelas). "
    "Mendukung prediksi single, multi-image, dan ZIP batch."
)

BASE_DIR = Path(__file__).resolve().parent         # .../src
MODELS_DIR = BASE_DIR.parent / "sawit_models"      # .../sawit_models

MODEL_FILES = {
    "Base CNN (Non-pretrained)": MODELS_DIR / "model_base_cnn.keras",
    "MobileNetV2 (Pretrained - Freeze)": MODELS_DIR / "model_mobilenetv2.keras",
    "EfficientNetB0 (Pretrained - Fine-tune)": MODELS_DIR / "model_efficientnetb0_ft.keras",
}
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

IMG_SIZE = (160, 160)
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png"}

# =====================
# CYBERPUNK THEME (CSS)
# =====================
CYBERPUNK_CSS = """
<style>
/* App background */
.stApp {
  background: radial-gradient(1200px 800px at 18% 10%, rgba(124,58,237,0.30), transparent 60%),
              radial-gradient(1000px 700px at 82% 28%, rgba(249,115,22,0.22), transparent 55%),
              radial-gradient(900px 600px at 55% 85%, rgba(236,72,153,0.10), transparent 55%),
              #070a14;
  color: #e5e7eb;
}

/* Typography */
h1, h2, h3, h4 {
  color: #e9d5ff;
  text-shadow: 0 0 14px rgba(124,58,237,0.28);
}
p, li, label, div { color: rgba(229,231,235,0.9); }

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(124,58,237,0.20), rgba(249,115,22,0.12));
  border-right: 1px solid rgba(124,58,237,0.25);
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: 14px !important;
  border: 1px solid rgba(249,115,22,0.45) !important;
  background: rgba(249,115,22,0.12) !important;
  color: #ffd7b5 !important;
  box-shadow: 0 0 16px rgba(249,115,22,0.10);
}
.stButton>button:hover, .stDownloadButton>button:hover {
  background: rgba(249,115,22,0.18) !important;
}

/* Cards */
.cy-card {
  background: rgba(17, 24, 39, 0.68);
  border: 1px solid rgba(124,58,237,0.35);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 0 18px rgba(124,58,237,0.12);
}
.cy-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(249,115,22,0.35);
  background: rgba(249,115,22,0.10);
  color: #ffd7b5;
}
.cy-metric {
  font-size: 34px;
  font-weight: 850;
  letter-spacing: 0.2px;
  color: #f97316;
  text-shadow: 0 0 12px rgba(249,115,22,0.35);
  margin: 2px 0 6px 0;
}
.cy-subtle { color: rgba(229,231,235,0.78); font-size: 13px; }
.cy-kpi {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 10px;
}
.cy-kpi .box {
  background: rgba(2,6,23,0.35);
  border: 1px solid rgba(124,58,237,0.22);
  border-radius: 14px;
  padding: 10px 12px;
}
hr { border-color: rgba(124,58,237,0.25); }
</style>
"""
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

# =====================
# LOAD ASSETS
# =====================
@st.cache_resource
def load_model_cached(model_path: Path):
    return tf.keras.models.load_model(str(model_path))

@st.cache_data
def load_class_names(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32")
    x = np.expand_dims(x, axis=0)
    return x

def topk(prob: np.ndarray, class_names: List[str], k: int = 3) -> List[Tuple[str, float]]:
    idx = np.argsort(prob)[::-1][:k]
    return [(class_names[i], float(prob[i])) for i in idx]

def predict_pil(model, img: Image.Image, class_names: List[str]):
    x = preprocess_pil(img)
    prob = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(prob))
    return class_names[pred_idx], float(prob[pred_idx]), prob

def validate_assets():
    missing = []
    if not CLASS_NAMES_PATH.exists():
        missing.append(str(CLASS_NAMES_PATH))
    for name, p in MODEL_FILES.items():
        if not p.exists():
            missing.append(f"{name} -> {p}")
    return missing

def extract_images_from_zip(zip_bytes: bytes) -> List[Tuple[str, Image.Image]]:
    """Ekstrak gambar dari ZIP (tanpa tulis ke disk)."""
    out = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            ext = Path(name).suffix.lower()
            if ext not in ALLOWED_IMG_EXT:
                continue
            with z.open(info) as f:
                try:
                    img = Image.open(f).convert("RGB")
                    out.append((Path(name).name, img))
                except Exception:
                    pass
    return out

def interpret_confidence(conf: float, margin: float) -> Tuple[str, str]:
    """
    Confidence: Top-1 probability
    Margin: Top1 - Top2
    """
    if conf >= 0.75 and margin >= 0.15:
        return "HIGH", "Prediksi kuat dan stabil."
    if conf < 0.55 or margin < 0.08:
        return "LOW", "Prediksi kurang yakin (gambar ambigu / kualitas input kurang)."
    return "MEDIUM", "Prediksi cukup baik, namun ada kelas yang berdekatan."

def insight_low_conf(top1: str, top2: str) -> str:
    """
    Insight khusus sawit: partially_ripe vs fully_ripe sering overlap.
    """
    pair = {top1, top2}
    if "partially_ripe" in pair and "fully_ripe" in pair:
        return (
            "Model ragu karena **partially_ripe** dan **fully_ripe** punya warna yang sangat berdekatan "
            "(dipengaruhi pencahayaan & proporsi buah merah yang terlihat)."
        )
    return (
        "Model ragu karena ciri visual antar kelas berdekatan (warna/tekstur mirip) atau kualitas foto kurang optimal."
    )

def make_result_row(filename: str, pred: str, conf: float, prob: np.ndarray, class_names: List[str]) -> Dict:
    t3 = topk(prob, class_names, k=3)
    top1_lbl, top1_p = t3[0]
    top2_lbl, top2_p = t3[1]
    margin = float(top1_p - top2_p)
    level, _ = interpret_confidence(conf, margin)

    return {
        "filename": filename,
        "pred_label": pred,
        "confidence": round(conf, 4),
        "margin_top1_top2": round(margin, 4),
        "confidence_level": level,
        "top3": ", ".join([f"{lbl}:{p:.3f}" for lbl, p in t3]),
        "top1": top1_lbl,
        "top2": top2_lbl,
    }

# =====================
# UI HEADER
# =====================
st.title(APP_TITLE)
st.caption(APP_DESC)

missing = validate_assets()
if missing:
    st.error("File model / class_names tidak ditemukan. Pastikan struktur project benar:")
    for m in missing:
        st.write(f"- {m}")
    st.info("Run dari root project: `pdm run streamlit run src/app.py`")
    st.stop()

class_names = load_class_names(CLASS_NAMES_PATH)

# =====================
# SIDEBAR
# =====================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    st.markdown("### 🧭 Alur Penggunaan")
    st.write("1) Pilih **Model**")
    st.write("2) Pilih **Mode Input**")
    st.write("3) Upload **Gambar/ZIP**")
    st.write("4) Baca **Prediksi + Top-3 + Insight**")

    st.divider()

    model_name = st.selectbox("Pilih Model", list(MODEL_FILES.keys()), index=2)
    model_path = MODEL_FILES[model_name]
    st.caption(f"Model file: `{model_path.name}`")

    mode = st.radio("Mode Input", ["Single / Multi Image", "ZIP Batch"], index=0)

    st.divider()

    st.markdown("### 🎚️ Interpretasi Keyakinan")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.55, 0.01)
    show_top3 = st.checkbox("Tampilkan Top-3 (table + chart)", value=True)
    only_low_conf = st.checkbox("Tampilkan hanya LOW confidence (filter)", value=False)

    st.caption("**Tips:** LOW confidence sering terjadi pada partially_ripe vs fully_ripe atau foto blur/backlight.")

# Load model once
model = load_model_cached(model_path)

# =====================
# MAIN LAYOUT
# =====================
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown(
        f"""
<div class="cy-card">
  <span class="cy-badge">UAP Machine Learning Mode</span>
  <div style="margin-top:10px" class="cy-subtle">Model Aktif</div>
  <div class="cy-metric">{model_name}</div>
  <div class="cy-subtle">Input size: <b>{IMG_SIZE[0]}×{IMG_SIZE[1]}</b> • Output: <b>5 kelas</b> • Mode: <b>{mode}</b></div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("")

    if mode == "Single / Multi Image":
        st.subheader("📤 Upload 1 atau Banyak Gambar")
        files = st.file_uploader(
            "Pilih gambar (bisa lebih dari 1)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
    else:
        st.subheader("📦 Upload ZIP Berisi Gambar")
        zip_file = st.file_uploader("Upload file .zip (isi: jpg/png)", type=["zip"])
        files = None

with right:
    st.subheader("🧠 Hasil & Insight")
    st.caption("Hasil prediksi akan tampil di sini setelah input masuk. Untuk kasus ambigu, cek Top-3 dan margin Top1–Top2.")

# =====================
# PREDICTION - MODE 1
# =====================
if mode == "Single / Multi Image":
    if not files:
        with right:
            st.info("Upload 1 atau beberapa gambar untuk melihat prediksi + Top-3 + insight.")
    else:
        rows = []
        previews = []

        for f in files:
            try:
                img = Image.open(f)
            except Exception:
                continue

            pred, conf, prob = predict_pil(model, img, class_names)
            row = make_result_row(f.name, pred, conf, prob, class_names)
            rows.append(row)
            previews.append((f.name, img, row))

        df = pd.DataFrame(rows).sort_values(["confidence_level", "confidence"], ascending=[True, False])

        if only_low_conf:
            df = df[df["confidence_level"] == "LOW"]

        # --------- RIGHT: Summary cards + table
        with right:
            if df.empty:
                st.warning("Tidak ada hasil (atau semua tersaring). Coba matikan filter LOW confidence.")
            else:
                # KPI
                total_n = len(df)
                low_n = int((df["confidence_level"] == "LOW").sum())
                med_n = int((df["confidence_level"] == "MEDIUM").sum())
                high_n = int((df["confidence_level"] == "HIGH").sum())

                st.markdown(
                    f"""
<div class="cy-card">
  <div class="cy-subtle">Ringkasan Batch</div>
  <div class="cy-kpi">
    <div class="box"><b>Total</b><br>{total_n}</div>
    <div class="box"><b>HIGH</b><br>{high_n}</div>
    <div class="box"><b>MEDIUM</b><br>{med_n}</div>
    <div class="box"><b>LOW</b><br>{low_n}</div>
  </div>
  <div class="cy-subtle" style="margin-top:10px">
    LOW confidence biasanya terjadi ketika <b>margin Top1–Top2 kecil</b> (kelas ambigu).
  </div>
</div>
""",
                    unsafe_allow_html=True
                )

                st.markdown("### 📋 Tabel Hasil")
                show_cols = ["filename", "pred_label", "confidence", "margin_top1_top2", "confidence_level", "top3"]
                st.dataframe(df[show_cols], use_container_width=True)

                # Download CSV
                csv = df[show_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download hasil (CSV)",
                    data=csv,
                    file_name="prediksi_sawit_multi.csv",
                    mime="text/csv"
                )

        # --------- LEFT: Previews (grid)
        with left:
            st.markdown("### 🖼️ Preview & Prediksi")
            cols = st.columns(3)
            shown = 0
            for i, (fname, img, row) in enumerate(previews):
                if only_low_conf and row["confidence_level"] != "LOW":
                    continue
                with cols[shown % 3]:
                    st.image(img, caption=f"{fname}\n→ {row['pred_label']} | conf={row['confidence']:.3f} | margin={row['margin_top1_top2']:.3f}", width="stretch")
                    if row["confidence_level"] == "LOW":
                        st.warning("LOW confidence", icon="⚠️")
                shown += 1

        # --------- RIGHT: Detail Top-3 + insight untuk 1 sample yang dipilih
        with right:
            st.markdown("### 🔮 Detail Top-3 (pilih 1 file)")
            pick = st.selectbox("Pilih file untuk detail", df["filename"].tolist())
            pick_row = df[df["filename"] == pick].iloc[0]

            conf = float(pick_row["confidence"])
            margin = float(pick_row["margin_top1_top2"])
            level, msg = interpret_confidence(conf, margin)

            st.markdown(
                f"""
<div class="cy-card">
  <span class="cy-badge">Selected</span>
  <div class="cy-subtle" style="margin-top:8px">File</div>
  <div style="font-size:16px; font-weight:700">{pick}</div>
  <div class="cy-subtle" style="margin-top:6px">Prediksi</div>
  <div class="cy-metric">{pick_row["pred_label"]}</div>
  <div class="cy-subtle">Confidence: <b>{conf:.3f}</b> • Margin Top1–Top2: <b>{margin:.3f}</b></div>
  <div class="cy-subtle">Interpretasi: <b>{level}</b> — {msg}</div>
</div>
""",
                unsafe_allow_html=True
            )

            if show_top3:
                # parse top3 string to dataframe (biar simpel)
                top3_items = pick_row["top3"].split(", ")
                trows = []
                for r, item in enumerate(top3_items, start=1):
                    lbl, p = item.split(":")
                    trows.append({"rank": r, "class": lbl, "prob": float(p)})
                df_top = pd.DataFrame(trows)

                st.markdown("#### Top-3 Candidates")
                st.dataframe(df_top, use_container_width=True, hide_index=True)
                st.bar_chart(df_top.set_index("class")["prob"])

            if level == "LOW":
                st.warning("⚠️ Low confidence terdeteksi.")
                st.markdown(f"- Insight: {insight_low_conf(str(pick_row['top1']), str(pick_row['top2']))}")
                st.markdown(
                    """
**Saran foto ulang (agar lebih yakin):**
1) Ambil lebih dekat (tandan dominan di frame)  
2) Cahaya merata (hindari backlight/bayangan keras)  
3) Hindari blur (fokus tajam)  
4) Hindari objek menutupi tandan (rumput/daun)  
"""
                )

# =====================
# PREDICTION - MODE 2 (ZIP)
# =====================
else:
    if "zip_file" not in locals() or zip_file is None:
        with right:
            st.info("Upload ZIP, sistem akan memprediksi semua gambar di dalamnya dan memberi ringkasan + CSV.")
    else:
        imgs = extract_images_from_zip(zip_file.read())
        if not imgs:
            with right:
                st.error("Tidak ditemukan gambar valid di ZIP. Pastikan isi ZIP adalah JPG/PNG.")
            st.stop()

        with right:
            st.success(f"✅ Ditemukan {len(imgs)} gambar di ZIP. Memulai prediksi...")

        rows = []
        previews = []

        for name, img in imgs:
            pred, conf, prob = predict_pil(model, img, class_names)
            row = make_result_row(name, pred, conf, prob, class_names)
            rows.append(row)
            previews.append((name, img, row))

        df = pd.DataFrame(rows).sort_values(["confidence_level", "confidence"], ascending=[True, False])

        if only_low_conf:
            df = df[df["confidence_level"] == "LOW"]

        with right:
            if df.empty:
                st.warning("Tidak ada hasil (atau semua tersaring). Coba matikan filter LOW confidence.")
            else:
                st.markdown("### 📋 Ringkasan Hasil Batch (ZIP)")
                show_cols = ["filename", "pred_label", "confidence", "margin_top1_top2", "confidence_level", "top3"]
                st.dataframe(df[show_cols], use_container_width=True)

                # Distribusi prediksi (WOW)
                st.markdown("### 📊 Distribusi Prediksi")
                st.bar_chart(df["pred_label"].value_counts())

                # Distribusi confidence level (WOW)
                st.markdown("### 🧪 Distribusi Keyakinan")
                st.bar_chart(df["confidence_level"].value_counts())

                # Download CSV
                csv = df[show_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download hasil batch (CSV)",
                    data=csv,
                    file_name="prediksi_sawit_zip.csv",
                    mime="text/csv"
                )

        with left:
            st.markdown("### 🖼️ Preview (9 gambar pertama)")
            preview_cols = st.columns(3)
            for i, (name, img, row) in enumerate(previews[:9]):
                with preview_cols[i % 3]:
                    st.image(img, caption=f"{name}\n→ {row['pred_label']} | conf={row['confidence']:.3f} | margin={row['margin_top1_top2']:.3f}", width="stretch")
                    if row["confidence_level"] == "LOW":
                        st.warning("LOW confidence", icon="⚠️")

# =====================
# FOOTER
# =====================
st.divider()
st.caption("Muhammad Wildan Nabila • 202210370311252 •  Demo UAP Pembelajaran Mesin A • Multi Image + ZIP Batch • Top-3 + Confidence Insight")
