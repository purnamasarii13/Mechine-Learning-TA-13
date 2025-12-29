from flask import Flask, render_template, request
import os
import re
import numpy as np
import pandas as pd  # masih dipakai (opsional untuk halaman/cek dataset)

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# PATH (OPSIONAL) - DATASET
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset_tiktok-comments-scraper_2025-12-25_12-58-44-970.csv"
)

RAW_TEXT_COL = "text"
TEXT_COL = "text_clean"
LABEL_NUM_COL = "label_num"

# =========================
# PREPROCESSING (SASTRAWI)
# =========================
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
stemmer = StemmerFactory().create_stemmer()

def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    # hilangkan emoji/karakter non-ascii
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text: str) -> str:
    text = basic_clean(text)
    tokens = [w for w in text.split() if w not in stopwords]
    text = " ".join(tokens)
    return stemmer.stem(text)

# =========================
# LEXICON (RULE-BASED)
# =========================
positive_words = {
    "bagus","mantap","keren","hebat","senang","setuju","suka","baik","top","lucu",
    "puas","berhasil","terima","kasih","bangga","aman","percaya","solid","jelas",
    "cakep","oke","sip","jujur","makasih","mantul"
}

negative_words = {
    "buruk","jelek","parah","benci","marah","kesal","bohong","salah","gagal","rusak",
    "kecewa","jahat","ngawur","bodoh","payah","menyebalkan","tolol","ribet"
}

def lexicon_predict(cleaned_text: str):
    """
    Return:
      pred (0/1),
      prob_neg, prob_pos (0..1),
      pos_count, neg_count
    """
    tokens = cleaned_text.split()
    pos = sum(t in positive_words for t in tokens)
    neg = sum(t in negative_words for t in tokens)

    # smoothing supaya tidak 0/0 (juga bikin confidence tetap ada)
    pos_s = pos + 1
    neg_s = neg + 1
    total = pos_s + neg_s

    prob_pos = pos_s / total
    prob_neg = neg_s / total

    pred = 1 if prob_pos >= prob_neg else 0
    return pred, float(prob_neg), float(prob_pos), int(pos), int(neg)

# =========================
# OPSIONAL: LOAD DATASET (TIDAK WAJIB)
# =========================
# Supaya aman untuk Vercel + .vercelignore *.csv,
# aplikasi TIDAK perlu memuat CSV untuk prediksi manual.
# Kalau kamu mau tetap load saat LOCAL, gunakan flag env.

LOAD_DATASET = os.getenv("LOAD_DATASET", "0") == "1"
df = None
if LOAD_DATASET:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset CSV tidak ditemukan! Matikan LOAD_DATASET atau pastikan file ada.")
    df = pd.read_csv(DATA_PATH)
    if RAW_TEXT_COL not in df.columns:
        raise ValueError("Kolom 'text' tidak ditemukan pada dataset!")
    df[TEXT_COL] = df[RAW_TEXT_COL].fillna("").apply(preprocess)
    # weak label (opsional) jika kamu butuh analisis dataset
    def weak_label(cleaned: str):
        tokens = cleaned.split()
        pos = sum(t in positive_words for t in tokens)
        neg = sum(t in negative_words for t in tokens)
        if pos > neg:
            return 1
        elif neg > pos:
            return 0
        return None
    df[LABEL_NUM_COL] = df[TEXT_COL].apply(weak_label)

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        filled_text="",
        selected_model="lexicon",
        selected_mode="simple",
        prediction_text=None,
        proba_text=None,
        cleaned_text=None,
        labels=None,
        values=None,
        error_text=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text", "").strip()
        model_choice = request.form.get("model", "lexicon")  # tetap diterima dari form
        mode = request.form.get("mode", "simple")

        if not text:
            raise ValueError("Komentar TikTok tidak boleh kosong.")

        cleaned = preprocess(text)

        pred, prob_neg, prob_pos, pos_count, neg_count = lexicon_predict(cleaned)
        confidence = max(prob_neg, prob_pos) * 100

        model_name = "Lexicon Rule-Based"

        if pred == 1:
            result_html = f"✅ Sentimen <b>POSITIF (1)</b> — Model: <b>{model_name}</b>"
        else:
            result_html = f"⚠️ Sentimen <b>NEGATIF (0)</b> — Model: <b>{model_name}</b>"

        proba_text = (
            f"P(Negatif)={prob_neg:.4f} | "
            f"P(Positif)={prob_pos:.4f} | "
            f"Keyakinan: {confidence:.2f}% | "
            f"Hitung Kata: pos={pos_count}, neg={neg_count}"
        )

        labels = ["Negatif (0)", "Positif (1)"]
        values = [prob_neg, prob_pos]

        return render_template(
            "index.html",
            filled_text=text,
            selected_model=model_choice,
            selected_mode=mode,
            prediction_text=result_html,
            proba_text=proba_text,
            cleaned_text=cleaned,
            labels=labels,
            values=values,
            error_text=None
        )

    except Exception as e:
        return render_template(
            "index.html",
            filled_text=request.form.get("text", ""),
            selected_model=request.form.get("model", "lexicon"),
            selected_mode=request.form.get("mode", "simple"),
            prediction_text=None,
            proba_text=None,
            cleaned_text=None,
            labels=None,
            values=None,
            error_text=str(e)
        )

if __name__ == "__main__":
    app.run(debug=True)
