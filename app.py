from flask import Flask, render_template, request
import os
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# PATH AMAN (UNTUK LOCAL & VERCEL)
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
# WEAK LABELING
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

def weak_label(text: str):
    tokens = text.split()
    pos = sum(t in positive_words for t in tokens)
    neg = sum(t in negative_words for t in tokens)

    if pos > neg:
        return 1
    elif neg > pos:
        return 0
    else:
        return None

# =========================
# LOAD DATA & TRAIN MODEL
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset CSV tidak ditemukan!")

df = pd.read_csv(DATA_PATH)

if RAW_TEXT_COL not in df.columns:
    raise ValueError("Kolom 'text' tidak ditemukan pada dataset!")

# Preprocess
df[TEXT_COL] = df[RAW_TEXT_COL].fillna("").apply(preprocess)

# Weak labeling
df[LABEL_NUM_COL] = df[TEXT_COL].apply(weak_label)

# Buang data netral
df_lab = df.dropna(subset=[LABEL_NUM_COL]).copy()
df_lab[LABEL_NUM_COL] = df_lab[LABEL_NUM_COL].astype(int)

if df_lab.empty:
    raise ValueError("Data berlabel kosong. Periksa kamus kata!")

# TF-IDF
vectorizer = TfidfVectorizer()
X_all = vectorizer.fit_transform(df_lab[TEXT_COL])
y_all = df_lab[LABEL_NUM_COL].values

# Train SVM
svm_linear = SVC(kernel="linear", probability=True)
svm_rbf = SVC(kernel="rbf", gamma="scale", probability=True)

svm_linear.fit(X_all, y_all)
svm_rbf.fit(X_all, y_all)

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        filled_text="",
        selected_model="linear",
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
        model_choice = request.form.get("model", "linear")
        mode = request.form.get("mode", "simple")

        if not text:
            raise ValueError("Komentar TikTok tidak boleh kosong.")

        cleaned = preprocess(text)
        X_input = vectorizer.transform([cleaned])

        if model_choice == "rbf":
            model = svm_rbf
            model_name = "SVM RBF"
        else:
            model = svm_linear
            model_name = "SVM Linear"

        pred = int(model.predict(X_input)[0])
        probs = model.predict_proba(X_input)[0]

        prob_neg = float(probs[0])
        prob_pos = float(probs[1])
        confidence = max(prob_neg, prob_pos) * 100

        if pred == 1:
            result_html = f"✅ Sentimen <b>POSITIF (1)</b> — Model: <b>{model_name}</b>"
        else:
            result_html = f"⚠️ Sentimen <b>NEGATIF (0)</b> — Model: <b>{model_name}</b>"

        proba_text = (
            f"P(Negatif)={prob_neg:.4f} | "
            f"P(Positif)={prob_pos:.4f} | "
            f"Keyakinan: {confidence:.2f}%"
        )

        # 🔑 INI PENTING → grafik muncul
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
            selected_model=request.form.get("model", "linear"),
            selected_mode=request.form.get("mode", "simple"),
            prediction_text=None,
            proba_text=None,
            cleaned_text=None,
            labels=None,
            values=None,
            error_text=str(e)
        )

# =========================
# RUN LOCAL
# =========================
if __name__ == "__main__":
    app.run(debug=True)
