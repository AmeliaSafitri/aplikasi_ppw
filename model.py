# =========================================================
# IMPORT LIBRARY
# =========================================================

import pandas as pd
import numpy as np
import re
import pickle

# Library untuk preprocessing bahasa Indonesia
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Library untuk pembagian data dan Grid Search
from sklearn.model_selection import train_test_split, ParameterGrid

# Digunakan untuk normalisasi nilai vektor Skip-Gram
from sklearn.preprocessing import MinMaxScaler

# Algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# Digunakan untuk menghitung performa model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Digunakan untuk membuat model Skip-Gram
from gensim.models import Word2Vec


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("Membaca dataset...")

# Membaca dataset berita dengan pemisah titik koma (;)
df = pd.read_csv(
    "dataset_detik_200_berita.csv",
    sep=";"
)

# Menampilkan jumlah data dan nama kolom
print("Jumlah data:", len(df))
print("Kolom:", df.columns.tolist())


# =========================================================
# 2. PREPROCESSING
# =========================================================

print("\nMelakukan preprocessing...")


# ---------------------------------------------------------
# Case Folding
# Mengubah seluruh huruf menjadi huruf kecil
# ---------------------------------------------------------

df["case_folding"] = df["isi_berita"].astype(str).str.lower()


# ---------------------------------------------------------
# Cleaning
# Menghapus tanda baca
# ---------------------------------------------------------

df["clean_text"] = df["case_folding"].apply(
    lambda x: re.sub(r"[^\w\s]", " ", x)
)

# Menghapus angka
df["clean_text"] = df["clean_text"].apply(
    lambda x: re.sub(r"\d+", " ", x)
)

# Menghapus spasi berlebih
df["clean_text"] = df["clean_text"].apply(
    lambda x: re.sub(r"\s+", " ", x).strip()
)


# ---------------------------------------------------------
# Stopword Removal
# Menghapus kata-kata umum yang kurang memberikan
# informasi penting dalam proses klasifikasi
# ---------------------------------------------------------

stop_factory = StopWordRemoverFactory()

stopword_remover = (
    stop_factory.create_stop_word_remover()
)

df["text_no_stopword"] = df["clean_text"].apply(
    lambda x: stopword_remover.remove(x)
)


# ---------------------------------------------------------
# Stemming
# Mengubah kata menjadi bentuk kata dasarnya
# ---------------------------------------------------------

stem_factory = StemmerFactory()

stemmer = stem_factory.create_stemmer()

df["text_stemmed"] = df["text_no_stopword"].apply(
    lambda x: stemmer.stem(x)
)


# ---------------------------------------------------------
# Tokenisasi
# Memecah teks menjadi kumpulan kata/token
# ---------------------------------------------------------

df["tokens_stemmed"] = df["text_stemmed"].apply(
    lambda x: x.split()
)

print("Preprocessing selesai.")


# =========================================================
# 3. DATA TRAINING DAN TESTING
# =========================================================

# X = data teks/token
# y = label/kategori berita

X = df["tokens_stemmed"]
y = df["label"]


# Membagi dataset menjadi:
# 80% data training
# 20% data testing
#
# stratify digunakan agar proporsi setiap kategori
# tetap seimbang pada data training dan testing.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData training:", len(X_train))
print("Data testing :", len(X_test))


# =========================================================
# 4. FUNGSI TRAINING SKIP-GRAM
# =========================================================

def train_skipgram(
    X_train,
    X_test,
    vector_size,
    window,
    min_count,
    epochs
):

    # -----------------------------------------------------
    # Membuat model Skip-Gram menggunakan Word2Vec
    #
    # sg=1 berarti menggunakan metode Skip-Gram.
    # -----------------------------------------------------

    model = Word2Vec(
        sentences=X_train.tolist(),

        # Ukuran vektor setiap kata
        vector_size=vector_size,

        # Jumlah kata di sekitar kata target
        window=window,

        # Minimal jumlah kemunculan kata
        min_count=min_count,

        workers=4,

        # sg=1 = Skip-Gram
        sg=1,

        # Jumlah iterasi training
        epochs=epochs,

        # Agar hasil dapat direproduksi
        seed=42
    )


    # -----------------------------------------------------
    # Fungsi untuk membuat satu vektor dari satu berita
    # -----------------------------------------------------

    def document_vector(tokens):

        vectors = []

        # Mengambil vektor setiap kata yang terdapat
        # di dalam vocabulary model Skip-Gram
        for word in tokens:

            if word in model.wv:

                vectors.append(
                    model.wv[word]
                )


        # Jika tidak ada kata yang ditemukan
        # maka menghasilkan vektor nol
        if len(vectors) == 0:

            return np.zeros(
                model.vector_size
            )


        # Mengambil rata-rata seluruh vektor kata
        # sehingga menjadi satu vektor untuk satu berita
        return np.mean(
            vectors,
            axis=0
        )


    # Membuat representasi numerik untuk data training
    X_train_vec = np.array([
        document_vector(tokens)
        for tokens in X_train
    ])


    # Membuat representasi numerik untuk data testing
    X_test_vec = np.array([
        document_vector(tokens)
        for tokens in X_test
    ])


    # -----------------------------------------------------
    # Normalisasi nilai vektor
    # -----------------------------------------------------

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_vec
    )

    X_test_scaled = scaler.transform(
        X_test_vec
    )


    # Mengembalikan model, scaler, dan data hasil vektorisasi
    return (
        model,
        scaler,
        X_train_scaled,
        X_test_scaled
    )


# =========================================================
# 5. GRID SEARCH
# =========================================================

print("\nMulai Grid Search...")


# ---------------------------------------------------------
# Parameter yang akan dicoba
#
# Grid Search akan mencoba berbagai kombinasi parameter
# Skip-Gram dan Naive Bayes untuk mencari performa terbaik.
# ---------------------------------------------------------

param_grid = {

    # Ukuran vektor Skip-Gram
    "vector_size": [50, 100],

    # Ukuran window/konteks kata
    "window": [3, 5],

    # Minimal frekuensi kemunculan kata
    "min_count": [1],

    # Jumlah epoch training
    "epochs": [10, 20],

    # Parameter alpha pada Naive Bayes
    "alpha": [0.1, 0.5, 1.0]
}


# Menyimpan hasil setiap percobaan
results = []


# Nilai awal untuk mencari model terbaik
best_f1 = -1
best_params = None
best_model = None
best_scaler = None
best_nb = None


# ---------------------------------------------------------
# Menjalankan seluruh kombinasi parameter
# ---------------------------------------------------------

for i, params in enumerate(
    ParameterGrid(param_grid),
    start=1
):

    print(
        f"\nPercobaan {i}/"
        f"{len(list(ParameterGrid(param_grid)))}"
    )

    print(params)


    # -----------------------------------------------------
    # Training Skip-Gram berdasarkan parameter yang sedang
    # dicoba
    # -----------------------------------------------------

    model, scaler, X_train_vec, X_test_vec = (
        train_skipgram(
            X_train,
            X_test,
            params["vector_size"],
            params["window"],
            params["min_count"],
            params["epochs"]
        )
    )


    # -----------------------------------------------------
    # Membuat model Naive Bayes
    # -----------------------------------------------------

    nb = MultinomialNB(
        alpha=params["alpha"]
    )


    # Melatih Naive Bayes menggunakan
    # hasil representasi Skip-Gram
    nb.fit(
        X_train_vec,
        y_train
    )


    # Melakukan prediksi terhadap data testing
    y_pred = nb.predict(
        X_test_vec
    )


    # -----------------------------------------------------
    # Menghitung performa model
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # Menampilkan hasil evaluasi
    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )


    # -----------------------------------------------------
    # Menyimpan hasil percobaan
    # -----------------------------------------------------

    results.append({

        "vector_size":
            params["vector_size"],

        "window":
            params["window"],

        "min_count":
            params["min_count"],

        "epochs":
            params["epochs"],

        "alpha":
            params["alpha"],

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1
    })


    # -----------------------------------------------------
    # Mengecek apakah model saat ini merupakan model terbaik
    # -----------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_params = params.copy()

        best_model = model

        best_scaler = scaler

        best_nb = nb


# =========================================================
# 6. HASIL GRID SEARCH
# =========================================================

# Mengubah hasil Grid Search menjadi DataFrame
results_df = pd.DataFrame(
    results
)


# Mengurutkan hasil berdasarkan F1 Score
# dari yang terbesar ke yang terkecil
results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)


print("\n====================================")
print("HASIL GRID SEARCH")
print("====================================")

print(
    results_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Menampilkan parameter terbaik
# ---------------------------------------------------------

print("\n====================================")
print("PARAMETER TERBAIK")
print("====================================")

print(best_params)

print(
    "\nF1 Score terbaik:",
    best_f1
)


# =========================================================
# 7. SIMPAN MODEL
# =========================================================

print("\nMenyimpan model...")


# ---------------------------------------------------------
# Menyimpan model Skip-Gram
# ---------------------------------------------------------

with open(
    "skipgram_model.pkl",
    "wb"
) as f:

    pickle.dump(
        best_model,
        f,
        protocol=4
    )


# ---------------------------------------------------------
# Menyimpan model Naive Bayes
# ---------------------------------------------------------

with open(
    "naive_bayes_skipgram.pkl",
    "wb"
) as f:

    pickle.dump(
        best_nb,
        f,
        protocol=4
    )


# ---------------------------------------------------------
# Menyimpan scaler
# Scaler dibutuhkan agar data baru dari aplikasi
# diproses dengan normalisasi yang sama seperti saat training.
# ---------------------------------------------------------

with open(
    "scaler_skipgram.pkl",
    "wb"
) as f:

    pickle.dump(
        best_scaler,
        f,
        protocol=4
    )


# ---------------------------------------------------------
# Menyimpan hasil Grid Search ke CSV
# ---------------------------------------------------------

results_df.to_csv(
    "hasil_grid_search.csv",
    index=False
)


# =========================================================
# 8. SELESAI
# =========================================================

print("\n====================================")
print("SELESAI")
print("====================================")

print(
    "skipgram_model.pkl        -> berhasil dibuat"
)

print(
    "naive_bayes_skipgram.pkl  -> berhasil dibuat"
)

print(
    "scaler_skipgram.pkl       -> berhasil dibuat"
)

print(
    "hasil_grid_search.csv     -> berhasil dibuat"
)