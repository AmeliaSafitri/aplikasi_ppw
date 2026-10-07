import streamlit as st
import pickle
import re
import requests
import numpy as np

from bs4 import BeautifulSoup
from urllib.parse import urlparse

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)
from Sastrawi.Stemmer.StemmerFactory import (
    StemmerFactory
)


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================
# Mengatur judul, icon, dan tampilan aplikasi Streamlit.

st.set_page_config(
    page_title="Klasifikasi Berita",
    page_icon="📰",
    layout="wide"
)


# =========================================================
# 2. MEMUAT MODEL
# =========================================================
# Model yang digunakan:
# - Skip-Gram       : mengubah teks menjadi vektor
# - Naive Bayes     : melakukan klasifikasi
# - Scaler          : menyesuaikan skala vektor
#
# Ketiga file .pkl harus berada di folder yang sama
# dengan app.py.

@st.cache_resource
def load_models():

    with open("skipgram_model.pkl", "rb") as file:
        skipgram_model = pickle.load(file)

    with open("naive_bayes_skipgram.pkl", "rb") as file:
        naive_bayes_model = pickle.load(file)

    with open("scaler_skipgram.pkl", "rb") as file:
        scaler = pickle.load(file)

    return (
        skipgram_model,
        naive_bayes_model,
        scaler
    )


skipgram_model, naive_bayes_model, scaler = load_models()


# =========================================================
# 3. MEMUAT SASTRAWI
# =========================================================
# Sastrawi digunakan untuk:
# - Menghapus stopword Bahasa Indonesia
# - Melakukan stemming
#
# Preprocessing di aplikasi harus sama dengan preprocessing
# yang digunakan saat membuat model.

stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()


# =========================================================
# 4. FUNGSI PREPROCESSING
# =========================================================
# Fungsi ini membersihkan teks berita sebelum masuk
# ke model Skip-Gram.

def preprocessing(text):

    # Case Folding
    # Mengubah seluruh huruf menjadi huruf kecil.
    text = text.lower()

    # Cleaning
    # Menghapus tanda baca dan karakter khusus.
    text = re.sub(r"[^\w\s]", " ", text)

    # Menghapus angka.
    text = re.sub(r"\d+", " ", text)

    # Menghapus spasi yang berlebihan.
    text = re.sub(r"\s+", " ", text).strip()

    # Menghapus stopword Bahasa Indonesia.
    text = stopword_remover.remove(text)

    # Mengubah kata menjadi bentuk dasar.
    text = stemmer.stem(text)

    # Memisahkan teks menjadi token/kata.
    tokens = text.split()

    return text, tokens


# =========================================================
# 5. MENGAMBIL ISI BERITA DARI URL
# =========================================================
# Fungsi ini mengambil halaman berita dari link yang
# dimasukkan pengguna dan mengambil teks dari tag <p>.

def get_news_content(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    # Jika halaman gagal diakses,
    # akan menghasilkan error.
    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Menghapus bagian halaman yang tidak diperlukan.
    for element in soup(
        ["script", "style", "nav", "footer", "header", "aside"]
    ):
        element.decompose()

    # Mengambil seluruh paragraf berita.
    paragraphs = soup.find_all("p")

    content = " ".join(
        paragraph.get_text(" ", strip=True)
        for paragraph in paragraphs
    )

    return content


# =========================================================
# 6. MEMBUAT DOCUMENT VECTOR
# =========================================================
# Skip-Gram menghasilkan vector untuk setiap kata.
#
# Karena satu berita terdiri dari banyak kata,
# vector seluruh kata dirata-ratakan menjadi satu
# vector yang mewakili satu berita.

def document_vector(model, tokens):

    vectors = []

    for word in tokens:

        # Hanya mengambil kata yang terdapat
        # di dalam vocabulary Skip-Gram.
        if word in model.wv:
            vectors.append(model.wv[word])

    # Jika tidak ada kata yang dikenal model,
    # gunakan vector nol.
    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    # Menghasilkan satu vector untuk seluruh berita.
    return np.mean(vectors, axis=0)


# =========================================================
# 7. CEK KATEGORI BERDASARKAN URL
# =========================================================
#
# Tujuannya:
# Jika URL secara jelas menunjukkan kategori lain,
# misalnya:
# health.detik.com
# maka berita langsung dianggap di luar
# kategori Finance/Sport.
#
# Ini diperlukan karena model Naive Bayes hanya mengenal
# kategori yang ada pada data training.

def cek_kategori_url(url):

    try:

        parsed_url = urlparse(url)

        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        # Gabungkan domain dan path URL.
        url_text = domain + " " + path

    except Exception:

        return False, ""


    # -----------------------------------------------------
    # Keyword Finance
    # -----------------------------------------------------

    finance_keywords = [
        "finance",
        "finansial",
        "bisnis",
        "business",
        "market",
        "ekonomi"
    ]


    # -----------------------------------------------------
    # Keyword Sport
    # -----------------------------------------------------

    sport_keywords = [
        "sport",
        "sports",
        "olahraga",
        "sepakbola",
        "bola",
        "football",
        "soccer"
    ]


    # -----------------------------------------------------
    # Keyword kategori lain
    # -----------------------------------------------------
    # Jika ditemukan salah satu keyword ini,
    # URL dianggap bukan Finance/Sport.

    other_keywords = [
        "health",
        "kesehatan",
        "fotohealth",
        "lifestyle",
        "gaya-hidup",
        "travel",
        "wisata",
        "food",
        "kuliner",
        "entertainment",
        "hiburan",
        "technology",
        "teknologi",
        "tekno",
        "otomotif",
        "detikhot",
        "wolipop",
        "inet"
    ]


    # Cek apakah URL termasuk Finance.
    for keyword in finance_keywords:

        if keyword in url_text:
            return True, "finance"


    # Cek apakah URL termasuk Sport.
    for keyword in sport_keywords:

        if keyword in url_text:
            return True, "sport"


    # Cek apakah URL termasuk kategori lain.
    for keyword in other_keywords:

        if keyword in url_text:
            return False, "other"


    # Jika URL tidak menunjukkan kategori tertentu.
    return True, "unknown"


# =========================================================
# 8. NORMALISASI LABEL MODEL
# =========================================================
# Fungsi ini menyamakan berbagai kemungkinan nama label
# menjadi hanya 3 hasil:
#
# finance
# sport
# other

def normalisasi_label(label):

    label = str(label).lower().strip()


    # Jika label merupakan Finance.
    if label in [
        "finance",
        "finansial",
        "bisnis",
        "business",
        "ekonomi"
    ]:
        return "finance"


    # Jika label merupakan Sport.
    if label in [
        "sport",
        "sports",
        "olahraga",
        "sepakbola",
        "bola"
    ]:
        return "sport"


    # Selain itu dianggap kategori lain.
    return "other"


# =========================================================
# 9. JUDUL APLIKASI
# =========================================================

st.title("📰 Klasifikasi Berita")

st.subheader(
    "Klasifikasi Berita Menggunakan Skip-Gram + Naive Bayes"
)

st.write(
    """
    Aplikasi ini menggunakan metode **Skip-Gram** untuk
    representasi teks dan **Naive Bayes** untuk klasifikasi
    kategori berita.

    Kategori yang digunakan dalam aplikasi ini hanya:
    **Finance** dan **Sport**.
    """
)


# =========================================================
# 10. INFORMASI KATEGORI
# =========================================================

col1, col2 = st.columns(2)

with col1:

    st.info("💰 **Finance**")

with col2:

    st.info("⚽ **Sport**")


# =========================================================
# 11. INPUT LINK BERITA
# =========================================================

st.markdown("### 🔗 Masukkan Link Berita")

url = st.text_input(
    "URL berita:",
    placeholder="https://contoh.com/berita..."
)


# =========================================================
# 12. TOMBOL KLASIFIKASI
# =========================================================

if st.button("🔍 Klasifikasikan Berita"):

    # Mengecek apakah URL sudah dimasukkan.
    if not url:

        st.warning(
            "Silakan masukkan link berita terlebih dahulu."
        )

        st.stop()


    # Mengecek format URL.
    if not url.startswith(
        ("http://", "https://")
    ):

        st.error(
            "URL harus diawali dengan http:// atau https://"
        )

        st.stop()


    try:

        # =================================================
        # 13. CEK KATEGORI URL
        # =================================================
        # URL diperiksa terlebih dahulu.
        #
        # Contoh:
        # health.detik.com -> other
        # sport.detik.com  -> sport
        # finance.detik.com -> finance

        url_valid, url_category = cek_kategori_url(url)


        # =================================================
        # 14. MENGAMBIL ISI BERITA
        # =================================================

        with st.spinner(
            "Mengambil isi berita..."
        ):

            news_content = get_news_content(url)


        if not news_content:

            st.error(
                "Isi berita tidak berhasil ditemukan "
                "dari halaman tersebut."
            )

            st.stop()


        # Menampilkan informasi bahwa berita berhasil
        # diambil dari URL.

        st.success(
            "Isi berita berhasil diambil."
        )


        # Menampilkan isi berita secara opsional.
        with st.expander(
            "📄 Lihat Isi Berita"
        ):

            st.write(news_content)


        # =================================================
        # 15. PREPROCESSING
        # =================================================
        # Teks berita dibersihkan agar formatnya sama
        # dengan data saat model dilatih.

        with st.spinner(
            "Melakukan preprocessing..."
        ):

            cleaned_text, tokens = preprocessing(
                news_content
            )


        # Mengecek apakah masih ada kata setelah
        # preprocessing.

        if len(tokens) == 0:

            st.warning(
                "Teks berita tidak memiliki kata "
                "yang dapat diproses."
            )

            st.stop()


        # =================================================
        # 16. MEMBUAT REPRESENTASI SKIP-GRAM
        # =================================================
        # Token berita diubah menjadi satu vector
        # menggunakan model Skip-Gram.

        with st.spinner(
            "Membuat representasi Skip-Gram..."
        ):

            vector = document_vector(
                skipgram_model,
                tokens
            )

            vector = vector.reshape(
                1,
                -1
            )

            # Vector disesuaikan menggunakan scaler
            # yang digunakan saat training.
            vector_scaled = scaler.transform(
                vector
            )


        # =================================================
        # 17. KLASIFIKASI NAIVE BAYES
        # =================================================
        # Naive Bayes memberikan probabilitas untuk
        # masing-masing kategori.

        with st.spinner(
            "Melakukan klasifikasi..."
        ):

            probabilities = (
                naive_bayes_model
                .predict_proba(
                    vector_scaled
                )[0]
            )

            classes = (
                naive_bayes_model.classes_
            )

            # Probabilitas terbesar.
            max_probability = np.max(
                probabilities
            )

            # Posisi probabilitas terbesar.
            max_index = np.argmax(
                probabilities
            )

            # Hasil prediksi asli dari model.
            model_prediction = classes[
                max_index
            ]


        # =================================================
        # 18. NORMALISASI HASIL PREDIKSI
        # =================================================
        # Hasil model disamakan menjadi:
        # finance / sport / other

        normalized_prediction = normalisasi_label(
            model_prediction
        )


        # =================================================
        # 19. THRESHOLD CONFIDENCE
        # =================================================
        # Jika confidence model kurang dari 60%,
        # hasil tidak dianggap cukup yakin.

        THRESHOLD = 0.60


        # =================================================
        # 20. MENENTUKAN HASIL AKHIR
        # =================================================
        #
        # Prioritas:
        #
        # 1. Jika URL menunjukkan kategori lain
        #    -> OTHER
        #
        # 2. Jika hasil model bukan Finance/Sport
        #    -> OTHER
        #
        # 3. Jika confidence < 60%
        #    -> OTHER
        #
        # 4. Selain itu
        #    -> Finance atau Sport

        if url_category == "other":

            final_prediction = "other"

        elif normalized_prediction == "other":

            final_prediction = "other"

        elif max_probability < THRESHOLD:

            final_prediction = "other"

        else:

            final_prediction = normalized_prediction


        # =================================================
        # 21. MENAMPILKAN HASIL
        # =================================================

        st.markdown("---")

        st.markdown("## 🎯 Hasil Klasifikasi")


        # =================================================
        # JIKA FINANCE
        # =================================================

        if final_prediction == "finance":

            st.success(
                "💰 Kategori berita: **FINANCE**"
            )

            st.info(
                f"Confidence model: "
                f"**{max_probability:.2%}**"
            )


            # Hanya Finance/Sport yang menampilkan
            # diagram probabilitas.

            st.markdown(
                "### 📊 Probabilitas Kategori"
            )

            probability_dict = {}

            for category, probability in zip(
                classes,
                probabilities
            ):

                probability_dict[
                    str(category)
                ] = float(probability)


            st.bar_chart(
                probability_dict
            )


        # =================================================
        # JIKA SPORT
        # =================================================

        elif final_prediction == "sport":

            st.success(
                "⚽ Kategori berita: **SPORT**"
            )

            st.info(
                f"Confidence model: "
                f"**{max_probability:.2%}**"
            )


            # Diagram hanya ditampilkan untuk Sport.

            st.markdown(
                "### 📊 Probabilitas Kategori"
            )

            probability_dict = {}

            for category, probability in zip(
                classes,
                probabilities
            ):

                probability_dict[
                    str(category)
                ] = float(probability)


            st.bar_chart(
                probability_dict
            )


        # =================================================
        # JIKA OTHER
        # =================================================
        # Jika berita bukan Finance/Sport,
        # hanya tampilkan pesan.
        #
        # Tidak menampilkan:
        # - confidence
        # - diagram
        # - prediksi asli model
        # - threshold

        else:

            st.warning(
                "⚠️ **Tidak termasuk kategori finance/sport**"
            )

            st.info(
                "Berita yang dimasukkan berada di luar "
                "kategori finance/sport."
            )


    # =====================================================
    # 22. ERROR SAAT MENGAMBIL BERITA
    # =====================================================

    except requests.exceptions.RequestException:

        st.error(
            "Gagal mengambil halaman berita. "
            "Pastikan URL benar dan dapat diakses."
        )


    # =====================================================
    # 23. ERROR LAINNYA
    # =====================================================

    except Exception as e:

        st.error(
            f"Terjadi kesalahan: {str(e)}"
        )