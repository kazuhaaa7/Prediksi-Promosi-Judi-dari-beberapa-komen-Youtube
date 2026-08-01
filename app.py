import streamlit as st
import joblib
import time

# ==========================
# Load Model
# ==========================
model = joblib.load("spam_classifier.pkl")

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Deteksi Promosi Judi",
    page_icon="🎰",
    layout="centered"
)

st.title("🎰 Deteksi Promosi Judi")
st.write(
    """
Aplikasi ini menggunakan **TF-IDF + Logistic Regression**
untuk mengklasifikasikan komentar YouTube menjadi:

- ✅ Bukan Promosi Judi
- 🚨 Promosi Judi
"""
)

# ==========================
# Input
# ==========================
comment = st.text_area(
    "Masukkan komentar",
    placeholder="Contoh: Ayo daftar di situs kami dan dapatkan bonus..."
)

# ==========================
# Prediksi
# ==========================
if st.button("🔍 Analisis"):

    if comment.strip() == "":
        st.warning("Silakan masukkan komentar terlebih dahulu.")

    else:

        with st.spinner("Sedang menganalisis komentar..."):

            progress = st.progress(0)

            for i in range(101):
                time.sleep(0.01)
                progress.progress(i)

            prediction = model.predict([comment])[0]

            # probabilitas (jika Logistic Regression)
            try:
                probability = model.predict_proba([comment])[0]
                confidence = max(probability) * 100
            except:
                confidence = None

            progress.empty()

        st.divider()

        # ==========================
        # HASIL
        # ==========================

        if prediction == 1:

            st.error("🚨 **PROMOSI JUDI TERDETEKSI**")

            st.markdown("""
            Sistem mendeteksi bahwa komentar ini mengandung
            indikasi **promosi perjudian online**.
            """)

        else:

            st.success("✅ **BUKAN PROMOSI JUDI**")

            st.markdown("""
            Sistem mendeteksi bahwa komentar ini **aman**
            dan tidak mengandung unsur promosi perjudian.
            """)

        # ==========================
        # Confidence
        # ==========================

        if confidence is not None:

            st.metric(
                label="Confidence Model",
                value=f"{confidence:.2f}%"
            )