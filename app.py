# =====================================================================
# APLIKASI KLASIFIKASI TINGKAT STRES MAHASISWA — STREAMLIT
# Instrumen  : PSS-10 (Cohen et al., 1983)
# Adaptasi   : Hakim et al. (2024)
# Algoritma  : Support Vector Machine (SVM) — kernel linear, C=100
# Universitas: Krisnadwipayana Jakarta
# Penulis    : Ambrosia Woga Daso | 2026
# =====================================================================

import streamlit as st
import joblib
import numpy as np
import os

# =====================================================================
# 1. KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Deteksi Stres Mahasiswa — UNKRIS",
    page_icon="🧠",
    layout="centered"
)

# =====================================================================
# 2. LOAD MODEL & SCALER
# =====================================================================
@st.cache_resource
def load_assets():
    """
    Load model terbaik dan scaler dari direktori yang sama dengan app.py.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    model  = joblib.load(os.path.join(base_path, 'model_stres_terbaik.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'scaler_pss10.pkl'))
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError as e:
    st.error(f"❌ File model atau scaler tidak ditemukan: `{e}`")
    st.info("Pastikan file `model_stres_terbaik.pkl` dan `scaler_pss10.pkl` berada di folder yang sama dengan `app.py`.")
    st.stop()

# =====================================================================
# 3. KONSTANTA PSS-10
# =====================================================================
PERTANYAAN = [
    "1. Dalam sebulan terakhir, seberapa sering Anda merasa kecewa karena sesuatu yang terjadi secara tidak terduga?",
    "2. Dalam sebulan terakhir, seberapa sering Anda merasa tidak mampu mengendalikan hal-hal penting dalam hidup Anda?",
    "3. Dalam sebulan terakhir, seberapa sering Anda merasa gelisah dan stres?",
    "4. Dalam sebulan terakhir, seberapa sering Anda merasa percaya diri dengan kemampuan Anda untuk menyelesaikan masalah pribadi Anda?",
    "5. Dalam sebulan terakhir, seberapa sering Anda merasa segala sesuatu berjalan sesuai keinginan Anda?",
    "6. Dalam sebulan terakhir, seberapa sering Anda mengetahui bahwa Anda tidak bisa mengatasi hal-hal yang harus Anda lakukan?",
    "7. Dalam sebulan terakhir, seberapa sering Anda mampu mengendalikan hal-hal yang menjengkelkan dalam hidup Anda?",
    "8. Dalam sebulan terakhir, seberapa sering Anda merasa mampu mengendalikan permasalahan Anda?",
    "9. Dalam sebulan terakhir, seberapa sering Anda marah karena hal-hal yang terjadi di luar kendali Anda?",
    "10. Dalam sebulan terakhir, seberapa sering Anda merasa tidak mampu menyelesaikan permasalahan-permasalahan yang menumpuk dalam hidup Anda?",
]

# Item positif yang di-reverse: PSS4, PSS5, PSS7, PSS8 (0-based index)
REVERSE_IDX = [3, 4, 6, 7]

LABEL_SKALA = {
    0: "0 – Tidak Pernah",
    1: "1 – Hampir Tidak Pernah",
    2: "2 – Kadang-kadang",
    3: "3 – Cukup Sering",
    4: "4 – Sangat Sering",
}

LABEL_KELAS = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}

# =====================================================================
# 4. HEADER APLIKASI
# =====================================================================
st.title("🧠 Deteksi Tingkat Stres Mahasiswa Berbasis PSS-10")
st.markdown("**Universitas Krisnadwipayana Jakarta**")
st.write(
    "Isi kuesioner PSS-10 di bawah ini sesuai kondisi yang Anda rasakan "
    "**dalam satu bulan terakhir**. Tidak ada jawaban benar atau salah — "
    "jawablah sejujur mungkin."
)

st.divider()

# =====================================================================
# 5. PANDUAN SKALA (collapsed by default)
# =====================================================================
with st.expander("📖 Panduan Pengisian Skala", expanded=False):
    st.markdown("""
    | Nilai | Keterangan          |
    |:-----:|---------------------|
    | **0** | Tidak Pernah        |
    | **1** | Hampir Tidak Pernah |
    | **2** | Kadang-kadang       |
    | **3** | Cukup Sering        |
    | **4** | Sangat Sering       |
    """)
    st.caption("Geser slider ke posisi yang paling menggambarkan kondisi Anda.")

st.write("")

# =====================================================================
# 6. FORM INPUT PSS-10
# =====================================================================
st.subheader("📋 Kuesioner PSS-10")

with st.form(key="form_pss10"):
    jawaban = []
    for i, teks in enumerate(PERTANYAAN):
        st.markdown(f"**{teks}**")
        val = st.select_slider(
            label=f"PSS{i+1}",
            options=[0, 1, 2, 3, 4],
            value=st.session_state.get(f"q_{i}", 0),
            format_func=lambda x: LABEL_SKALA[x],
            label_visibility="collapsed",
            key=f"q_{i}",
        )
        jawaban.append(val)
        st.write("")

    st.divider()
    tombol_hitung = st.form_submit_button(
        "🔍 Hitung Tingkat Stres Saya",
        type="primary",
        use_container_width=True,
    )

# =====================================================================
# 7. TOMBOL ISI ULANG
# =====================================================================
if st.button("🔄 Isi Ulang Kuesioner", use_container_width=False):
    for i in range(10):
        if f"q_{i}" in st.session_state:
            del st.session_state[f"q_{i}"]
    if "hasil_analisis" in st.session_state:
        del st.session_state["hasil_analisis"]
    st.rerun()

# =====================================================================
# 8. PROSES PREDIKSI (Disimpan ke Session State)
# =====================================================================
if tombol_hitung:
    # Reverse scoring
    jawaban_processed = jawaban.copy()
    for idx in REVERSE_IDX:
        jawaban_processed[idx] = 4 - jawaban[idx]

    total_skor = sum(jawaban_processed)

    # Kategori skor manual PSS-10
    if total_skor <= 13:
        kategori_skor = "Rendah (0–13)"
    elif total_skor <= 26:
        kategori_skor = "Sedang (14–26)"
    else:
        kategori_skor = "Tinggi (27–40)"

    # Prediksi Model ML
    input_arr = np.array(jawaban_processed).reshape(1, -1)
    input_scaled = scaler.transform(input_arr)
    prediksi = model.predict(input_scaled)[0]
    
    # Handle jika model tidak support predict_proba (misal SVM tanpa probability=True)
    try:
        probabilitas = model.predict_proba(input_scaled)[0]
        conf_score = float(probabilitas[prediksi])
    except AttributeError:
        probabilitas = [0.0, 0.0, 0.0]
        probabilitas[prediksi] = 1.0
        conf_score = 1.0

    # Simpan ke session state agar data persistent
    st.session_state["hasil_analisis"] = {
        "total_skor": total_skor,
        "kategori_skor": kategori_skor,
        "prediksi": prediksi,
        "label_prediksi": LABEL_KELAS[prediksi],
        "conf_score": conf_score,
        "probabilitas": probabilitas
    }

# =====================================================================
# 9. TAMPILAN HASIL (Mengambil dari Session State)
# =====================================================================
if "hasil_analisis" in st.session_state:
    res = st.session_state["hasil_analisis"]
    
    st.divider()
    st.subheader("📊 Hasil Analisis Tingkat Stres")

    # Metrik utama
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Skor PSS-10", f"{res['total_skor']} / 40")
    with col_b:
        st.metric("Kategori Skor (PSS-10)", res['kategori_skor'])

    st.write("")

    # Hasil klasifikasi model
    col_c, col_d = st.columns(2)
    with col_c:
        st.metric("Prediksi Model", res['label_prediksi'])
    with col_d:
        st.metric("Confidence Score", f"{res['conf_score']*100:.1f}%")

    st.progress(res['conf_score'])

    # Catatan Metodologis jika ada perbedaan klasifikasi
    skor_ke_kelas = 0 if res['total_skor'] <= 13 else 1 if res['total_skor'] <= 26 else 2
    if skor_ke_kelas != res['prediksi']:
        st.caption(
            "ℹ️ *Catatan: Prediksi model berbeda dari kategorisasi skor PSS-10 "
            "karena model menganalisis pola kombinasi jawaban pada 10 item secara individual, "
            "bukan hanya sekadar bersandar pada total skor akumulatif.*"
        )

    st.write("")

    # Tampilan Alert & Interpretasi Berdasarkan Hasil
    if res['prediksi'] == 0:
        st.success("### ✅ Tingkat Stres Anda: RENDAH")
        st.markdown("""
        **Interpretasi:** Anda berada dalam kondisi psikologis yang relatif baik. Tekanan yang Anda rasakan masih dalam batas wajar.
        
        **Saran:**
        - Pertahankan kebiasaan positif dan pola koping adaptif yang sudah Anda lakukan.
        - Tetap luangkan waktu untuk relaksasi dan aktivitas yang Anda nikmati.
        """)
    elif res['prediksi'] == 1:
        st.warning("### ⚠️ Tingkat Stres Anda: SEDANG")
        st.markdown("""
        **Interpretasi:** Anda mengalami tekanan yang cukup berarti dan perlu diperhatikan sebelum berkembang menjadi stres berat.
        
        **Saran:**
        - Kelola waktu belajar dan tugas akademik dengan jadwal harian yang realistis.
        - Ceritakan beban yang Anda rasakan kepada teman dekat, keluarga, atau dosen wali.
        - Batasi konsumsi kafein dan perbaiki kualitas tidur secara bertahap.
        - Luangkan waktu untuk hobi atau aktivitas relaksasi di luar jam perkuliahan.
        """)
    else:
        st.error("### 🚨 Tingkat Stres Anda: TINGGI")
        st.markdown("""
        **Interpretasi:** Anda mengalami tekanan yang signifikan dan memerlukan perhatian segera.
        
        **Saran:**
        - Segera hubungi layanan konseling atau psikolog di kampus Anda.
        - Bicarakan kondisi Anda kepada dosen wali atau pihak kampus yang dapat membantu.
        - Prioritaskan istirahat — kondisi stres tinggi justru menurunkan produktivitas belajar.
        """)

    # Distribusi Probabilitas
    st.write("")
    st.markdown("**Distribusi Probabilitas Prediksi Model:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rendah", f"{res['probabilitas'][0]*100:.1f}%")
    with col2:
        st.metric("Sedang", f"{res['probabilitas'][1]*100:.1f}%")
    with col3:
        st.metric("Tinggi", f"{res['probabilitas'][2]*100:.1f}%")

    # Expandable Catatan Metodologis
    st.write("")
    with st.expander("ℹ️ Catatan Metodologis", expanded=False):
        st.markdown("""
        - **Instrumen:** PSS-10 (*Perceived Stress Scale*, Cohen et al., 1983)
        - **Adaptasi Bahasa Indonesia:** Hakim et al. (2024)
        - **Algoritma klasifikasi:** *Support Vector Machine* (SVM), kernel *linear*
                    (model terbaik dari perbandingan KNN, SVM, dan Random Forest — F1-Macro: 97,54%) 
        - ***Reverse scoring*** **diterapkan pada item:** PSS4, PSS5, PSS7, PSS8
        - **Normalisasi:** *Min-Max Scaler* / *Standard Scaler* yang sesuai dengan training data.
        - Aplikasi ini merupakan *Proof of Concept* penelitian skripsi dan **bukan pengganti diagnosis klinis profesional**.
        """)

# =====================================================================
# 10. FOOTER
# =====================================================================
st.divider()
st.caption(
    "Dikembangkan sebagai bagian dari penelitian skripsi · "
    "Ambrosia Woga Daso · Universitas Krisnadwipayana Jakarta · 2026"
)