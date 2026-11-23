"""MusicType: aplikasi Streamlit untuk mengenali instrumen dominan pada audio.

Alur: unggah audio -> potong per 3 detik -> 166 fitur (features.py) ->
predict_proba pada model SVM -> rata-rata probabilitas seluruh segmen.
Jalankan dengan: streamlit run app.py
"""
import os
import tempfile
from pathlib import Path
import tensorflow as tf
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st

SEGMENT_SEC = 3.0

from features import (
    SR, SEG_SEC, load_segments,
    seg_features, extract_mel
)


# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="MusicType",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

NAMA = {
    "cel": "Cello",
    "cla": "Clarinet",
    "flu": "Flute",
    "gac": "Gitar Akustik",
    "gel": "Gitar Elektrik",
    "org": "Organ",
    "pia": "Piano",
    "sax": "Saxophone",
    "tru": "Trumpet",
    "vio": "Biola",
    "voi": "Vokal"
}

def nama(kode):
    return NAMA.get(str(kode), str(kode))

def style_axes(ax):
    ax.set_facecolor("#211719")
    ax.tick_params(colors="#E8DCDC")
    ax.xaxis.label.set_color("#E8DCDC")
    ax.yaxis.label.set_color("#E8DCDC")
    ax.title.set_color("#F5E9E9")

    for spine in ax.spines.values():
        spine.set_color("#765B60")

    ax.grid(color="#765B60", alpha=0.2)

# Palet red velvet
BG = "#140A0C"
SURFACE = "#1F1013"
SURFACE_2 = "#2A161A"
MAROON = "#7B1E2B"
MAROON_LIGHT = "#A3293A"
CREAM = "#F5E9DF"
MUTED = "#B49A9A"
GOLD = "#E3B778"
LINE = "#3A2227"

MAROON_CMAP = LinearSegmentedColormap.from_list(
    "velvet",
    ["#140A0C", "#4A1220", "#7B1E2B", "#C2463F", "#E3B778", "#F5E9DF"]
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=DM+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg: #140A0C;
        --surface: #1F1013;
        --surface2: #2A161A;
        --maroon: #7B1E2B;
        --maroon-light: #A3293A;
        --cream: #F5E9DF;
        --muted: #B49A9A;
        --gold: #E3B778;
        --line: #3A2227;
    }

    .stApp {
        background: var(--bg);
        color: var(--cream);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }

    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label, span, div {
        font-family: 'DM Sans', sans-serif;
    }

    /* ---------- brand ---------- */
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 26px;
    }
    .brand-icon {
        background: var(--maroon);
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        color: var(--cream);
    }
    .brand-name {
        font-family: 'Fraunces', serif;
        font-size: 26px;
        font-weight: 700;
        color: var(--cream);
        letter-spacing: -0.5px;
        line-height: 1;
    }
    .brand-sub {
        color: var(--muted);
        font-size: 11px;
        margin-top: 4px;
    }

    /* ---------- hero ---------- */
    .hero {
        background: linear-gradient(180deg, #5E1623 0%, #3A0F18 55%, var(--bg) 100%);
        border-radius: 18px;
        padding: 44px 40px 40px 40px;
        margin-bottom: 28px;
    }
    .hero-tag {
        color: var(--gold);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: 44px;
        line-height: 1.12;
        font-weight: 800;
        color: var(--cream);
        margin: 12px 0 14px 0;
        letter-spacing: -1px;
    }
    .hero-desc {
        color: #E2CFC8;
        font-size: 15px;
        max-width: 560px;
        line-height: 1.7;
    }

    /* ---------- section ---------- */
    .section-title {
        font-family: 'Fraunces', serif;
        font-size: 24px;
        font-weight: 700;
        color: var(--cream);
        margin: 26px 0 10px 0;
        letter-spacing: -0.3px;
    }

    /* ---------- result (now playing style) ---------- */
    .result-card {
        display: flex;
        align-items: center;
        gap: 24px;
        background: var(--surface2);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0 20px 0;
    }
    .result-cover {
        flex: 0 0 120px;
        height: 120px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--maroon-light), #4A1220);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 52px;
        box-shadow: 0 10px 30px rgba(0,0,0,.45);
    }
    .result-label {
        color: var(--gold);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .result-name {
        font-family: 'Fraunces', serif;
        color: var(--cream);
        font-size: 40px;
        font-weight: 800;
        margin: 4px 0 6px 0;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .result-desc {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
    }

    /* ---------- probability list ---------- */
    .prob-row {
        display: grid;
        grid-template-columns: 150px 1fr 60px;
        align-items: center;
        gap: 16px;
        padding: 10px 14px;
        border-radius: 8px;
    }
    .prob-row:hover { background: var(--surface); }
    .prob-name { color: var(--cream); font-size: 14px; font-weight: 500; }
    .prob-row.top .prob-name { color: var(--gold); font-weight: 700; }
    .prob-track {
        background: var(--line);
        height: 6px;
        border-radius: 99px;
        overflow: hidden;
    }
    .prob-fill {
        background: var(--muted);
        height: 100%;
        border-radius: 99px;
    }
    .prob-row.top .prob-fill { background: var(--maroon-light); }
    .prob-val {
        color: var(--muted);
        font-size: 13px;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    .prob-row.top .prob-val { color: var(--cream); font-weight: 700; }

    /* ---------- misc cards ---------- */
    .info-card {
        background: var(--surface2);
        border-radius: 14px;
        padding: 20px 22px;
    }
    .info-label {
        color: var(--gold);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .info-value {
        font-family: 'Fraunces', serif;
        color: var(--cream);
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
    }
    .info-note {
        color: var(--muted);
        font-size: 12px;
        margin-top: 4px;
    }
    .upload-note {
        color: var(--muted);
        font-size: 12px;
        margin-top: 8px;
    }

    /* ---------- streamlit widgets ---------- */
    div[data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px dashed #6B2A36;
        border-radius: 14px;
        padding: 14px;
    }
    div[data-testid="stFileUploader"] section { background: transparent; }
    div[data-testid="stFileUploader"] button {
        background: var(--cream);
        color: var(--bg);
        border: none;
        border-radius: 99px;
        font-weight: 700;
        padding: 6px 20px;
    }
    div[data-testid="stFileUploader"] button:hover {
        background: #fff;
        color: var(--bg);
    }

    div[data-testid="stTabs"] button {
        color: #5b0c0c;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--cream);
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--maroon-light);
    }
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        background-color: var(--line);
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border-radius: 14px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] {
        color: var(--cream);
        font-family: 'Fraunces', serif;
    }

    div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--cream) !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: linear-gradient(to right, var(--maroon-light) 0%, var(--maroon-light) 100%);
    }

    .stButton button { border-radius: 99px; }

    hr { border-color: var(--line); }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL
# ==========================================

@st.cache_resource
def load_models():
    svm_bundle = joblib.load("model.joblib")
    cnn = tf.keras.models.load_model("musictype_cnn.keras")
    preprocessing = joblib.load("cnn_preprocessing.joblib")

    return svm_bundle, cnn, preprocessing


try:
    svm_bundle, cnn_model, cnn_preprocessing = load_models()

    svm_model = svm_bundle["model"]
    best_name = svm_bundle["best_name"]

    classes = svm_bundle.get("classes", list(svm_model.classes_))
    class_names = svm_bundle.get("class_names", NAMA)

except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()


# ==========================================
# 4. HEADER
# ==========================================

st.markdown("""
<div class="brand">
    <div class="brand-icon">♪</div>
    <div>
        <div class="brand-name">MusicType</div>
        <div class="brand-sub">Kenali instrumen di balik lagu</div>
    </div>
</div>

<div class="hero">
    <div class="hero-tag">Pengenal instrumen musik</div>
    <div class="hero-title">Lagu ini dimainkan<br>pakai apa, ya?</div>
    <div class="hero-desc">
        Unggah potongan lagu, lalu kami coba tebak instrumen apa yang
        paling terdengar di dalamnya. Lengkap dengan tingkat keyakinan
        dan gambaran bunyinya.
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. NAVIGASI
# ==========================================

tab_prediksi, tab_hasil = st.tabs([
    "Analisis Audio",
    "Performa Model"
])


# ==========================================
# 6. TAB PREDIKSI AUDIO
# ==========================================

with tab_prediksi:

    st.markdown(
        '<div class="section-title">Unggah audio kamu</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Pilih file audio untuk mengetahui instrumen "
        "yang paling dominan."
    )

    uploaded_file = st.file_uploader(
        "Seret file audio ke sini",
        type=["wav", "mp3", "ogg", "flac"],
        help="Format audio yang didukung: WAV, MP3, OGG, FLAC."
    )

    st.markdown(
        '<div class="upload-note">Maksimal 60 detik dianalisis '
        '· Tiap segmen berdurasi 3 detik</div>',
        unsafe_allow_html=True
    )

    metode = st.radio(
        "Pilih metode klasifikasi",
        ["SVM (Machine Learning)", "CNN (Deep Learning)"],
        horizontal=True
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        suffix = Path(uploaded_file.name).suffix or ".wav"
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            with st.spinner("Lagi mendengarkan audiomu..."):

                audio, segments = load_segments(
                    temp_path,
                    max_seconds=60
                )

                if metode == "SVM (Machine Learning)":

                    features = np.asarray([
                        seg_features(segment) for segment in segments
                    ])

                    probabilities = svm_model.predict_proba(features)
                    prediction_classes = svm_model.classes_

                else:

                    mel_segments = np.array([
                        extract_mel(segment) for segment in segments
                    ])

                    mel_segments = mel_segments[..., np.newaxis]

                    mean = cnn_preprocessing["mean"]
                    std = cnn_preprocessing["std"]

                    mel_segments = (mel_segments - mean) / std

                    probabilities = cnn_model.predict(
                        mel_segments,
                        verbose=0
                    )

                    prediction_classes = np.arange(probabilities.shape[1])

            # ----------------------------------
            # HASIL PREDIKSI
            # ----------------------------------

            mean_prob = probabilities.mean(axis=0)

            best_index = int(np.argmax(mean_prob))
            best_class = prediction_classes[best_index]
            best_label = nama(best_class)

            confidence = mean_prob[best_index] * 100

            st.divider()

            st.markdown(
                '<div class="section-title">Hasil analisis</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="result-card">
                <div class="result-cover">🎵</div>
                <div>
                    <div class="result-label">Instrumen dominan</div>
                    <div class="result-name">{best_label}</div>
                    <div class="result-desc">
                        Instrumen dengan rata-rata probabilitas
                        tertinggi dari seluruh segmen audio.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Tingkat keyakinan",
                f"{confidence:.1f}%"
            )

            col2.metric(
                "Jumlah segmen",
                len(segments)
            )

            col3.metric(
                "Durasi audio",
                f"{len(audio) / SR:.1f} detik"
            )

            # ----------------------------------
            # PROBABILITAS
            # ----------------------------------

            st.markdown(
                '<div class="section-title">Peluang tiap instrumen</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Rata-rata probabilitas setiap kelas dari seluruh segmen."
            )

            sorted_idx = np.argsort(mean_prob)[::-1]

            prob_html = ""
            for rank, idx in enumerate(sorted_idx):
                label = nama(prediction_classes[idx])
                value = float(mean_prob[idx])
                top_class = " top" if rank == 0 else ""
                prob_html += (
                    f'<div class="prob-row{top_class}">'
                    f'<div class="prob-name">{label}</div>'
                    f'<div class="prob-track">'
                    f'<div class="prob-fill" style="width:{value * 100:.1f}%"></div>'
                    f'</div>'
                    f'<div class="prob-val">{value * 100:.1f}%</div>'
                    f'</div>'
                )

            st.markdown(prob_html, unsafe_allow_html=True)

            # ----------------------------------
            # TIMELINE
            # ----------------------------------

            st.divider()

            st.markdown(
                '<div class="section-title">Per segmen</div>',
                unsafe_allow_html=True
            )

            threshold = st.slider(
                "Batas keyakinan",
                min_value=0,
                max_value=100,
                value=40,
                step=5
            ) / 100

            st.caption(
                "Prediksi dengan confidence di bawah ambang "
                "akan ditandai sebagai Tidak yakin."
            )

            top3_idx = np.argsort(
                probabilities,
                axis=1
            )[:, ::-1][:, :3]

            rows = []

            for i, idx in enumerate(top3_idx):

                conf = probabilities[i, idx[0]]

                prediction = (
                    nama(classes[idx[0]])
                    if conf >= threshold
                    else "Tidak yakin"
                )

                top3 = ", ".join([
                    f"{nama(classes[j])} "
                    f"({probabilities[i, j] * 100:.1f}%)"
                    for j in idx
                ])

                rows.append({
                    "Segmen": f"{i + 1:02d}",
                    "Waktu": (
                        f"{i * SEGMENT_SEC:.1f} - "
                        f"{(i + 1) * SEGMENT_SEC:.1f} detik"
                    ),
                    "Prediksi": prediction,
                    "Confidence": f"{conf * 100:.1f}%",
                    "Top 3 Prediksi": top3
                })

            timeline = pd.DataFrame(rows)

            st.dataframe(
                timeline,
                use_container_width=True,
                hide_index=True
            )

            n_unsure = int(
                (timeline["Prediksi"] == "Tidak yakin").sum()
            )

            st.caption(
                f"{n_unsure} dari {len(rows)} segmen "
                "memiliki confidence di bawah threshold."
            )

            # ----------------------------------
            # VISUALISASI AUDIO
            # ----------------------------------

            st.divider()

            st.markdown(
                '<div class="section-title">Gambaran bunyinya</div>',
                unsafe_allow_html=True
            )

            left, right = st.columns(2)

            with left:

                st.markdown("**Waveform**")
                st.caption("Bentuk gelombang audio terhadap waktu.")

                fig_wave, ax_wave = plt.subplots(figsize=(7, 3))

                librosa.display.waveshow(
                    audio,
                    sr=SR,
                    ax=ax_wave,
                    color="#C2463F"
                )

                ax_wave.set_xlabel("Waktu (detik)")
                ax_wave.set_ylabel("Amplitudo")
                style_axes(ax_wave, fig_wave)
                ax_wave.spines["bottom"].set_color(LINE)
                ax_wave.spines["left"].set_color(LINE)
                ax_wave.spines["top"].set_visible(False)
                ax_wave.spines["right"].set_visible(False)

                st.pyplot(fig_wave)
                plt.close(fig_wave)

            with right:

                st.markdown("**Mel Spectrogram**")
                st.caption("Representasi frekuensi audio terhadap waktu.")

                fig_mel, ax_mel = plt.subplots(figsize=(7, 3))

                mel = librosa.feature.melspectrogram(
                    y=audio,
                    sr=SR,
                    n_mels=128
                )

                mel_db = librosa.power_to_db(
                    mel,
                    ref=np.max
                )

                image = librosa.display.specshow(
                    mel_db,
                    sr=SR,
                    x_axis="time",
                    y_axis="mel",
                    ax=ax_mel,
                    cmap=MAROON_CMAP
                )

                style_axes(ax_mel, fig_mel)

                cbar = fig_mel.colorbar(
                    image,
                    ax=ax_mel,
                    format="%+2.0f dB"
                )
                cbar.ax.yaxis.set_tick_params(color=MUTED)
                plt.setp(cbar.ax.get_yticklabels(), color=MUTED)
                cbar.outline.set_edgecolor(LINE)

                st.pyplot(fig_mel)
                plt.close(fig_mel)

            st.success("Selesai! Audio berhasil dianalisis.")

        except Exception as e:

            st.error(f"Audio tidak dapat diproses: {e}")

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# ==========================================
# 7. TAB HASIL EKSPERIMEN
# ==========================================

with tab_hasil:

    st.markdown(
        '<div class="section-title">Performa model</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Perbandingan performa algoritma Machine Learning "
        "yang digunakan dalam eksperimen MusicType."
    )

    st.caption(
        "Tabel berikut berasal dari 5-fold StratifiedGroupKFold pada 6.705 "
        "klip IRMAS dengan 166 fitur audio dan parameter awal setiap algoritma "
        "(sebelum tuning). Waktu dalam detik."
    )

    result_path = Path("results_summary.csv")

    if result_path.exists():

        results = pd.read_csv(
            result_path,
            index_col=0
        )

        # TABEL HASIL
        st.markdown("**Perbandingan algoritma**")

        st.dataframe(
            results,
            use_container_width=True
        )

        # GRAFIK PERBANDINGAN
        available_metrics = [
            col for col in [
                "F1_macro",
                "Akurasi",
                "Precision",
                "Recall"
            ]
            if col in results.columns
        ]

        if available_metrics:

            st.markdown(
                '<div class="section-title">Grafik perbandingan</div>',
                unsafe_allow_html=True
            )

            chart_data = results[available_metrics].copy()

            st.bar_chart(
                chart_data,
                color=["#7B1E2B", "#C2463F", "#E3B778", "#F5E9DF"][:len(available_metrics)]
            )

    else:

        st.info(
            "Hasil eksperimen belum tersedia. "
            "Jalankan notebook training terlebih dahulu."
        )

    # HASIL EVALUASI CNN
    cnn_path = Path("cnn_results.csv")

    if cnn_path.exists():

        st.markdown(
            '<div class="section-title">Performa CNN</div>',
            unsafe_allow_html=True
        )

        cnn_results = pd.read_csv(cnn_path)

        st.dataframe(
            cnn_results,
            use_container_width=True,
            hide_index=True
        )

    # CONFUSION MATRIX SVM DAN CNN
    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    col_svm, col_cnn = st.columns(2)

    matrix_path = Path("confusion_matrix.png")
    cnn_matrix_path = Path("cnn_confusion_matrix.png")

    with col_svm:
        st.markdown("**SVM**")
        if matrix_path.exists():
            st.image(str(matrix_path), use_container_width=True)
        else:
            st.info("Confusion Matrix SVM belum tersedia.")

    with col_cnn:
        st.markdown("**CNN**")
        if cnn_matrix_path.exists():
            st.image(str(cnn_matrix_path), use_container_width=True)
        else:
            st.info("Confusion Matrix CNN belum tersedia.")

    st.divider()

    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">Model yang dipakai</div>
        <div class="info-value">{best_name}</div>
        <div class="info-note">
            Model yang digunakan untuk melakukan klasifikasi audio.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 8. FOOTER
# ==========================================

st.markdown("""
<br><br>
<div style="text-align:center; padding:20px 0; color:#8A7274;">
    <div style="font-family:'Fraunces',serif; font-size:18px; font-weight:700; color:#E3B778;">
        MusicType
    </div>
    <div style="font-size:11px; margin-top:6px;">
        Proyek Machine Learning · Klasifikasi instrumen musik
    </div>
</div>
""", unsafe_allow_html=True)