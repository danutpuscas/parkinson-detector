import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from feature_extractor import extract_features

st.set_page_config(page_title="Parkinson's Detection", layout="centered")
st.title("🎙️ Parkinson’s Detection Interface")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
age = st.number_input("Age", 1, 100, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
sex_bin = 1 if sex == "Male" else 0

if uploaded_file:
    path = "temp.wav"
    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Extracting features..."):
        features, snd, pitch = extract_features(path, age, sex_bin)

    df_feat = pd.DataFrame([features])
    st.subheader("🔍 Extracted Features")
    st.dataframe(df_feat.T.rename(columns={0: "Value"}))

    expected = ['Jitter(%)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
                'Shimmer', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
                'Shimmer:DDA', 'HNR', 'RPDE', 'DFA', 'PPE']

    df_model = df_feat[expected].copy()
    df_model.fillna(df_model.mean(), inplace=True)
    scaled = scaler.transform(df_model)
    proba = model.predict_proba(scaled)[0][1]
    prediction = int(proba > threshold)

    st.subheader("🧪 Prediction")
    st.markdown(f"**Result:** {'🟥 Parkinson Detected' if prediction else '🟩 Healthy'}")
    st.markdown(f"**Confidence:** {proba * 100:.2f}%")

    st.subheader("📈 Audio Waveform")
    fig1, ax1 = plt.subplots()
    ax1.plot(snd.xs(), snd.values[0])
    st.pyplot(fig1)

    st.subheader("📉 Pitch Contour")
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    fig2, ax2 = plt.subplots()
    ax2.plot(pitch.xs(), pitch_values, color="orange")
    st.pyplot(fig2)

    st.subheader("🌈 Spectrogram")
    spectrogram = snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    X, Y = np.meshgrid(np.linspace(0, snd.duration, spectrogram.xs().shape[0]),
                       np.linspace(0, 8000, spectrogram.values.shape[0]))
    fig3, ax3 = plt.subplots()
    c = ax3.pcolormesh(X, Y, 10 * np.log10(spectrogram.values), shading='auto', cmap='viridis')
    fig3.colorbar(c, ax=ax3)
    st.pyplot(fig3)
