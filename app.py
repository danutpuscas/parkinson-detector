import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, scaler, and threshold
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    return model, scaler, threshold

model, scaler, threshold = load_model()

st.title("Parkinson's Detection from Audio Features")
st.write("Upload a CSV file with pre-extracted features.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Extract the correct feature columns
        # Match number of features expected by scaler
        numeric_features = df.select_dtypes(include=[np.number])
        if numeric_features.shape[1] < scaler.n_features_in_:
            raise ValueError(f"CSV only has {numeric_features.shape[1]} features, but model expects {scaler.n_features_in_}")

        features = numeric_features.iloc[:, :scaler.n_features_in_]

        # Scale and predict
        scaled_features = scaler.transform(features)
        preds = model.predict_proba(scaled_features)[:, 1]
        mean_score = np.mean(preds)

        st.write(f"**Mean Prediction Score:** {mean_score:.3f}")
        st.write(f"**Threshold:** {threshold}")

        if mean_score > threshold:
            st.error("Prediction: Likely Parkinson's Disease")
        else:
            st.success("Prediction: Likely Healthy")

    except Exception as e:
        st.error(f"Error processing file: {e}")
