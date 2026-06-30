"""
app.py — Weather → Location/Country Predictor (Streamlit)
===========================================================
Loads the PyTorch models trained by train.py (matching
Location_Prediction.ipynb) and predicts location_name or country from
10 numeric weather measurements.

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Weather → Location Predictor", page_icon="🌦️", layout="wide")


# ── Model definition (must match train.py / the notebook) ───────
class LocationClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LocationClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_DIR / "meta.json") as f:
        meta = json.load(f)
    features = meta["features"]

    models, scalers, encoders = {}, {}, {}
    for target in ["location_name", "country"]:
        num_classes = meta[target]["num_classes"]
        model = LocationClassifier(input_dim=len(features), output_dim=num_classes)
        model.load_state_dict(torch.load(
            ARTIFACT_DIR / f"location_classifier_{target}.pth", map_location="cpu"))
        model.eval()
        models[target] = model
        scalers[target] = joblib.load(ARTIFACT_DIR / f"scaler_{target}.joblib")
        encoders[target] = joblib.load(ARTIFACT_DIR / f"label_encoder_{target}.joblib")

    sample_rows = pd.read_csv(ARTIFACT_DIR / "sample_rows.csv")
    return meta, features, models, scalers, encoders, sample_rows


meta, FEATURES, models, scalers, encoders, sample_rows = load_artifacts()


def predict(target: str, raw_values: dict, top_k: int = 5):
    row_df = pd.DataFrame([raw_values])[FEATURES]
    X_scaled = scalers[target].transform(row_df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = models[target](X_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top = torch.topk(probs, k=min(top_k, probs.shape[0]))
    results = [
        (encoders[target].inverse_transform([idx.item()])[0], prob.item())
        for prob, idx in zip(top.values, top.indices)
    ]
    return results


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🌦️ Model info")
    st.metric("Location test accuracy", f"{meta['location_name']['test_accuracy']*100:.1f}%")
    st.metric("Country test accuracy", f"{meta['country']['test_accuracy']*100:.1f}%")
    st.caption(
        f"{meta['location_name']['num_classes']} locations · "
        f"{meta['country']['num_classes']} countries · {len(FEATURES)} input features"
    )
    st.divider()
    st.caption(
        "PyTorch MLP (256→128, Dropout 0.2) trained on 10 weather measurements only "
        "(no lat/long or timezone) — accuracy is modest because location isn't fully "
        "determined by weather alone."
    )

st.title("🌦️ Weather → Location / Country Predictor")
st.write(
    "Predicts **location** or **country** purely from weather measurements "
    "(temperature, wind, pressure, precipitation, humidity, cloud, visibility, UV, gust), "
    "using the model from `Location_Prediction.ipynb`."
)

target_choice = st.radio("Predict:", ["location_name", "country"], horizontal=True,
                          format_func=lambda x: "Location name" if x == "location_name" else "Country")

tab1, tab2 = st.tabs(["🎲 Predict from a real example", "✍️ Enter values manually"])

# ── Tab 1: real example row ──────────────────────────────────────
with tab1:
    st.subheader("Pick a sample row")
    idx = st.selectbox(
        "Choose a row from the dataset sample",
        options=sample_rows.index,
        format_func=lambda i: f"{sample_rows.loc[i, 'location_name']}, {sample_rows.loc[i, 'country']}",
    )
    row = sample_rows.loc[idx]
    actual = row[target_choice]

    with st.expander("Show feature values for this row"):
        st.dataframe(row[FEATURES].to_frame("value"), use_container_width=True)

    if st.button("Predict", type="primary", key="predict_sample"):
        results = predict(target_choice, row[FEATURES].to_dict())
        st.markdown(f"**Predicted {target_choice.replace('_', ' ')}**")
        for name, prob in results:
            st.progress(prob, text=f"{name} — {prob*100:.1f}%")
        correct = results[0][0] == actual
        st.markdown(f"Actual: **{actual}** {'✅' if correct else '❌'}")

# ── Tab 2: manual entry ──────────────────────────────────────────
with tab2:
    st.subheader("Enter weather measurements")
    default_row = sample_rows.sample(1).iloc[0]
    manual_values = {}

    cols = st.columns(5)
    for i, feat in enumerate(FEATURES):
        manual_values[feat] = cols[i % 5].number_input(
            feat, value=float(default_row[feat]), format="%.3f", key=f"manual_{feat}"
        )

    if st.button("Predict", type="primary", key="predict_manual"):
        results = predict(target_choice, manual_values)
        st.markdown(f"**Predicted {target_choice.replace('_', ' ')}**")
        for name, prob in results:
            st.progress(prob, text=f"{name} — {prob*100:.1f}%")

st.divider()
st.caption(
    "Built with the same MLP architecture and feature set as `Location_Prediction.ipynb`. "
    "Run `python train.py` once to (re)generate `artifacts/` before launching this app."
)
