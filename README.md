# Digital Twin Corn Yield Simulator - IoT4Ag Hackathon 2026

An Explainable AI (XAI) Digital Twin ecosystem that completely solves the GxExM (Genotype x Environment x Management) puzzle for corn breeders and farmers.

This project delivers rigorous algorithmic XAI for AgTech R&D and condenses that power into an intuitive, highly-polished Streamlit portal for end-user adoption.

## 🚀 Two-Part Architecture

### 1. The UX/UI Product Mockup (Farmer-Centric Portal)
Located in `/farmer_dashboard/`. This Streamlit application serves as the production-ready polished UI targeted directly at agronomists and farmers. It visually hooks the IoT hardware and satellite models together and focuses purely on financial impact ("Gap Bushels").

**Run the Farmer Portal Mockup:**
```bash
streamlit run farmer_dashboard/app.py
```

### 2. The XGBoost ML Core (Breeder Portal)
Located in `/hybridscout/`. This is the fully-functional backend implementation. It consumes real 6-band Pléiades Neo satellite TIF imagery alongside physical metadata to generate rigorously cross-validated predictions utilizing target-encoded data and Iterative Imputation.

**Run the Full ML Backend & Breeder Portal:**
```bash
# 1. Activate Environment & Run ML Pipeline (Train Models & Extract Geodata)
PYTHONPATH=. python -m hybridscout.ml.extract_features
PYTHONPATH=. python -m hybridscout.ml.train_model
PYTHONPATH=. python -m hybridscout.ml.scoring

# 2. Start the Breeder Presentation App
streamlit run hybridscout/app/streamlit_app.py
```

## 🧠 Scientific Highlights (Judges Cheat Sheet)
*   **Hardware Bridge:** Mocks real-time edge-node telemetry (Subcanopy Temperature & Soil Moisture).
*   **Explainable AI (SHAP):** Does not treat agriculture as a black box. Displays precisely which variables elevated or reduced estimated yields.
*   **Iterative Imputation:** Missing field data is non-linearly recovered via Random Forest (`MissForest` algorithms).
*   **Biological Validity:** Timepoints strictly respect accumulated Growing Degree Days (GDDs), standardizing macro-environmental anomalies across varying states.

## 🛠 Setup & Installation
```bash
pip install -r requirements.txt
```

## 🧪 Testing the Pipeline
We maintain a suite of unit tests validating mock yield formulations over variable Nitrogen configurations.
```bash
pytest farmer_dashboard/test_app.py
```
