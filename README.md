# AI vs Explainable AI — LIME Classroom Demo

A small interactive Streamlit application demonstrating the difference between a model prediction and a local explanation produced by LIME.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Sign in with GitHub.
3. Select this repository: `Vinodvinum/xai-lime-classroom-demo`
4. Select branch: `main`
5. Set the main file path to: `app.py`
6. Click Deploy.

Streamlit installs packages from `requirements.txt`.

## Classroom flow

1. Show the prediction and confidence on the left (without an explanation).
2. Explain that confidence is not the same as explainability.
3. Click **Generate explanation** on the right.
4. Explain the positive and negative LIME bars.

## Important limitation

The fruit dataset and labels are synthetic, created for teaching. This is not a real fruit-quality model and must not be used for real inspection or operational decisions. LIME provides a local approximate explanation; it does not prove causation or guarantee correctness.
