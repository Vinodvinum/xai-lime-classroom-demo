# Explainable AI (XAI) Classroom Demo

A mobile-friendly Streamlit classroom demonstration comparing the same Random Forest prediction in two views:

- **Without XAI:** prediction and model confidence.
- **With XAI:** the same prediction plus a LIME local explanation.

## What changed in this version

- The LIME explanation is generated automatically; students do not have to find or press a separate explanation button.
- Changing any fruit-property slider updates both the prediction and the explanation.
- The interface is arranged for phone and desktop screens.
- The chart clearly distinguishes positive and negative LIME contributions.
- Added separate tabs for reading the chart and a short classroom presentation script.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

This repository is configured with `app.py` as the main application file. Streamlit installs the packages from `requirements.txt`.

If the app is already deployed, a new commit to `main` should trigger a redeployment. Open the app's **Manage app** menu on Streamlit Community Cloud to inspect deployment logs if an error appears.

## Classroom demonstration

1. Open the **Live demo** tab.
2. Change one or more fruit-property sliders.
3. Compare the prediction and confidence in both cards.
4. Read the LIME chart: green bars support the displayed prediction; red bars oppose it.
5. Use **How to read LIME** or **Presentation script** when explaining the result.

## Important limitations

The fruit dataset and labels are synthetic and are used only to teach the idea. This is not a real or validated fruit-quality model and must not be used for real inspection or operational decisions.

LIME provides a local, approximate explanation of one model prediction. It does not prove causation and does not guarantee that the prediction is correct. Model confidence is not an explanation.
