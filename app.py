import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="AI vs Explainable AI", page_icon="🔍", layout="wide")
st.title("🔍 AI vs Explainable AI")
st.write("A hands-on classroom demo: compare a prediction without an explanation with the same prediction explained by LIME.")

FEATURES = ["Weight (g)", "Size (cm)", "Colour score (1–10)", "Firmness (1–10)", "Defect score (1–10)"]
rng = np.random.default_rng(42)
n = 1400
X = pd.DataFrame({
    FEATURES[0]: rng.normal(160, 35, n).clip(60, 280),
    FEATURES[1]: rng.normal(7.5, 1.2, n).clip(4, 11),
    FEATURES[2]: rng.uniform(1, 10, n),
    FEATURES[3]: rng.uniform(1, 10, n),
    FEATURES[4]: rng.uniform(1, 10, n),
})
# Synthetic teaching labels, not real fruit-quality measurements.
score = 0.012*(X[FEATURES[0]]-100) + 0.45*X[FEATURES[1]] + 0.75*X[FEATURES[2]] + 0.85*X[FEATURES[3]] - 1.25*X[FEATURES[4]]
y = np.where(score > 9.2, "Good", "Bad")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
class_names = list(model.classes_)

st.info("Teaching simulation only: the fruit data is artificially generated. This is not a real or validated fruit-inspection system.")

st.subheader("1. Change the fruit's properties")
a, b, c = st.columns(3)
with a:
    weight = st.slider("Weight (g)", 60, 280, 160)
    size = st.slider("Size (cm)", 4.0, 11.0, 7.5, 0.1)
with b:
    colour = st.slider("Colour score", 1, 10, 7)
    firmness = st.slider("Firmness score", 1, 10, 7)
with c:
    defect = st.slider("Defect score", 1, 10, 3)

sample = pd.DataFrame([[weight, size, colour, firmness, defect]], columns=FEATURES)
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]
confidence = probabilities[class_names.index(prediction)] * 100

st.divider()
st.subheader("2. Compare the two experiences")
left, right = st.columns(2)
with left:
    st.markdown("### ❌ Without XAI")
    st.metric("Prediction", prediction)
    st.metric("Model confidence", f"{confidence:.1f}%")
    st.warning("The AI gives an answer, but this screen does not tell us which inputs influenced this prediction.")
with right:
    st.markdown("### ✅ With XAI — LIME")
    st.metric("Same prediction", prediction)
    st.write("LIME estimates which features supported or opposed this individual prediction.")
    if st.button("Generate explanation", type="primary", use_container_width=True):
        explainer = LimeTabularExplainer(
            X_train.to_numpy(), feature_names=FEATURES, class_names=class_names,
            mode="classification", discretize_continuous=True, random_state=7
        )
        explanation = explainer.explain_instance(
            sample.iloc[0].to_numpy(), model.predict_proba, num_features=5, num_samples=3000
        )
        contributions = explanation.as_list(label=class_names.index(prediction))
        labels = [x[0] for x in contributions]
        values = [x[1] for x in contributions]
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colours = ["#238b57" if v >= 0 else "#d9534f" for v in values]
        ax.barh(labels[::-1], values[::-1], color=bar_colours[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME contribution toward the selected class")
        ax.set_title(f"Local explanation for: {prediction}")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Positive values support the selected class; negative values oppose it. LIME is a local approximation, not proof of cause.")

st.divider()
st.subheader("3. Key learning points")
st.markdown("""
- **Prediction:** What does the model think?
- **Confidence:** How strongly does the model favour its chosen class? Confidence is not an explanation.
- **LIME:** Which input features influenced this particular prediction, and in which direction?
- **Fair comparison:** The model is unchanged. XAI adds an explanation; it does not guarantee that the prediction is correct.
""")
with st.expander("A short script to explain this demo to students"):
    st.write("“On the left, the AI predicts whether the fruit is good or bad. It gives us an answer and a confidence score, but not the reasons. On the right, LIME estimates which features influenced this one prediction. This is Explainable AI: we ask not only what the AI predicted, but also why.”")
