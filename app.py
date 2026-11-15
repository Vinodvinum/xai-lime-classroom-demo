import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries

MODEL_PATH = Path("models/fruit_quality.keras")
LABELS_PATH = Path("models/class_names.json")
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="Fruit Quality XAI Lab", page_icon="🍎", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 80% 0%, #17264a 0, #080d19 42%); color:#f5f7ff}
.block-container {max-width:1300px; padding-top:1.6rem}
.hero {padding:1.4rem 1.6rem;border:1px solid #66519a;border-radius:18px;
background:linear-gradient(110deg,#171b39,#10253a);margin-bottom:1rem}
.hero h1 {color:#fff;font-size:clamp(1.8rem,4vw,2.7rem);margin:0}
.hero p {color:#cbd5e8;margin:.55rem 0 0}
.note {padding:.85rem 1rem;border:1px solid #526c91;border-radius:12px;
background:#14263c;color:#dcecff;margin:.8rem 0 1rem}
.card {border:1px solid #394c69;border-radius:15px;padding:1rem;background:#101a2b}
.good {border-color:#22c985;background:#102b2a}
.bad {border-color:#f16b7e;background:#2b1a28}
.small {color:#b8c5dc;font-size:.9rem}
</style>
<div class="hero">
  <div style="color:#d6a5ff;font-weight:800;letter-spacing:.12em">IMAGE CLASSIFICATION · EXPLAINABLE AI</div>
  <h1>🍎 Fruit Quality XAI Lab</h1>
  <p>Upload a real fruit photo. See the model's prediction, then explore the image regions that influenced it with LIME.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained fruit classifier…")
def load_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, labels

def prepare_batch(images):
    batch = []
    for image in images:
        image = Image.fromarray(image.astype("uint8")).convert("RGB").resize(IMAGE_SIZE)
        arr = tf.keras.utils.img_to_array(image)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        batch.append(arr)
    return np.asarray(batch, dtype=np.float32)

def predict_images(images, model):
    return model.predict(prepare_batch(images), verbose=0)

model, class_names = load_model()

with st.sidebar:
    st.title("🍏 Fruit Quality XAI")
    st.caption("Classroom research prototype")
    st.markdown("**Model**")
    st.write("MobileNetV2 transfer learning")
    st.markdown("**Dataset**")
    st.write("Fresh and Rotten Fruits (16 classes)")
    st.markdown("**Explanation**")
    st.write("LIME image superpixels")
    st.divider()
    st.caption("Predictions are visual estimates, not food-safety certification.")

if model is None:
    st.error("The trained model is not installed yet.")
    st.markdown("### One-time model setup")
    st.markdown("""
    1. Download the real dataset from [Mendeley Data](https://data.mendeley.com/datasets/bdd69gyhv8/1).
    2. Extract the **Original Image.zip** so each class has its own folder (for example, Fresh Apple and Rotten Apple).
    3. Run `python train.py --data_dir /path/to/Original\ Image`.
    4. Copy the generated `models/fruit_quality.keras` and `models/class_names.json` into this repository's `models/` folder and push them to GitHub.
    5. Streamlit will redeploy and the upload predictor will become available.
    """)
    st.info("The GitHub repository you supplied contains only a README; it did not include a trained model or training features. This app will not make up predictions until the real model is installed.")
    st.stop()

st.markdown('<div class="note"><b>Dataset note:</b> This model is trained on a published dataset of eight fruit types, each labelled fresh or rotten. It cannot assess fruits outside those 16 classes reliably.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a fruit image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns([0.8, 1.2], gap="large")
    with left:
        st.subheader("1. Uploaded fruit")
        st.image(image, use_container_width=True)
        st.caption(f"{image.width} × {image.height} pixels")
    with st.spinner("Analysing image…"):
        original = np.asarray(image.resize(IMAGE_SIZE))
        probabilities = predict_images([original], model)[0]
        top_idx = int(np.argmax(probabilities))
        predicted_label = class_names[top_idx]
        confidence = float(probabilities[top_idx])
        top_indices = np.argsort(probabilities)[::-1][:5]

    with right:
        st.subheader("2. Same model — two views")
        no_xai, with_xai = st.columns(2)
        with no_xai:
            st.markdown('<div class="card bad">', unsafe_allow_html=True)
            st.markdown("### ❌ Without XAI")
            st.markdown(f"## {predicted_label}")
            st.metric("Model confidence", f"{confidence:.1%}")
            st.caption("Prediction only. No visual reason is shown.")
            st.markdown("</div>", unsafe_allow_html=True)
        with with_xai:
            st.markdown('<div class="card good">', unsafe_allow_html=True)
            st.markdown("### ✅ With XAI — LIME")
            st.markdown(f"## {predicted_label}")
            st.metric("Same model confidence", f"{confidence:.1%}")
            st.caption("Same prediction, with an image-region explanation below.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("3. LIME explanation — which image regions influenced the prediction?")
    with st.spinner("LIME is analysing image regions. This may take a few seconds…"):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            original,
            classifier_fn=lambda imgs: predict_images(imgs, model),
            top_labels=1,
            hide_color=0,
            num_samples=500,
            batch_size=32,
        )
        temp, mask = explanation.get_image_and_mask(
            top_idx, positive_only=False, num_features=8, hide_rest=False
        )
        lime_image_result = mark_boundaries(temp / 255.0, mask)

    c1, c2 = st.columns(2)
    with c1:
        st.image(original, caption="Original uploaded image", use_container_width=True)
    with c2:
        st.image(lime_image_result, caption="LIME superpixel explanation", use_container_width=True)

    st.caption("LIME highlights image regions that locally support or oppose the selected class. It is an approximate explanation, not proof of causation.")
    st.subheader("4. Top model probabilities")
    probability_rows = [
        {"Class": class_names[i], "Probability": f"{probabilities[i]:.2%}"}
        for i in top_indices
    ]
    st.dataframe(probability_rows, hide_index=True, use_container_width=True)

    st.markdown("""
    **How to explain this to students**
    - Without XAI: the model predicts the fruit class and freshness label.
    - With XAI: the same model prediction is shown with LIME's highlighted image regions.
    - LIME is not a second model and does not improve the prediction; it explains one prediction locally.
    """)
else:
    st.info("Upload a fruit photo to begin. The prediction and LIME explanation will appear after the image is analysed.")

st.caption("Educational prototype · Real image dataset · MobileNetV2 · LIME · Not a food-safety assessment")
