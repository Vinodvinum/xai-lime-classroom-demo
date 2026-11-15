import os
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries, slic
from transformers import AutoImageProcessor, AutoModelForImageClassification


DEFAULT_MODEL_ID = "hafeezaismail/fruit-freshness-convnext"
try:
    MODEL_ID = st.secrets.get("FRUIT_MODEL_ID", os.getenv("FRUIT_MODEL_ID", DEFAULT_MODEL_ID))
except Exception:
    MODEL_ID = os.getenv("FRUIT_MODEL_ID", DEFAULT_MODEL_ID)
DATASET_ID = "Project-AgML/fresh_rotten_fruit_classification"
SUPPORTED_FRUITS = [
    "Apple", "Banana", "Grape", "Guava",
    "Jujube", "Orange", "Pomegranate", "Strawberry",
]

st.set_page_config(
    page_title="Fruit Quality XAI Lab",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #080d19; --panel: #111a2b; --panel2: #172338;
        --line: #2c3a54; --text: #f4f6ff; --muted: #aab6cc;
        --purple: #bd6cff; --green: #42d991; --red: #ff6c80;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 88% 0%, rgba(100,60,170,.20), transparent 34%),
          radial-gradient(ellipse at 0% 38%, rgba(30,110,150,.12), transparent 30%),
          var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1420px; padding-top: 1.6rem; padding-bottom: 3rem; }
    h1,h2,h3,p,label { color: var(--text); }
    .hero {
      padding: 1.45rem 1.6rem; border: 1px solid rgba(189,108,255,.45);
      border-radius: 20px; background: linear-gradient(115deg,rgba(31,37,65,.97),rgba(15,31,48,.97));
      box-shadow: 0 14px 44px rgba(0,0,0,.22); margin-bottom: 1rem;
    }
    .eyebrow { color:#d8aaff; font-size:.76rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
    .hero-title { font-size:clamp(1.75rem,4vw,2.7rem); font-weight:850; line-height:1.12; color:#fff; margin:.35rem 0; }
    .hero-copy { color:#c7d2e5; font-size:1rem; line-height:1.55; }
    .pill-row { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.9rem; }
    .pill { border:1px solid #43516e; border-radius:999px; color:#e5eaf7; background:rgba(7,14,27,.42); padding:.35rem .7rem; font-size:.78rem; font-weight:650; }
    .notice { border:1px solid #49688e; border-radius:13px; background:rgba(31,67,102,.32); color:#dcecff; padding:.8rem .95rem; line-height:1.5; margin:.9rem 0 1.1rem 0; }
    .section-label { color:#d9a6ff; font-size:.76rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; margin-bottom:.35rem; }
    .panel-heading { font-size:1.2rem; font-weight:780; color:#f7f5ff; margin:0 0 .35rem 0; }
    .muted { color:var(--muted); font-size:.9rem; line-height:1.5; }
    .result-card { padding:1rem; border-radius:16px; height:100%; }
    .plain { border:1px solid rgba(255,108,128,.72); background:linear-gradient(145deg,rgba(48,28,43,.88),rgba(22,25,39,.94)); }
    .explained { border:1px solid rgba(66,217,145,.75); background:linear-gradient(145deg,rgba(17,48,44,.88),rgba(17,31,42,.95)); }
    .result-label { font-size:.9rem; font-weight:800; color:#fff; margin-bottom:.2rem; }
    .result-sub { color:#bbc7d9; font-size:.78rem; margin-bottom:.7rem; }
    .class-name { font-size:1.35rem; font-weight:850; color:#a6ffbf; line-height:1.3; }
    .confidence { color:#e1e8f5; font-size:.85rem; margin-top:.6rem; }
    .takeaway { border:1px solid rgba(189,108,255,.6); border-radius:15px; padding:1rem 1.1rem; background:linear-gradient(120deg,rgba(71,39,112,.35),rgba(23,46,69,.45)); color:#f4efff; line-height:1.6; }
    .takeaway strong { color:#e4bdff; }
    .stButton > button { border:1px solid #a45cf0; border-radius:12px; background:linear-gradient(100deg,#8f43d7,#5b54d9); color:#fff; font-weight:750; min-height:3rem; }
    div[data-testid="stTabs"] button { color:#dfe7f7; font-weight:700; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:#e4bdff; border-bottom-color:#bd6cff; }
    @media(max-width:760px) {
      .block-container { padding:1rem .8rem 2rem .8rem; }
      .hero { padding:1.1rem .95rem; border-radius:16px; }
      .hero-copy { font-size:.92rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_classifier(model_id: str):
    """Load the published ConvNeXt fruit-freshness model once per app worker."""
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    return processor, model, id2label


def model_probabilities(image_arrays, processor, model):
    """LIME callback: convert a batch of RGB arrays into class probabilities."""
    if isinstance(image_arrays, np.ndarray) and image_arrays.ndim == 3:
        image_arrays = [image_arrays]

    pil_images = []
    for array in image_arrays:
        arr = np.asarray(array)
        if arr.dtype != np.uint8:
            if arr.max(initial=0) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(arr).convert("RGB"))

    inputs = processor(images=pil_images, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities.cpu().numpy()


def human_label(label: str) -> str:
    parts = label.replace("-", "_").split("_")
    if len(parts) >= 2 and parts[0].lower() in {"fresh", "rotten"}:
        return f"{parts[0].title()} {parts[1].title()}"
    if len(parts) >= 2:
        return f"{parts[0].title()} {parts[1].title()}"
    return label.replace("_", " ").title()


def make_lime_overlay(image_array: np.ndarray, mask: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Tint LIME-positive superpixels green and negative ones red."""
    base = image_array.astype(np.float32) / 255.0
    overlay = base.copy()
    positive = mask == 1
    negative = mask == -1
    green = np.array([0.10, 0.95, 0.48], dtype=np.float32)
    red = np.array([1.00, 0.20, 0.34], dtype=np.float32)
    overlay[positive] = 0.48 * overlay[positive] + 0.52 * green
    overlay[negative] = 0.48 * overlay[negative] + 0.52 * red
    overlay = mark_boundaries(overlay, segments, color=(0.75, 0.85, 1.0), mode="thick")
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


def run_lime(image: Image.Image, processor, model, predicted_index: int):
    """Explain the selected prediction using LIME image superpixels."""
    small = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(small, dtype=np.uint8)

    segmenter = partial(
        slic,
        n_segments=65,
        compactness=12,
        sigma=0.8,
        start_label=0,
        channel_axis=-1,
    )
    explainer = LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        image=array,
        classifier_fn=lambda batch: model_probabilities(batch, processor, model),
        labels=(predicted_index,),
        top_labels=None,
        hide_color=0,
        num_samples=400,
        batch_size=16,
        segmentation_fn=segmenter,
    )
    _, mask = explanation.get_image_and_mask(
        label=predicted_index,
        positive_only=False,
        num_features=12,
        hide_rest=False,
        min_weight=0.0,
    )
    segments = explanation.segments
    overlay = make_lime_overlay(array, mask, segments)
    contributions = explanation.local_exp[predicted_index]
    contributions = sorted(contributions, key=lambda pair: abs(pair[1]), reverse=True)
    return overlay, contributions[:8]


def show_prediction_card(kind: str, label: str, confidence: float, explained: bool):
    style = "explained" if explained else "plain"
    heading = "✓ WITH XAI — LIME" if explained else "✕ WITHOUT XAI"
    subtitle = "Prediction + local image explanation" if explained else "Prediction only"
    color = "#a6ffbf" if explained else "#ffd1d8"
    st.markdown(
        f"""
        <div class="result-card {style}">
          <div class="result-label">{heading}</div>
          <div class="result-sub">{subtitle}</div>
          <div class="class-name" style="color:{color}">{label}</div>
          <div class="confidence">Model confidence: <strong>{confidence:.1%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Computer vision · Explainable AI · Classroom edition</div>
      <div class="hero-title">Fruit Quality XAI Lab 🍎</div>
      <div class="hero-copy">
        Upload a real fruit photo. The same image classifier predicts fruit type
        and freshness, then LIME highlights image regions that influenced its result.
      </div>
      <div class="pill-row">
        <span class="pill">Real image upload</span>
        <span class="pill">8 supported fruit types</span>
        <span class="pill">16 fruit-quality labels</span>
        <span class="pill">With &amp; without XAI</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
      <strong>Use as a learning prototype only.</strong> This model predicts how a fruit
      looks in an image; it cannot determine internal spoilage, contamination, pesticides,
      bacteria, or whether food is safe to eat. Always use human judgement.
    </div>
    """,
    unsafe_allow_html=True,
)

upload_tab, model_tab, guide_tab, script_tab = st.tabs(
    ["🍎 Upload & analyse", "🧠 Dataset & model", "💡 How LIME works", "🎤 Presentation script"]
)

with upload_tab:
    left, right = st.columns([0.85, 1.4], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="section-label">01 · Upload</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-heading">Choose one fruit image</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="muted">Use one fruit, close to the camera, in good light. '
                'Supported formats: JPG, JPEG, PNG, WEBP.</p>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Upload a fruit photo",
                type=["jpg", "jpeg", "png", "webp"],
                help="For best results, upload one fruit with a simple background.",
            )
            if uploaded is not None:
                try:
                    image = Image.open(uploaded).convert("RGB")
                    st.image(image, caption=uploaded.name, use_container_width=True)
                    st.caption(f"Image size: {image.width} × {image.height} pixels")
                except Exception:
                    image = None
                    st.error("This image could not be opened. Please upload a JPG or PNG photo.")
            else:
                image = None
                st.info("Your uploaded fruit image will appear here.")
            analyse = st.button(
                "Analyse image + generate LIME explanation",
                type="primary",
                use_container_width=True,
                disabled=image is None,
            )

    with right:
        with st.container(border=True):
            st.markdown('<div class="section-label">02 · Same image, two views</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-heading">Prediction comparison</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="muted">Both cards use exactly the same image and model. '
                'The XAI view adds an explanation.</p>',
                unsafe_allow_html=True,
            )

            if analyse and image is not None:
                try:
                    with st.spinner("Loading the image model and analysing your photo…"):
                        processor, model, id2label = load_classifier(MODEL_ID)
                        model_input = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                        probabilities = model_probabilities(
                            [np.asarray(model_input, dtype=np.uint8)], processor, model
                        )[0]
                        predicted_index = int(np.argmax(probabilities))
                        predicted_label = id2label[predicted_index]
                        confidence = float(probabilities[predicted_index])

                    st.markdown("#### Prediction results")
                    no_xai, with_xai = st.columns(2, gap="small")
                    with no_xai:
                        show_prediction_card(
                            "without", human_label(predicted_label), confidence, explained=False
                        )
                        st.caption("This view gives the model's answer only.")
                    with with_xai:
                        show_prediction_card(
                            "with", human_label(predicted_label), confidence, explained=True
                        )
                        st.caption("Same answer, now with a LIME explanation.")

                    st.markdown("#### LIME explanation — image regions")
                    st.markdown(
                        '<p class="muted">Green regions support the selected class; '
                        'red regions oppose it. These are approximate local contributions.</p>',
                        unsafe_allow_html=True,
                    )
                    with st.spinner("LIME is testing image regions to explain this prediction…"):
                        overlay, region_weights = run_lime(
                            image, processor, model, predicted_index
                        )

                    original_col, explanation_col = st.columns(2, gap="small")
                    with original_col:
                        st.image(image, caption="Original uploaded image", use_container_width=True)
                    with explanation_col:
                        st.image(
                            overlay,
                            caption="LIME overlay: green supports · red opposes",
                            use_container_width=True,
                        )

                    st.markdown("#### Top model class probabilities")
                    top_indices = np.argsort(probabilities)[::-1][:5]
                    chart_df = pd.DataFrame(
                        {
                            "Class": [human_label(id2label[int(i)]) for i in top_indices],
                            "Probability": [float(probabilities[i]) for i in top_indices],
                        }
                    ).set_index("Class")
                    st.bar_chart(chart_df, horizontal=True)

                    if confidence < 0.60:
                        st.warning(
                            "The model is not confident about this image. Treat the result as uncertain "
                            "and ask a person to inspect it."
                        )
                    else:
                        st.info(
                            "The percentage is the model's confidence for its chosen class, "
                            "not a guarantee that the fruit is fresh or safe."
                        )

                    if region_weights:
                        with st.expander("Inspect LIME's region contribution values"):
                            st.write(
                                "These values belong to image superpixels, not named human features "
                                "such as colour or texture."
                            )
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {
                                            "Region ID": int(region_id),
                                            "Contribution": float(weight),
                                            "Direction": "Supports" if weight > 0 else "Opposes",
                                        }
                                        for region_id, weight in region_weights
                                    ]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                except Exception as error:
                    st.error(
                        "The model could not finish the analysis. Check the Streamlit app logs "
                        "under Manage app → Logs. Technical detail: " + str(error)[:500]
                    )
            else:
                st.info(
                    "1. Upload a fruit photo.\n\n"
                    "2. Tap **Analyse image + generate LIME explanation**.\n\n"
                    "3. Compare both prediction cards and inspect the highlighted regions."
                )

    st.markdown(
        """
        <div class="takeaway">
          <strong>Remember:</strong> Without XAI, the model tells us <strong>WHAT</strong>
          it predicts. With LIME, we also get an approximate view of <strong>WHICH IMAGE
          REGIONS</strong> influenced that prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

with model_tab:
    st.markdown("### The real dataset used by this model")
    st.write(
        "Project-AgML's Fresh and Rotten Fruits dataset contains 3,200 original images "
        "covering eight fruit types, labelled Fresh or Rotten. The dataset card also lists "
        "an augmented version with 12,335 images. This app uses a published ConvNeXt-Tiny "
        "checkpoint fine-tuned on the raw dataset."
    )
    st.markdown("**Supported fruit types:** " + ", ".join(SUPPORTED_FRUITS))
    st.markdown(
        "**Model output:** 16 classes — each supported fruit paired with Fresh or Rotten."
    )
    st.markdown(
        "**Model architecture:** ConvNeXt-Tiny image classifier. "
        "**Visual input:** RGB pixels resized and normalized by the model's image processor."
    )
    st.markdown(
        "**What are the features?** The model learns visual patterns from pixels during training. "
        "This repository did not define a manual list of numerical features. LIME highlights "
        "image superpixels that locally support or oppose the selected class."
    )
    st.markdown(
        f"- Dataset: [{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID})"
    )
    st.markdown(
        f"- Model weights and model card: [hafeezaismail/fruit-freshness-convnext]"
        "(https://huggingface.co/hafeezaismail/fruit-freshness-convnext)"
    )
    st.markdown(
        "- Dataset licence: CC BY 4.0. Model repository licence: Apache 2.0."
    )
    st.warning(
        "The model card reports 99.38% accuracy on its held-out test split. "
        "That number is not a promise of performance on your own phone photos. "
        "The model card warns that its training images have relatively clean backgrounds "
        "and that real-world performance may be lower."
    )

with guide_tab:
    st.markdown("### What LIME is showing")
    st.write(
        "LIME stands for Local Interpretable Model-agnostic Explanations. For this image, "
        "it divides the picture into small connected regions (superpixels), changes or hides "
        "some regions, and observes how the model's class probabilities change."
    )
    st.markdown(
        """
        - **Green regions:** support the displayed prediction in this local explanation.
        - **Red regions:** oppose the displayed prediction.
        - **More visible area:** means more selected superpixels are being shown, not necessarily greater certainty.
        - **Important:** LIME is an approximation for this one image, not a causal proof.
        """
    )
    st.info(
        "The model learns from image pixels; LIME helps us inspect regions. "
        "It does not provide a verified human explanation like 'this spot is definitely rot'."
    )

with script_tab:
    st.markdown("### A classroom explanation")
    st.markdown(
        """
        “I am uploading a real fruit photograph. The image model looks at the picture and
        predicts the fruit type and whether it looks fresh or rotten.

        On the **Without XAI** side, we see only the prediction and confidence.

        On the **With XAI** side, the prediction is exactly the same because we use the same
        model. LIME adds a visual explanation. Green regions are areas that supported the
        prediction, and red regions are areas that opposed it.

        This demonstrates the main idea of Explainable AI: we want to understand not only
        what a model predicts, but also which parts of the input influenced that prediction.”
        """
    )

st.caption(
    "Educational prototype · ConvNeXt-Tiny · Project-AgML Fresh/Rotten Fruits dataset · LIME Image Explainer"
)
