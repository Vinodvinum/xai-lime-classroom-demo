import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURES = [
    "Weight (g)",
    "Size (cm)",
    "Colour score",
    "Firmness score",
    "Defect score",
]

st.set_page_config(
    page_title="Explainable AI Classroom Demo",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #080d19;
        --panel: #111a2b;
        --panel-soft: #172338;
        --line: #2c3a54;
        --text: #f4f6ff;
        --muted: #aab6cc;
        --purple: #bd6cff;
        --cyan: #51d6e8;
        --green: #42d991;
        --red: #ff6c80;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(104, 63, 177, .20), transparent 34%),
            radial-gradient(ellipse at 0% 35%, rgba(33, 117, 155, .12), transparent 28%),
            var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label { color: var(--text); }
    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(189,108,255,.42);
        border-radius: 22px;
        background: linear-gradient(115deg, rgba(31,37,65,.97), rgba(15,31,48,.97));
        box-shadow: 0 14px 44px rgba(0,0,0,.22);
        margin-bottom: 1rem;
    }
    .eyebrow {
        color: #d8aaff;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .hero-title {
        font-size: clamp(1.7rem, 4vw, 2.65rem);
        font-weight: 850;
        line-height: 1.12;
        margin: 0;
        color: #ffffff;
    }
    .hero-copy {
        color: #c7d2e5;
        font-size: 1.04rem;
        line-height: 1.55;
        margin: .65rem 0 0 0;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1rem;
    }
    .pill {
        border: 1px solid #43516e;
        border-radius: 999px;
        color: #e5eaf7;
        background: rgba(7,14,27,.42);
        padding: .38rem .72rem;
        font-size: .79rem;
        font-weight: 650;
    }
    .notice {
        border: 1px solid #49688e;
        border-radius: 14px;
        background: rgba(31,67,102,.32);
        color: #dcecff;
        padding: .85rem 1rem;
        line-height: 1.5;
        margin: .9rem 0 1.2rem 0;
    }
    .section-label {
        color: #d9a6ff;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .11em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .panel-heading {
        font-size: 1.25rem;
        font-weight: 780;
        color: #f7f5ff;
        margin: 0 0 .25rem 0;
    }
    .muted {
        color: var(--muted);
        font-size: .92rem;
        line-height: 1.5;
    }
    .comparison {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: .8rem;
        margin: .75rem 0 1rem 0;
    }
    .result-card {
        padding: 1rem;
        border-radius: 17px;
        min-width: 0;
        background: linear-gradient(145deg, rgba(35,28,43,.94), rgba(22,25,39,.94));
    }
    .result-card.plain {
        border: 1px solid rgba(255,108,128,.72);
        box-shadow: inset 0 0 22px rgba(255,108,128,.06);
    }
    .result-card.explained {
        border: 1px solid rgba(66,217,145,.75);
        background: linear-gradient(145deg, rgba(17,48,44,.92), rgba(17,31,42,.95));
        box-shadow: inset 0 0 22px rgba(66,217,145,.07);
    }
    .result-label {
        font-size: .88rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: .25rem;
    }
    .result-sub {
        color: #bbc7d9;
        font-size: .76rem;
        margin-bottom: .85rem;
    }
    .prediction {
        border-radius: 12px;
        padding: .62rem .7rem;
        background: rgba(255,255,255,.07);
        text-align: center;
        font-size: 1.48rem;
        font-weight: 850;
        color: #ffffff;
        margin-bottom: .7rem;
    }
    .confidence-row {
        display: flex;
        justify-content: space-between;
        gap: .5rem;
        color: #d8e0ef;
        font-size: .8rem;
    }
    .confidence-track {
        height: 8px;
        background: #354258;
        border-radius: 99px;
        overflow: hidden;
        margin-top: .35rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #8d63ed, #46d99b);
    }
    .lime-note {
        border-left: 3px solid var(--purple);
        background: rgba(135,83,195,.12);
        color: #e5dcf8;
        padding: .75rem .9rem;
        border-radius: 0 12px 12px 0;
        line-height: 1.5;
        font-size: .9rem;
        margin-top: .75rem;
    }
    .takeaway {
        border: 1px solid rgba(189,108,255,.6);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: linear-gradient(120deg, rgba(71,39,112,.35), rgba(23,46,69,.45));
        color: #f4efff;
        line-height: 1.6;
    }
    .takeaway strong { color: #e4bdff; }
    div[data-testid="stTabs"] button {
        color: #dfe7f7;
        font-weight: 700;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #e4bdff;
        border-bottom-color: #bd6cff;
    }
    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #e4bdff;
        font-weight: 800;
    }
    .stButton > button {
        border: 1px solid #a45cf0;
        border-radius: 12px;
        background: linear-gradient(100deg, #8f43d7, #5b54d9);
        color: #ffffff;
        font-weight: 750;
    }
    @media (max-width: 760px) {
        .block-container { padding: 1rem .85rem 2rem .85rem; }
        .hero { padding: 1.2rem 1rem; border-radius: 16px; }
        .comparison { grid-template-columns: 1fr; }
        .hero-copy { font-size: .95rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Create one deterministic teaching model and its LIME reference data."""
    rng = np.random.default_rng(42)
    row_count = 1400

    data = pd.DataFrame(
        {
            "Weight (g)": rng.normal(160, 35, row_count).clip(60, 280),
            "Size (cm)": rng.normal(7.5, 1.2, row_count).clip(4, 11),
            "Colour score": rng.uniform(1, 10, row_count),
            "Firmness score": rng.uniform(1, 10, row_count),
            "Defect score": rng.uniform(1, 10, row_count),
        }
    )

    # Synthetic rule used only to create classroom example labels.
    score = (
        0.012 * (data["Weight (g)"] - 100)
        + 0.45 * data["Size (cm)"]
        + 0.75 * data["Colour score"]
        + 0.85 * data["Firmness score"]
        - 1.25 * data["Defect score"]
    )
    labels = np.where(score > 9.2, "Good", "Bad")

    x_train, _, y_train, _ = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)
    return classifier, x_train, list(classifier.classes_)


@st.cache_data(show_spinner=False)
def lime_explanation(weight, size, colour, firmness, defect):
    """Return LIME's local feature contributions for the selected fruit."""
    classifier, x_train, class_names = load_model()
    explainer = LimeTabularExplainer(
        training_data=x_train.to_numpy(dtype=float),
        feature_names=FEATURES,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=7,
    )

    row = np.array([weight, size, colour, firmness, defect], dtype=float)
    predicted_class = classifier.predict(row.reshape(1, -1))[0]
    class_index = class_names.index(predicted_class)

    explanation = explainer.explain_instance(
        data_row=row,
        predict_fn=classifier.predict_proba,
        labels=(class_index,),
        num_features=len(FEATURES),
        num_samples=1800,
    )
    return explanation.as_list(label=class_index)


def draw_lime_chart(contributions, prediction):
    labels = [item[0] for item in contributions]
    values = [float(item[1]) for item in contributions]
    colours = ["#42d991" if value >= 0 else "#ff6c80" for value in values]

    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    fig.patch.set_facecolor("#111a2b")
    ax.set_facecolor("#111a2b")

    positions = np.arange(len(labels))
    ax.barh(positions, values, color=colours, height=0.58, edgecolor="none")
    ax.axvline(0, color="#aab6cc", linewidth=1, alpha=0.75)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, color="#e8edf8", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"Contribution toward the {prediction} prediction",
        color="#b9c5d9",
        labelpad=10,
        fontsize=9,
    )
    ax.tick_params(axis="x", colors="#aab6cc", labelsize=8)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.grid(axis="x", color="#71809a", alpha=0.16, linestyle="--")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    max_abs = max([abs(value) for value in values] + [0.1])
    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)

    for index, value in enumerate(values):
        offset = max_abs * 0.045
        ax.text(
            value + (offset if value >= 0 else -offset),
            index,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            color="#cfe7dc" if value >= 0 else "#ffd0d7",
            fontsize=8,
            fontweight="bold",
        )

    fig.tight_layout(pad=1.5)
    return fig


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">XAI learning lab · classroom edition</div>
      <div class="hero-title">Explainable AI, made visible</div>
      <div class="hero-copy">
        Compare one AI prediction with and without an explanation.
        Then explore how LIME estimates which features influenced that decision.
      </div>
      <div class="pill-row">
        <span class="pill">● Same AI model</span>
        <span class="pill">↔ Two views</span>
        <span class="pill">▥ LIME explanation</span>
        <span class="pill">✦ Learning demo</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice"><strong>Teaching simulation:</strong> '
    "The fruit data is artificially generated. This is not a real or validated "
    "fruit-inspection system.</div>",
    unsafe_allow_html=True,
)

demo_tab, guide_tab, script_tab = st.tabs(
    ["🍎 Live demo", "💡 How to read LIME", "🎤 Presentation script"]
)

with demo_tab:
    input_col, result_col = st.columns([0.82, 1.35], gap="large")

    with input_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">01 · Input</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-heading">Set fruit properties</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="muted">Move any slider. The prediction and explanation '
                "will update automatically.</p>",
                unsafe_allow_html=True,
            )
            weight = st.slider("Weight (g)", 60, 280, 160, 1)
            size = st.slider("Size (cm)", 4.0, 11.0, 7.5, 0.1)
            colour = st.slider("Colour score (1–10)", 1, 10, 7, 1)
            firmness = st.slider("Firmness score (1–10)", 1, 10, 7, 1)
            defect = st.slider("Defect score (1–10)", 1, 10, 3, 1)

            st.markdown(
                '<div class="muted">Tip: Try increasing the defect score or '
                "reducing firmness to see how the model responds.</div>",
                unsafe_allow_html=True,
            )

    classifier, _, class_names = load_model()
    sample = pd.DataFrame(
        [[weight, size, colour, firmness, defect]],
        columns=FEATURES,
    )
    prediction = classifier.predict(sample)[0]
    probabilities = classifier.predict_proba(sample)[0]
    confidence = float(probabilities[class_names.index(prediction)] * 100)

    with result_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">02 · Prediction</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-heading">Same model, two experiences</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="comparison">
                  <div class="result-card plain">
                    <div class="result-label">✕ Without XAI</div>
                    <div class="result-sub">Prediction only</div>
                    <div class="prediction">{prediction}</div>
                    <div class="confidence-row"><span>Model confidence</span><strong>{confidence:.1f}%</strong></div>
                    <div class="confidence-track"><div class="confidence-fill" style="width:{confidence:.1f}%;background:linear-gradient(90deg,#e95c7d,#ff8c9a)"></div></div>
                    <div class="result-sub" style="margin-top:.7rem;margin-bottom:0">The answer is shown, but no reason is provided.</div>
                  </div>
                  <div class="result-card explained">
                    <div class="result-label">✓ With XAI · LIME</div>
                    <div class="result-sub">Same prediction + explanation</div>
                    <div class="prediction">{prediction}</div>
                    <div class="confidence-row"><span>Model confidence</span><strong>{confidence:.1f}%</strong></div>
                    <div class="confidence-track"><div class="confidence-fill" style="width:{confidence:.1f}%"></div></div>
                    <div class="result-sub" style="margin-top:.7rem;margin-bottom:0">The prediction is accompanied by a local explanation.</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-label">03 · Explanation</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel-heading">Why did the model predict {prediction}?</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="muted">This chart is generated automatically for the '
                "current slider values. No extra button is needed.</p>",
                unsafe_allow_html=True,
            )

            with st.spinner("LIME is preparing the explanation…"):
                contributions = lime_explanation(weight, size, colour, firmness, defect)

            chart = draw_lime_chart(contributions, prediction)
            st.pyplot(chart, use_container_width=True)
            plt.close(chart)

            st.markdown(
                '<div class="lime-note"><strong>How to read it:</strong> '
                'Green bars support the displayed prediction. Red bars oppose it. '
                "Longer bars indicate a larger contribution in this local explanation.</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="takeaway">
          <strong>Key takeaway:</strong> AI tells us <strong>WHAT</strong> it predicts.
          Explainable AI with LIME helps us understand <strong>WHICH features</strong>
          influenced that particular prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

with guide_tab:
    st.markdown("### Understand the LIME chart")
    st.write(
        "LIME (Local Interpretable Model-agnostic Explanations) explains one "
        "prediction at a time. It creates a simple local approximation around "
        "the selected example to estimate which input features support or oppose "
        "the model's chosen class."
    )
    st.markdown(
        """
        **Read the bars this way:**

        - **Green / positive:** pushes the explanation toward the prediction shown.
        - **Red / negative:** pushes the explanation away from the prediction shown.
        - **Longer bar:** larger estimated contribution in this explanation.
        - **Different slider values:** may produce a different prediction and a different explanation.

        **Remember:** LIME is an approximation of the model's behaviour near this one example.
        It does not prove cause and effect, and it does not guarantee that the prediction is correct.
        """
    )
    st.info(
        "The confidence percentage is the model's predicted probability for its selected class. "
        "It is not the explanation and should not be treated as proof that the model is correct."
    )

with script_tab:
    st.markdown("### A short explanation for your students")
    st.markdown(
        """
        “First, look at the left side of our comparison: the AI predicts whether
        the fruit is Good or Bad and shows its confidence. But it does not explain
        which inputs influenced its decision.

        Now look at the LIME chart. The model is the same. LIME gives us a local
        explanation for this particular prediction. Green bars support the
        prediction, while red bars oppose it.

        This is the basic idea of Explainable AI: we do not only ask what the AI
        predicted; we also try to understand why it made that prediction.”
        """
    )
    st.markdown("### Three points to remember")
    st.markdown(
        """
        1. A prediction is the model's answer.
        2. Confidence is a probability estimate, not a reason.
        3. LIME gives an approximate, local explanation of the model's behaviour.
        """
    )

st.caption(
    "Classroom prototype · Synthetic data · Random Forest classifier · LIME local explanation"
)
