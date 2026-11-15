# Fruit Quality XAI Lab — Real Image Upload + LIME

This is the working image-classification version of the original **Fruits_quality_detection** project idea.

The original `Vinodvinum/Fruits_quality_detection` repository currently contains only a short README; it does not contain a trained model, training script, feature definition, or dataset. This app therefore uses an explicitly attributed public fruit-freshness model and dataset rather than claiming to reuse missing project files.

## What the app does

1. Upload one fruit image (JPG, JPEG, PNG, or WEBP).
2. Run one fruit image classifier.
3. Show the same prediction in two cards:
   - **Without XAI:** predicted fruit-quality class and model confidence.
   - **With XAI:** same prediction plus a LIME image explanation.
4. Show the top class probabilities and an overlay in which green regions support the prediction and red regions oppose it.

The model predicts 16 labels: fresh and rotten for apple, banana, grape, guava, jujube, orange, pomegranate, and strawberry.

## Model and real dataset

- Dataset: [Project-AgML/fresh_rotten_fruit_classification](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification)
- Model: [hafeezaismail/fruit-freshness-convnext](https://huggingface.co/hafeezaismail/fruit-freshness-convnext)
- Architecture: ConvNeXt-Tiny image classifier
- Raw dataset: 3,200 images; 8 fruit types; Fresh/Rotten labels
- Dataset licence: CC BY 4.0
- Model repository licence: Apache 2.0

The model card reports 99.38% accuracy on its held-out test split. That is not a guarantee of performance on new phone photos. Its model card warns that it was trained on a small dataset with relatively clean backgrounds and may be unreliable on cluttered scenes, unusual lighting, or unsupported produce.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first analysis downloads the model weights from Hugging Face (about 112 MB). The first upload may therefore take longer than later ones.

## Deploy on Streamlit Community Cloud

The repository's main file is `app.py`. Streamlit installs dependencies from `requirements.txt`. A new commit to `main` should trigger deployment automatically.

## How to take a good test photo

- Photograph one fruit at a time.
- Keep the fruit large and clearly visible.
- Use even light and a simple background.
- Avoid hands covering the fruit, severe blur, and clutter.
- Supported fruit labels are limited to the eight listed above.

## What the explanation means

LIME Image Explainer segments the photo into superpixels and estimates which regions support or oppose the current prediction. It is a local approximation. The highlighted regions are not proof that a specific spot is rot or that the fruit is safe.

## Important limitation

This is a computer-vision classroom prototype, not a food-safety or commercial grading system. Visual appearance cannot establish internal spoilage, contamination, pesticides, bacteria, or whether food is safe to eat. Human inspection remains necessary.

## Optional: train your own checkpoint

A reproducible fine-tuning script is included at `training/train_model.py`. It loads the raw Project-AgML data, creates 16 fruit/state labels, makes a stratified 85/15 train/evaluation split, and fine-tunes ConvNeXt-Tiny. It is intended for Google Colab or Kaggle with a GPU; training is not run by the Streamlit app.

```bash
pip install -r training/requirements.txt
python training/train_model.py --output_dir fruit-convnext-custom --epochs 4
```

After training, upload the saved model and processor folder to your own Hugging Face model repository. Set `FRUIT_MODEL_ID` in the Streamlit Community Cloud app Secrets to that model ID to use your checkpoint. Without that secret, the app uses the attributed public checkpoint listed above.

The optional script has been committed but has not been executed in this chat. Do not report a custom training run or custom accuracy until you run it and review its evaluation metrics.
