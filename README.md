# Fruit Quality XAI Lab — Real Image Upload + LIME

This application is designed for the **Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality** (Sultana, Jahan & Uddin, 2022).

- 8 fruit types × fresh/rotten = 16 image classes.
- 3,200 original images and 12,335 augmented images are described by the dataset authors.
- Dataset: https://data.mendeley.com/datasets/bdd69gyhv8/1
- DOI: 10.17632/bdd69gyhv8.1
- License listed by Mendeley Data: CC BY 4.0.

## Important repository inspection result

The original `Vinodvinum/Fruits_quality_detection` repository contained only a README with the stated goal of classifying fruit as good or bad. It did not contain source code, training scripts, a dataset, a model checkpoint, or documented feature engineering. This project therefore implements the missing image-classification pipeline rather than claiming to reuse a model that was not present.

## Quick training option: Google Colab

Open the [Colab training notebook](https://colab.research.google.com/github/Vinodvinum/xai-lime-classroom-demo/blob/main/Train_Fruit_Quality_Model_Colab.ipynb). Save the original dataset ZIP to Google Drive, set the ZIP and extracted class-folder paths in the notebook, run the cells, and download the trained model files. The notebook uses the training script in the companion [Fruits_quality_detection repository](https://github.com/Vinodvinum/Fruits_quality_detection).

## 1. Download and extract the real dataset

Download **Original Image.zip** from the Mendeley Data page above. Extract it so that each class is a separate folder, for example:

```
Original Image/
  Fresh Apple/
  Rotten Apple/
  Fresh Banana/
  Rotten Banana/
  Fresh Orange/
  Rotten Orange/
  ...
```

The training script reads folder names as class labels. Keep the original folder names and do not mix augmented copies across train/validation splits.

## 2. Train the model

Use Python 3.10 or 3.11. Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py --data_dir "path/to/Original Image" --epochs 12
```

The script uses MobileNetV2 transfer learning, image resizing to 224×224, augmentation on training batches, an 80/20 train/validation split, and a final fine-tuning stage. It saves:

- `models/fruit_quality.keras`
- `models/class_names.json`

The validation accuracy printed by the script is the result from your training run; do not claim an accuracy before training.

## 3. Deploy the trained model

The Streamlit app requires both files in the repository under `models/`. Add the generated model and labels to this repository and push them to GitHub. The model is expected to be a small MobileNetV2 fine-tuned checkpoint, but check its actual size before committing; GitHub blocks ordinary files above 100 MB. If it exceeds the limit, host the model artifact externally (for example, Hugging Face Hub) and configure the app to download it.

Then Streamlit Community Cloud will redeploy from the updated main branch.

## 4. Use the app

1. Upload a JPG, JPEG, or PNG fruit image.
2. The same model returns a class prediction and confidence.
3. **Without XAI** shows only the prediction.
4. **With XAI — LIME** shows the same prediction plus a superpixel explanation over the uploaded image.
5. Review the top five class probabilities.

## Scientific limitations

- The model only knows the 16 classes represented by this dataset.
- A single uploaded photo may not show internal damage or contamination.
- LIME is a local approximation, not proof of causation.
- This is an educational computer-vision demo, not a food-safety or commercial grading system.
- Use a separate held-out test set and report per-class precision, recall, F1, and confusion matrix before making research performance claims.
