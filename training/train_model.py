"""
Fine-tune ConvNeXt-Tiny on the public Project-AgML fruit freshness dataset.

Run this on Google Colab or Kaggle with a GPU:
    pip install -r training/requirements.txt
    python training/train_model.py --output_dir fruit-convnext-custom

The Streamlit app can load the exported model by setting FRUIT_MODEL_ID to
a Hugging Face model repository containing the saved model and processor files.
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

DATASET_ID = "Project-AgML/fresh_rotten_fruit_classification"
BASE_MODEL_ID = "facebook/convnext-tiny-224"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="fruit-convnext-custom")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    # The raw split contains original labelled images, not the augmented copy.
    dataset = load_dataset(DATASET_ID, "raw", split="train")
    label_feature = dataset.features["label"]
    fruit_types = sorted(set(dataset["fruit_type"]))
    class_names = [
        f"{state}_{fruit}"
        for state in ("fresh", "rotten")
        for fruit in fruit_types
    ]
    label_to_id = {name: index for index, name in enumerate(class_names)}
    id_to_label = {index: name for name, index in label_to_id.items()}

    def add_combined_class(example):
        quality = label_feature.int2str(example["label"])
        combined = f"{quality}_{example['fruit_type']}"
        if combined not in label_to_id:
            raise ValueError(f"Unexpected dataset label: {combined}")
        return {"class_id": label_to_id[combined]}

    dataset = dataset.map(add_combined_class)
    dataset = dataset.cast_column(
        "class_id", ClassLabel(names=class_names)
    )
    splits = dataset.train_test_split(
        test_size=0.15,
        seed=42,
        stratify_by_column="class_id",
    )

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL_ID)

    def preprocess(batch):
        rgb_images = [image.convert("RGB") for image in batch["image"]]
        pixel_values = processor(
            images=rgb_images,
            return_tensors="np",
        )["pixel_values"]
        return {
            "pixel_values": [value for value in pixel_values],
            "labels": batch["class_id"],
        }

    train_data = splits["train"].map(
        preprocess,
        batched=True,
        batch_size=24,
        remove_columns=splits["train"].column_names,
    )
    eval_data = splits["test"].map(
        preprocess,
        batched=True,
        batch_size=24,
        remove_columns=splits["test"].column_names,
    )
    train_data.set_format("torch")
    eval_data.set_format("torch")

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=len(class_names),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )

    def collate(batch):
        return {
            "pixel_values": torch.stack(
                [item["pixel_values"] for item in batch]
            ),
            "labels": torch.tensor(
                [item["labels"] for item in batch],
                dtype=torch.long,
            ),
        }

    def metrics(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(eval_prediction.label_ids, predicted),
            "macro_f1": f1_score(
                eval_prediction.label_ids,
                predicted,
                average="macro",
                zero_division=0,
            ),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collate,
        compute_metrics=metrics,
    )
    trainer.train()
    final_metrics = trainer.evaluate()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    with open(
        os.path.join(args.output_dir, "class_mapping.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {"id2label": id_to_label, "label2id": label_to_id},
            file,
            indent=2,
        )
    with open(
        os.path.join(args.output_dir, "evaluation_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_metrics, file, indent=2)

    print("Training complete.")
    print(f"Saved model and image processor to: {args.output_dir}")
    print(json.dumps(final_metrics, indent=2))
    print(
        "Before using a custom model in the hosted app, upload this folder "
        "to a Hugging Face model repository and set FRUIT_MODEL_ID."
    )


if __name__ == "__main__":
    main()
