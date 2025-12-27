from typing import TypedDict

import torch
import numpy as np
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer import Trainer, TrainOutput
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import roc_auc_score

from models.simple_dataset import SimpleDataset


class Doc(TypedDict):
    """Data sample."""
    text: str
    label: str


def tokenizer() -> BertTokenizer:
    """Tokenizer for BERT-based swedish classifier."""
    return BertTokenizer.from_pretrained("KB/bert-base-swedish-cased")


def model() -> BertForSequenceClassification:
    """BERT-based swedish classifier for fine-tuning."""
    return BertForSequenceClassification.from_pretrained(
        "KB/bert-base-swedish-cased", num_labels=2
    )


def lora_model(lora_r: int = 4) -> PeftModel:
    """BERT-based swedish classifier configured for LoRA fine-tuning."""
    return get_peft_model(
        model(),
        LoraConfig(
            "SEQ_CLS",
            target_modules=["query", "value"],
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.2,
            modules_to_save=["classifier"],
        )
    )


def compute_metrics(
    preds: tuple[np.ndarray, np.ndarray], prefix: str = "validation"
) -> dict[str, float]:
    """Compute accuracy and one-class ROC-AUC."""
    logits, labels = preds
    predictions = np.argmax(logits, axis=-1)
    return {
        f"{prefix}_accuracy": np.mean(predictions == labels).item(),
        f"{prefix}_roc_auc": roc_auc_score(labels, logits[:, 1]),
    }


def trainer(
    model: BertForSequenceClassification | PeftModel,
    train_dataset: SimpleDataset,
    val_dataset: SimpleDataset,
    learn_rate: float = 1e-5,
    max_epochs: int = 50,
    report_to: list[str] = [],
) -> Trainer:
    """Trainer configured for fine-tuning of BERT for classification."""
    return Trainer(
        model,
        TrainingArguments(
            "/tmp/tmp-model-training",
            num_train_epochs=max_epochs,
            learning_rate=learn_rate,
            per_device_train_batch_size=8,
            report_to=report_to,
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            dataloader_pin_memory=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(2)],  # stop after 2 epochs with no improv
    )


def train(
    tokenizer: BertTokenizer,
    model: PeftModel,
    train_docs: list[Doc],
    val_docs: list[Doc],
    test_docs: list[Doc],
    learn_rate: float = 1e-5,
    max_epochs: int = 50,
    report_to: list[str] = [],
) -> tuple[Trainer, TrainOutput, dict[str, float]]:
    """Training routine configured for fine-tuning of LoRA-BERT for classification."""
    label2idx = {src: idx for idx, src in enumerate(set(d["label"] for d in train_docs))}

    train_labels, val_labels, test_labels = (
        [label2idx[d["label"]] for d in docs] for docs in [train_docs, val_docs, test_docs]
    )
    train_tokens, val_tokens, test_tokens = (
        # study truncation, max_length not needed if anticipating texts longer than 512 tokens
        tokenizer([s["text"] for s in docs], padding="longest", return_tensors="pt")
        for docs in [train_docs, val_docs, test_docs]
    )
    train_dataset, val_dataset, test_dataset = (
        SimpleDataset(tokens["input_ids"], tokens["attention_mask"], torch.tensor(labels))
        for tokens, labels in [
            (train_tokens, train_labels), (val_tokens, val_labels), (test_tokens, test_labels)
        ]
    )

    model.config.id2label = {idx: lbl for lbl, idx in label2idx.items()}
    model.config.label2id = label2idx

    trainer_ = trainer(model, train_dataset, val_dataset, learn_rate, max_epochs, report_to)
    train_output = trainer_.train()

    test_preds = trainer_.predict(test_dataset)
    test_metrics = compute_metrics((test_preds.predictions, test_preds.label_ids), prefix="test")
    print(test_metrics)

    return trainer_, train_output, test_metrics