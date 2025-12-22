import mlflow
import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers.trainer import Trainer
from peft import LoraConfig, get_peft_model

from typing import TypedDict


class Sentence(TypedDict):
    text: str
    source: str


class SimpleDataset:
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


def train(sentences: list[Sentence]):

    tokenizer: DistilBertTokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = get_peft_model(
        AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        ),
        LoraConfig(
            "SEQ_CLS",
            target_modules=["q_lin", "v_lin"],
            r=5,
        )
    )

    src2label = {
        src: idx for idx, src in enumerate(set(d["source"] for d in sentences))
    }
    labels = [src2label[d["source"]] for d in sentences]

    texts = [s["text"] for s in sentences]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt"
    )

    dataset = SimpleDataset(
        encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels)
    )

    trainer = Trainer(
        model,
        TrainingArguments(
            "/tmp/tmp-model-training",
            num_train_epochs=3,
            learning_rate=1e-2,
            #logging_dir="./logs",
            report_to=["mlflow"],  # TODO parameterize?
            eval_strategy="epoch",
            #log every N steps/epochs?
            #logging_strategy="steps",
            #logging_steps=10,
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
    )

    train_results = trainer.train()
    train_metrics = trainer.evaluate(dataset, metric_key_prefix="train")
    return tokenizer, trainer, train_results, train_metrics


if __name__ == "__main__":
    from os import environ

    MLFLOW_TRACKING_URL = environ["MLFLOW_TRACKING_URL"]

    with open("data/raw/sentences.json") as f:
        sentences = json.load(f)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment("edvard-testar")
    with mlflow.start_run():
        tokenizer, trainer, results, metrics = train(sentences[:5])
        mlflow.log_metrics(metrics)
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            name="edvard-testar",
            task="text-classification",
        )

    print(results)
    print(metrics)
