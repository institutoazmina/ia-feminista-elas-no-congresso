# %%

import os
import dotenv
dotenv.load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
)

from transformers.utils.notebook import NotebookProgressCallback

import torch

from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd
import evaluate
from sklearn import metrics

import torch.nn.functional as F

import mlflow
mlflow.set_tracking_uri(MLFLOW_URI)

model_name = "neuralmind/bert-base-portuguese-cased"

EXPERIMENT_DESFAVORAVEL_NAME = f"QUITERIA_desfavoravel_{model_name.replace('/', '_').replace('-', '_')}"
mlflow.set_experiment(experiment_name=EXPERIMENT_DESFAVORAVEL_NAME)

# %%

df_train = pd.read_parquet("../dados/train.parquet")
df_val = pd.read_parquet("../dados/validation.parquet")
df_test = pd.read_parquet("../dados/test.parquet")


# %%

def text_to_embedding(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()


dataset = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "validation": Dataset.from_pandas(df_val),
    "test": Dataset.from_pandas(df_test),
})

# %%

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%

def preprocess(examples):
    tokens = tokenizer(
        examples["textFormat"],
        truncation=True,
        max_length=512,
    )
    tokens["labels"] = examples["fl_desfavoravel"]
    return tokens

tokenized_datasets = dataset.map(preprocess, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["tema", "textFormat", "fl_desfavoravel"])
tokenized_datasets

# %%

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probabilities = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    positive_class_probs = probabilities[:, 1]
    
    preds = (positive_class_probs >= df_train['fl_desfavoravel'].mean()).astype(int)  # Use the mean of the training labels as the threshold
    
    roc_auc_val = metrics.roc_auc_score(labels, positive_class_probs)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"],
        "precision": precision.compute(predictions=preds, references=labels)["precision"],
        "recall": recall.compute(predictions=preds, references=labels)["recall"],
        "roc_auc": roc_auc_val,
    }

# %%

runs = 100

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
for i in range(runs):
    
    mlflow.start_run(run_name=f"run_{i+1}")

    training_args = TrainingArguments(
        output_dir=f"./results/_run_{i+1}",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        
        num_train_epochs=10,                # Aumentamos aqui...
        weight_decay=0.01,

        warmup_ratio=0.1,

        eval_strategy="steps",              # Avalia a cada X passos, não só no fim da época
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,                 # Mantém apenas o melhor checkpoint local

        load_best_model_at_end=True,        # Garante que o modelo final é o melhor 'checkpoint'
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        full_determinism=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
    )

    trainer.train()
    trainer.remove_callback(NotebookProgressCallback)

    test_true = tokenized_datasets["test"]['labels']

    test_pred = trainer.predict(tokenized_datasets["test"])

    logits = torch.from_numpy(test_pred.predictions)
    test_pred_proba = F.softmax(logits, dim=-1).numpy()
    
    test_pred_label = (test_pred_proba[:, 1] >= df_train['fl_desfavoravel'].mean()).astype(int)

    acc_test = metrics.accuracy_score(test_true, test_pred_label)
    auc_test = metrics.roc_auc_score(test_true, test_pred_proba[:, 1])
    f1_test_0 = metrics.f1_score(test_true, test_pred_label, pos_label=0)
    f1_test_1 = metrics.f1_score(test_true, test_pred_label, pos_label=1)
    precision_test_0 = metrics.precision_score(test_true, test_pred_label, pos_label=0)
    precision_test_1 = metrics.precision_score(test_true, test_pred_label, pos_label=1)
    recall_test_0 = metrics.recall_score(test_true, test_pred_label, pos_label=0)
    recall_test_1 = metrics.recall_score(test_true, test_pred_label, pos_label=1)

    metrics_dict = {
        "accuracy": acc_test,
        "roc_auc": auc_test,
        "f1_0": f1_test_0,
        "f1_1": f1_test_1,
        "precision_0": precision_test_0,
        "precision_1": precision_test_1,
        "recall_0": recall_test_0,
        "recall_1": recall_test_1,
    }

    mlflow.log_metrics(metrics_dict)

    model_final = pipeline(
        task="text-classification",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    mlflow.transformers.log_model(model_final, "model")
    mlflow.end_run()
    
