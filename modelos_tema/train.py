# %%

import os
import dotenv
dotenv.load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
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

EXPERIMENT_DESFAVORAVEL_NAME = f"QUITERIA_tema_{model_name.replace('/', '_').replace('-', '_')}"
mlflow.set_experiment(experiment_name=EXPERIMENT_DESFAVORAVEL_NAME)

# %%

df_train = pd.read_parquet("../dados/train.parquet")
df_val = pd.read_parquet("../dados/validation.parquet")
df_test = pd.read_parquet("../dados/test.parquet")

df_train['tema'] = df_train['tema'].apply(lambda x: x.replace(",", "").replace(" ", "_").replace("+", ""))
df_val['tema'] = df_val['tema'].apply(lambda x: x.replace(",", "").replace(" ", "_").replace("+", ""))
df_test['tema'] = df_test['tema'].apply(lambda x: x.replace(",", "").replace(" ", "_").replace("+", ""))

themes_list = sorted(df_train['tema'].unique().tolist())
theme2id = {theme: idx for idx, theme in enumerate(themes_list)}
id2theme = {idx: theme for idx, theme in enumerate(themes_list)}
num_labels = len(themes_list)

print("Unique df_train themes:", num_labels, "->", themes_list)
print("Unique df_val:", df_val['tema'].nunique())
print("Unique df_test:", df_test['tema'].nunique())

df_train['tema_id'] = df_train['tema'].map(theme2id)
df_val['tema_id'] = df_val['tema'].map(theme2id)
df_test['tema_id'] = df_test['tema'].map(theme2id)


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
    tokens["labels"] = [theme2id[tema] for tema in examples["tema"]]
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
    preds = probabilities.argmax(axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "precision_macro": precision.compute(predictions=preds, references=labels, average="macro")["precision"],
        "recall_macro": recall.compute(predictions=preds, references=labels, average="macro")["recall"],
    }

# %%

runs = 100

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                           ignore_mismatched_sizes=True)

for i in range(runs):
    
    mlflow.start_run(run_name=f"run_{i+1}")
    

    training_args = TrainingArguments(
        output_dir=f"./results/_run_{i+1}",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        
        num_train_epochs=10,                # Aumentamos aqui...
        weight_decay=0.01,

        warmup_ratio=0.1,

        eval_strategy="steps",              # Avalia a cada X passos, não só no fim da época
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,                 # Mantém apenas o melhor checkpoint local

        load_best_model_at_end=True,        # Garante que o modelo final é o melhor 'checkpoint'
        metric_for_best_model="f1_macro",
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
    # Resolve predicting multi-class output instead of binary threshold
    test_pred_label = np.apply_along_axis(np.argmax, arr=test_pred.predictions, axis=1)

    acc_test = metrics.accuracy_score(test_true, test_pred_label)
    f1_test_macro = metrics.f1_score(test_true, test_pred_label, average="macro")
    precision_test_macro = metrics.precision_score(test_true, test_pred_label, average="macro")
    recall_test_macro = metrics.recall_score(test_true, test_pred_label, average="macro")

    metrics_dict = {
        "accuracy": acc_test,
        "f1_macro": f1_test_macro,
        "precision_macro": precision_test_macro,
        "recall_macro": recall_test_macro,
    }

    f1_per_class = metrics.f1_score(test_true, test_pred_label, average=None)
    for class_id, score in enumerate(f1_per_class):
        class_name = id2theme[class_id]
        metrics_dict[f"f1_class_{class_name}"] = score

    mlflow.log_metrics(metrics_dict)

    model_final = pipeline(
        task="text-classification",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    mlflow.transformers.log_model(model_final, "model")
    mlflow.end_run()
    