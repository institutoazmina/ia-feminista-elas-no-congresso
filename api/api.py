from flask import Flask, request, jsonify
import mlflow
import torch

import pandas as pd
import numpy as np

import functions as f

from lime.lime_text import LimeTextExplainer
import shap

import dotenv
dotenv.load_dotenv()

import os

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
MODEL_DESFAVORAVEL_NAME = os.getenv("MODEL_DESFAVORAVEL_NAME", "quiteria_desfavoravel")
MODEL_DESFAVORAVEL_VERSION = os.getenv("MODEL_DESFAVORAVEL_VERSION", "1" )

mlflow.set_tracking_uri(MLFLOW_URI)

print("Subindo API...")

pipe = mlflow.transformers.load_model(
    f"models:/{MODEL_DESFAVORAVEL_NAME}/{MODEL_DESFAVORAVEL_VERSION}",
    return_type="pipeline",
)

pipe.tokenizer.model_max_length = 512

app = Flask(__name__)

# EXPLAINERS
CLASS_NAMES = ["favoravel", "desfavoravel"]
LABELS = sorted(pipe.model.config.label2id, key=pipe.model.config.label2id.get)

LIME_EXPLAINER = LimeTextExplainer(class_names=CLASS_NAMES)


def predict_proba_lime(batch_texts):
    raw = pipe(list(batch_texts), truncation=True, max_length=512, top_k=None)
    out = np.zeros((len(raw), 2))
    for i, r in enumerate(raw):
        d = {x["label"]: x["score"] for x in r}
        out[i, 0] = d["LABEL_0"]
        out[i, 1] = d["LABEL_1"]
    return out


def f_shap(x):
    tv = torch.tensor(
        [pipe.tokenizer.encode(v, padding="max_length", max_length=512, truncation=True) for v in x]
    ).to(pipe.model.device)
    attention_mask = (tv != pipe.tokenizer.pad_token_id).type(torch.int64).to(pipe.model.device)
    with torch.no_grad():
        logits = pipe.model(tv, attention_mask=attention_mask).logits
    return logits.detach().cpu().numpy()

SHAP_EXPLAINER = shap.Explainer(f_shap, pipe.tokenizer, output_names=LABELS)



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "instances" not in data:
        return jsonify({"error": "Body precisa conter a chave 'instances'."}), 400
    
    items = data["instances"]

    if not isinstance(items, list) or len(items) == 0:
        return jsonify({"error": "'instances' deve ser uma lista não vazia."}), 400

    fields = [
        "id",
        "textInteiroTeor",
        "ementa",
        "genero",
        "partido",
        "uf",
    ]
    
    for i, item in enumerate(items):

        if not isinstance(item, dict):
            return jsonify({"error": f"Item {i} deve ser um dicionário."}), 400

        for field in fields:
            if field not in item:
                return jsonify({"error": f"Item {i} deve conter a chave '{field}'."}), 400


    df = pd.DataFrame(items)
    df = f.df_transform(df)

    raw = pipe(
        df['text'].tolist(),
        truncation=True,
        max_length=512,
        batch_size=8,
        top_k=None,            # para devolver score de TODAS as classes (0,1)
    )
       
    df["proba_desfavoravel"] = [f.to_proba(r) for r in raw]
    df["pred"] = (df["proba_desfavoravel"] >= 0.5).astype(int)
    
    if data.get("lime", False):
        df["lime_explanation"] = df["text"].apply(lambda x: LIME_EXPLAINER.explain_instance(x, predict_proba_lime, num_features=15, num_samples=500).as_html())
    
    if data.get("shap", False):
        df["shap_explanation"] = df["text"].apply(lambda x: shap.plots.text(SHAP_EXPLAINER([x]), display=False))

    predictions = df.to_dict(orient="records")
    return jsonify({"predictions": predictions}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)