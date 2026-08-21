# %%
import os
import sys
import dotenv
dotenv.load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import mlflow
mlflow.set_tracking_uri(MLFLOW_URI)

# %%

model_name = "quiteria_desfavoravel"

try:
    registered_models = mlflow.search_registered_models(filter_string=f"name='{model_name}'")
    final_model = registered_models[0]

except IndexError:
    print(f"Modelo '{model_name}' não registrado no MLflow.")
    sys.exit(1)

versions = [i.version for i in final_model.latest_versions]
versions.sort(reverse=True)

last_version = [i for i in final_model.latest_versions if i.version == versions[0]][0]

model_run_stats = mlflow.get_run(last_version.run_id).data.metrics
model_run_stats

# %%


exp_name = "QUITERIA_desfavoravel_neuralmind_bert_base_portuguese_cased"

df_runs = mlflow.search_runs(experiment_names=[exp_name], filter_string="attributes.status = 'FINISHED'")
df_runs =df_runs.dropna(subset=['metrics.accuracy'])
df_runs

# %%

cols = [
    'run_id',
    'experiment_id',
    'status',
    'artifact_uri',
    'start_time',
    'end_time',
    'metrics.accuracy',
    'metrics.roc_auc',
    'metrics.f1_0',
    'metrics.f1_1',
    'metrics.precision_0',
    'metrics.precision_1',
    'metrics.recall_0',
    'metrics.recall_1',
]

df_runs = df_runs[cols].dropna()

#%%
rename_cols = {i: i.split(".")[1] for i in cols if i.startswith("metrics.")}
df_runs = df_runs.rename(columns=rename_cols)

stats = df_runs.describe().T
stats['range'] = ((stats['max'] - stats['min'])*100).round(2)

cols_plot = {
    'accuracy': 'Acurácia',
    'roc_auc': 'Área da Curva ROC',
    'f1_0': 'F1 (Classe 0)',
    'f1_1': 'F1 (Classe 1)',
    'precision_0': 'Precisão (Classe 0)',
    'precision_1': 'Precisão (Classe 1)',
    'recall_0': 'Recall (Classe 0)',
    'recall_1': 'Recall (Classe 1)',
}

current_values = {
    'accuracy': 0.82,
    'f1_0': 0.88,
    'f1_1': 0.64,
    'precision_0': 0.91,
    'precision_1': 0.59,
    'recall_0': 0.86,
    'recall_1': 0.70,
}


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
axes = axes.flatten() # Achata a matriz 2x4 para uma lista simples de 8 posições

for i, col in enumerate(cols_plot.keys()):
    if i < 8: # Garante que não ultrapasse o limite do grid
        sns.histplot(df_runs[col], ax=axes[i], kde=True, color='royalblue')
        axes[i].set_title(f'Distribuição de {cols_plot[col]}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Frequência')
        axes[i].axvline(model_run_stats[col], color='green', linestyle='--', label=f'Último Modelo Registrado ({model_run_stats[col]:.2f})')
        try:
            axes[i].axvline(current_values[col], color='red', linestyle='--', label=f'Atual ({current_values[col]})')
        except KeyError:
            pass
        axes[i].legend()

for j in range(i + 1, 8):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(f"validation_plots_{exp_name}.png")

stats.to_markdown(open(f"validation_stats_{exp_name}.md", "w"), tablefmt="pipe")

# %%
