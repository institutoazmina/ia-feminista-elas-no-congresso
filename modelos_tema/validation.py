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
model_name = "quiteria_tema"

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

model_run_stats_df = pd.Series(model_run_stats).reset_index()
model_run_stats_df['index'] = model_run_stats_df['index'].apply(lambda x: x.replace("f1_class_", "F1 ").replace("_", " "))
model_run_stats_df.to_csv(f"validation_last_model_stats_{model_name}.csv", sep=";", index=False)

# %%


exp_name = "QUITERIA_tema_neuralmind_bert_base_portuguese_cased"

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
    'metrics.f1_macro',
    'metrics.precision_macro',
    'metrics.recall_macro',
    'metrics.f1_class_Direitos_Sexuais_e_Reprodutivos',
    'metrics.f1_class_Educação_e_Cultura',
    'metrics.f1_class_Família_Parentalidade_e_Relações_Civis',
    'metrics.f1_class_Igualdade_e_Antidiscriminação',
    'metrics.f1_class_Infância_e_Adolescência',
    'metrics.f1_class_LGBTQIAPN',
    'metrics.f1_class_Participação_Política_e_Institucionalidade',
    'metrics.f1_class_Saúde',
    'metrics.f1_class_Trabalho_Economia_Cuidado_e_Proteção_Social',
    'metrics.f1_class_Violências_de_Gênero',
]
df_runs = df_runs[cols].dropna()

#%%
rename_cols = {i:i.split(".")[1] for i in cols if i.startswith("metrics.")}
df_runs = df_runs.rename(columns=rename_cols)

stats = df_runs.describe().T
stats['range'] = ((stats['max'] - stats['min'])*100).round(2)

cols_plot = {
    'accuracy': 'Acurácia',
    'f1_macro': 'F1 Macro',
    'precision_macro': 'Precisão Macro',
    'recall_macro': 'Recall Macro',
    'f1_class_Direitos_Sexuais_e_Reprodutivos': 'F1 (Direitos Sexuais e Reprodutivos)',
    'f1_class_Educação_e_Cultura': 'F1 (Educação e Cultura)',
    'f1_class_Família_Parentalidade_e_Relações_Civis': 'F1 (Família, Parentalidade e Relações Civis)',
    'f1_class_Igualdade_e_Antidiscriminação': 'F1 (Igualdade e Antidiscriminação)',
    'f1_class_Infância_e_Adolescência': 'F1 (Infância e Adolescência)',
    'f1_class_LGBTQIAPN': 'F1 (LGBTQIAPN)',
    'f1_class_Participação_Política_e_Institucionalidade': 'F1 (Participação Política e Institucionalidade)',
    'f1_class_Saúde': 'F1 (Saúde)',
    'f1_class_Trabalho_Economia_Cuidado_e_Proteção_Social': 'F1 (Trabalho, Economia, Cuidado e Proteção Social)',
    'f1_class_Violências_de_Gênero': 'F1 (Violências de Gênero)',
}



fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 15), dpi=500)
axes = axes.flatten() # Achata a matriz 2x4 para uma lista simples de 8 posições

for i, col in enumerate(cols_plot.keys()):
    if i < 15: # Garante que não ultrapasse o limite do grid
        sns.histplot(df_runs[col], ax=axes[i], kde=True, color='royalblue')
        axes[i].set_title(f'Distribuição de {cols_plot[col]}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Frequência')
        axes[i].axvline(model_run_stats[col], color='green', linestyle='--', label=f'Último Modelo Registrado ({model_run_stats[col]:.2f})')
        axes[i].legend()

for j in range(i + 1, 15):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(f"validation_plots_{exp_name}.png")

stats.to_markdown(open(f"validation_stats_{exp_name}.md", "w"), tablefmt="pipe")
