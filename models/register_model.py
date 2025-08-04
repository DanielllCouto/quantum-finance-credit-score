import os
import mlflow
from mlflow.tracking import MlflowClient

# URL do tracking no Dagshub
mlflow.set_tracking_uri("https://dagshub.com/estrellacouto05/quantum-finance-credit-score.mlflow")

client = MlflowClient()
model_name = "credit-score-model"

# Busca as versões já registradas
registered_versions = sorted(
    client.search_model_versions(f"name='{model_name}'"),
    key=lambda v: int(v.version),
    reverse=True
)

# Se não houver nenhum modelo registrado ainda
if not registered_versions:
    print(f"Nenhuma versão registrada encontrada para '{model_name}'.")

# Pega o último modelo de produção (se houver)
prod_version = registered_versions[0] if registered_versions else None
prod_metrics = client.get_run(prod_version.run_id).data.metrics if prod_version else {}

# Busca últimas execuções no MLflow
all_runs = mlflow.search_runs(
    search_all_experiments=True, 
    order_by=["start_time DESC"], 
    max_results=5
)

if all_runs.empty:
    raise ValueError("Nenhuma execução encontrada no MLflow.")

# Pega o run_id mais recente
latest_exp_run_id = all_runs.iloc[0]["run_id"]
latest_exp_run = client.get_run(latest_exp_run_id)
exp_metrics = latest_exp_run.data.metrics

# Função para formatar métricas
def format_metrics(metrics: dict):
    return "\n".join([f"- {k}: {v:.4f}" for k, v in metrics.items()])

# Verifica se já foi registrado
already_registered = any(v.run_id == latest_exp_run_id for v in registered_versions)

# Registra novo modelo se ainda não estiver registrado
if not already_registered:
    model_uri = f"runs:/{latest_exp_run_id}/model"
    new_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=latest_exp_run_id
    )
    print(f"Novo modelo registrado: versão {new_version.version} (run_id={latest_exp_run_id})")
else:
    print("O modelo mais recente já está registrado.")

# Gera resumo no console
print("\n=== MLflow Report ===")
print(f"Modelo: {model_name}")

if prod_version:
    print(f"\nÚltima versão em produção: {prod_version.version}")
    print(f"Run ID produção: {prod_version.run_id}")
    print("Métricas de Produção:")
    print(format_metrics(prod_metrics))

print("\nÚltimo experimento treinado:")
print(f"Run ID: {latest_exp_run_id}")
print("Métricas do experimento:")
print(format_metrics(exp_metrics))
