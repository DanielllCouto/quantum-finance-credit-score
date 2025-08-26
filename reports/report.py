import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://dagshub.com/estrellacouto05/quantum-finance-credit-score.mlflow")

client = MlflowClient()
model_name = "credit-score-model"

registered_versions = sorted(
    client.search_model_versions(f"name='{model_name}'"),
    key=lambda v: int(v.version),
    reverse=True
)

if not registered_versions:
    raise ValueError(f"No registered versions found for model '{model_name}'")

prod_version = registered_versions[0]
prod_metrics = client.get_run(prod_version.run_id).data.metrics

all_runs = mlflow.search_runs(search_all_experiments=True, 
                              order_by=["start_time DESC"], 
                              filter_string="metrics.Recall_class_0_Poor > 0",
                              max_results=5)

if all_runs.empty:
    raise ValueError("No experimental runs found.")

latest_exp_run_id = all_runs.iloc[0]["run_id"]
latest_exp_run = client.get_run(latest_exp_run_id)
exp_metrics = latest_exp_run.data.metrics

def format_metrics(metrics: dict):
    return "\n".join([f"- `{k}`: {v:.4f}" for k, v in metrics.items()])


report = f"""
## 📊 Relatório MLflow: `{model_name}`

---

### 🏁 Modelo de Produção (Última versão registrada)
- **ID da Execução**: `{prod_version.run_id}`
- **Versão do Modelo**: `{prod_version.version}`

#### 🔢 Métricas
{format_metrics(prod_metrics)}

---

### 🧪 Última Execução Experimental
- **ID da Execução**: `{latest_exp_run_id}`

#### 🔢 Métricas
{format_metrics(exp_metrics)}

---

### 📈 Comparação das Métricas
"""

for metric in prod_metrics:
    if metric in exp_metrics:
        delta = exp_metrics[metric] - prod_metrics[metric]
        report += f"- `{metric}`: Experimental = {exp_metrics[metric]:.4f}, Produção = {prod_metrics[metric]:.4f}, Δ = {delta:+.4f}\n"

out_path = "mlflow_report.md"
with open(out_path, "w", encoding="utf-8", newline="") as f:
    f.write(report)


print("MLflow report comparing latest experiment with production model generated.")
