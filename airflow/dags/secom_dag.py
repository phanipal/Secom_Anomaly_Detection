"""Daily SECOM retrain pipeline.

download_data -> clean_features -> cross_validate -> explain -> publish_report

Tasks hand data to each other through files under data/ and results/, not XCom;
only small summaries go through XCom so they show in the UI.
"""
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import secom  # noqa: E402


def download_data():
    paths = secom.download()
    return [str(p) for p in paths]


def clean_features():
    return secom.stage_clean()


def cross_validate():
    m = secom.stage_cv()
    best = m["best_model"]
    return {"best_model": best, **m[best]}


def explain():
    return secom.stage_explain()


def publish_report(ds, ti):
    metrics = json.loads((secom.OUT / "metrics.json").read_text())
    best = metrics["best_model"]
    report = {
        "run_date": ds,
        "rows": ti.xcom_pull(task_ids="clean_features")["rows"],
        "features": ti.xcom_pull(task_ids="clean_features")["features"],
        "best_model": best,
        "pr_auc": metrics[best]["pr_auc"],
        "roc_auc": metrics[best]["roc_auc"],
        "recall_at_10pct_fpr": metrics[best]["recall_at_10pct_fpr"],
        "top_sensors": list(metrics["top_10_sensors"])[:5],
    }
    out = secom.OUT / "reports"
    out.mkdir(exist_ok=True)
    (out / f"report_{ds}.json").write_text(json.dumps(report, indent=2))
    shutil.copy(secom.OUT / "metrics.json", out / f"metrics_{ds}.json")
    # ponytail: threshold is a plain constant; wire an alert operator when this runs for real
    if report["pr_auc"] < 0.10:
        raise ValueError(f"PR-AUC {report['pr_auc']} below floor 0.10, not publishing")
    return report


default_args = {
    "owner": "phanendra",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="secom_pipeline",
    description="Retrain and evaluate the SECOM wafer fail detector",
    schedule="0 2 * * *",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    default_args=default_args,
    tags=["secom", "ml"],
) as dag:
    t_download = PythonOperator(task_id="download_data", python_callable=download_data)
    t_clean = PythonOperator(task_id="clean_features", python_callable=clean_features)
    t_cv = PythonOperator(task_id="cross_validate", python_callable=cross_validate)
    t_explain = PythonOperator(task_id="explain", python_callable=explain)
    t_publish = PythonOperator(task_id="publish_report", python_callable=publish_report)

    t_download >> t_clean >> t_cv >> t_explain >> t_publish
