import argparse
import json
from os import environ
from datetime import datetime, timedelta

import mlflow
from sqlalchemy import create_engine, text
import pandas as pd
from slack_sdk import WebClient

from models import monitoring


parser = argparse.ArgumentParser(
    description="Service detecting prediction data drift and logging to MLflow and Slack"
)
parser.add_argument("model", help="Model name in MLflow")
parser.add_argument(
    "--lookback-days",
    type=int,
    default=7,
    help="Number of days to look back for new data (default: 7)",
)
parser.add_argument(
    "--thresholds",
    type=json.loads,
    help='JSON object with drift score thresholds, e.g. \'{"fraction_out_of_vocab": 0.05}\'',
)
args = parser.parse_args()

MLFLOW_TRACKING_URI = environ["MLFLOW_TRACKING_URI"]
DATABASE_URL = environ["DATABASE_URL"]
SLACK_BOT_TOKEN = environ["SLACK_BOT_TOKEN"]
SLACK_CHANNEL = environ["SLACK_CHANNEL"]


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


if __name__ == "__main__":

    def _load_docs(split):
        with open(f"data/swedish_sentiment_{split}.jsonl") as f:
            docs = [json.loads(l)["text"] for l in f.read().split("\n") if l]
        return docs

    train_docs, val_docs = (_load_docs(s) for s in ["train", "val"])

    cutoff_date = datetime.now() - timedelta(days=args.lookback_days)
    engine = create_engine(DATABASE_URL)
    query = text(f'SELECT * FROM "{args.model}" WHERE created_at > :cutoff_date')
    new_docs_df = pd.read_sql_query(query, engine, params={"cutoff_date": cutoff_date})

    drift_scores = monitoring.data_drift_scores(
        train_docs, val_docs, new_docs_df["input_text"].to_list()
    )

    source_run_id = (
        mlflow.MlflowClient()
        .get_registered_model(args.model)
        .latest_versions[0]
        .run_id
    )
    with mlflow.start_run():
        mlflow.log_metrics(
            {f"{k}_drift_score": v for k, v in drift_scores.items()},
            run_id=source_run_id,
        )

    violations = [
        f"• *{metric}*: {drift_scores[metric]:.4f} (threshold: {threshold})"
        for metric, threshold in args.thresholds.items()
        if metric in drift_scores and drift_scores[metric] > threshold
    ]
    if violations:
        message = f":warning: *Drift alert for model: {args.model}*\n\n"
        message += (
            "The following KS statistics exceeded their thresholds:\n" + "\n".join(violations)
        )
        WebClient(token=SLACK_BOT_TOKEN).chat_postMessage(channel=SLACK_CHANNEL, text=message)
