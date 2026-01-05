import argparse
from os import environ
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
import mlflow
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base


parser = argparse.ArgumentParser(
    description="Service of MLflow-logged prediction model with logging"
)
parser.add_argument("model", help="Model name in MLflow")
# TODO support using other version than latest
# parser.add_argument("version", help="Model version (e.g., '1', 'latest')")
args = parser.parse_args()

MLFLOW_TRACKING_URI = environ["MLFLOW_TRACKING_URI"]
DATABASE_URL = environ["DATABASE_URL"]

MODEL_NAME = args.model
# MODEL_VERSION = args.version
MODEL_VERSION = "latest"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = MODEL_NAME
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(256), nullable=False)
    model_version = Column(String(64), nullable=False)
    input_text = Column(Text, nullable=False)
    output_label = Column(String(256))
    output_score = Column(Float)
    created_at = Column(DateTime())


class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)

    engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.state.async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    yield

    await engine.dispose()


api = FastAPI(lifespan=lifespan)

@api.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION}

@api.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    pred_df = api.state.model.predict([request.text])
    assert isinstance(pred_df, pd.DataFrame)

    [[label, score]] = pred_df[["label", "score"]].values

    async with api.state.async_session() as session:
        session.add(
            PredictionLog(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                input_text=request.text,
                output_label=label,
                output_score=score,
                created_at=datetime.now(),
            )
        )
        await session.commit()

    return PredictionResponse(label=label, score=score)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8001)