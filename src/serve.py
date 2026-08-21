import math
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel


app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = Path(os.path.expanduser("~/models/model.joblib"))


def download_model() -> None:
    """Download the production model bundle from Cloud Storage."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(ARTIFACT_BUCKET)
    bucket.blob(MODEL_KEY).download_to_filename(str(MODEL_PATH))
    print("Model downloaded from Cloud Storage.")


def load_model_bundle(path: Path):
    """Load the threshold-aware bundle, with support for the legacy model file."""
    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"], float(loaded.get("decision_threshold", 0.5))
    return loaded, 0.5


download_model()
model, decision_threshold = load_model_bundle(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    if len(req.features) != 10:
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")
    if not all(math.isfinite(value) for value in req.features):
        raise HTTPException(status_code=400, detail="Features must be finite numbers")

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba([req.features])[0, 1])
        prediction = int(probability >= decision_threshold)
    else:
        prediction = int(model.predict([req.features])[0])

    label = "thu_nhap_cao" if prediction == 1 else "thu_nhap_thap"
    return {"prediction": prediction, "label": label}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
