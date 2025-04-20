# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from app.model import SentimentModel
from app.unlearn import forget_sample
from app.certifier import compute_hash, certify_unlearning

app = FastAPI()
DATASET_PATH = "data/imdb_subset.csv"

base_model = SentimentModel()
base_model.load_model()

class ForgetRequest(BaseModel):
    text: str

@app.post("/unlearn/")
def unlearn_sample(req: ForgetRequest):
    text = req.text
    text_hash = compute_hash(text)
    
    # Get baseline prediction
    baseline_prob = base_model.predict_prob(text)

    # Retrain model without this sample
    new_model = forget_sample(DATASET_PATH, text_hash)
    new_prob = new_model.predict_prob(text)

    # Certify and log
    cert_log = certify_unlearning(text, baseline_prob, new_prob)
    return cert_log
