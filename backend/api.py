import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
import logging
from typing import Dict, Any
import requests 
import json

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unlearning Certification System API",
    description="API for sentiment analysis and SISA-based unlearning.",
)

# --- Model Definitions ---
class UnlearnRequest(BaseModel):
    id: str
    text: str
    full_record: Dict[str, Any]

class UnlearnResponse(BaseModel):
    id: str
    status: str
    sentiment: dict
    cid: str | None = None
    message: str

# --- Global Components ---
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    logger.info("Sentiment analysis model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {e}")
    sentiment_pipeline = None

# --- SISA (Sharded, Isolated, Sliced, and Aggregated) Unlearning Model ---
class SISAModel:
    def __init__(self, n_shards=10):
        self.n_shards = n_shards
        self.vectorizer = HashingVectorizer(n_features=2**10)
        self._initialize_shards()
        logger.info(f"SISA model initialized with {n_shards} shards.")

    def _initialize_shards(self):
        self.shards_data = {i: [] for i in range(self.n_shards)}
        self.shards_models = {i: SGDClassifier(loss='log_loss') for i in range(self.n_shards)}
        self.is_fitted = {i: False for i in range(self.n_shards)}

    def _get_shard_index(self, text_id: str) -> int:
        return hash(text_id) % self.n_shards

    def add_data(self, text_id: str, text: str, label: int):
        shard_idx = self._get_shard_index(text_id)
        if any(d['id'] == text_id for d in self.shards_data[shard_idx]):
            return
        self.shards_data[shard_idx].append({'id': text_id, 'text': text, 'label': label})
        X_shard = self.vectorizer.transform([d['text'] for d in self.shards_data[shard_idx]])
        y_shard = np.array([d['label'] for d in self.shards_data[shard_idx]])
        if len(np.unique(y_shard)) > 1:
            self.shards_models[shard_idx].partial_fit(X_shard, y_shard, classes=np.array([0, 1]))
            self.is_fitted[shard_idx] = True

    def unlearn_and_retrain(self, text_id: str):
        shard_idx = self._get_shard_index(text_id)
        logger.info(f"Unlearning request for ID '{text_id}' in shard {shard_idx}.")
        data_to_keep = [d for d in self.shards_data[shard_idx] if d['id'] != text_id]
        if len(data_to_keep) == len(self.shards_data[shard_idx]):
            return
        self.shards_data[shard_idx] = data_to_keep
        self.shards_models[shard_idx] = SGDClassifier(loss='log_loss')
        if not self.shards_data[shard_idx]:
            self.is_fitted[shard_idx] = False
            return
        X_shard = self.vectorizer.transform([d['text'] for d in self.shards_data[shard_idx]])
        y_shard = np.array([d['label'] for d in self.shards_data[shard_idx]])
        if len(np.unique(y_shard)) > 1:
            self.shards_models[shard_idx].fit(X_shard, y_shard)
            self.is_fitted[shard_idx] = True
        else:
            self.is_fitted[shard_idx] = False

sisa_model = SISAModel()

# --- API Endpoints ---
IPFS_API_URL = "http://127.0.0.1:5001/api/v0"

@app.get("/status")
def get_status():
    ipfs_online = False
    try:
        # Directly call the IPFS API /id endpoint using requests
        response = requests.post(f"{IPFS_API_URL}/id", timeout=2)
        if response.status_code == 200:
            ipfs_online = True
            logger.info("IPFS daemon is online.")
    except requests.exceptions.RequestException:
        logger.error("Could not connect to IPFS daemon via requests.")
        ipfs_online = False
    
    return {
        "status": "online",
        "services": {
            "ipfs_daemon": "online" if ipfs_online else "offline",
            "sentiment_model": "loaded" if sentiment_pipeline else "error"
        }
    }

@app.post("/certify-unlearning", response_model=UnlearnResponse)
async def certify_unlearning(request: UnlearnRequest):
    if not sentiment_pipeline:
        raise HTTPException(status_code=503, detail="Sentiment model is unavailable.")

    try:
        sentiment_result = sentiment_pipeline(request.text)[0]
        label = 1 if sentiment_result['label'] == 'POSITIVE' else 0
        sisa_model.add_data(request.id, request.text, label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

    if sentiment_result['label'] != 'POSITIVE':
        try:
            sisa_model.unlearn_and_retrain(request.id)
            certification_data = {
                "certification_timestamp": str(torch.cuda.is_available()), # Converted to string for JSON
                "certified_unlearned_id": request.id,
                "original_record": request.full_record,
                "reason": f"Sentiment was '{sentiment_result['label']}', which triggered unlearning.",
                "model_state": f"Retrained shard {sisa_model._get_shard_index(request.id)}."
            }
            
            # Use requests to add the certification data to IPFS
            files = {'file': ('certification.json', json.dumps(certification_data))}
            response = requests.post(f"{IPFS_API_URL}/add", files=files, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes
            
            ipfs_response = response.json()
            cid = ipfs_response.get('Hash')
            
            if not cid:
                raise Exception("Could not get CID from IPFS response.")

            logger.info(f"Certified unlearning for ID {request.id}. IPFS CID: {cid}")

            return UnlearnResponse(
                id=request.id, status="Certified", sentiment=sentiment_result, cid=cid,
                message="Data unlearned and certificate stored on IPFS."
            )
        except Exception as e:
            logger.error(f"Unlearning or IPFS upload failed: {e}")
            return UnlearnResponse(
                id=request.id, status="Failed", sentiment=sentiment_result,
                message=f"Unlearning or IPFS upload failed: {e}"
            )
    else:
        return UnlearnResponse(
            id=request.id, status="Not Applicable", sentiment=sentiment_result,
            message="Positive sentiment; unlearning not required."
        )
