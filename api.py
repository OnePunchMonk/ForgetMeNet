import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import logging
import time
import json
import os
import shutil
import requests
import gc
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Verifiable Transformer Unlearning System API",
    description="API for sentiment analysis, SISA-based Transformer unlearning, and accuracy benchmarking.",
)

class TrainRequest(BaseModel):
    data: list[dict]

class UnlearnRequest(BaseModel):
    id: str
    text: str

BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SHARD_MODEL_DIR = "./sharded_transformer_models"
IPFS_API_URL = "http://127.0.0.1:5001/api/v0"

class SISA_Transformer_Model:
    def __init__(self, n_shards=3):
        self.n_shards = n_shards
        self.model_dir = SHARD_MODEL_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.benchmark_dataset = None
        self.shards_data = {i: [] for i in range(self.n_shards)}
        self.shards_models = {i: None for i in range(self.n_shards)}
        os.makedirs(self.model_dir, exist_ok=True)
        self.load_trained_shards()

    def _get_shard_index(self, record_id: str) -> int:
        return hash(record_id) % self.n_shards

    def load_trained_shards(self):
        for i in range(self.n_shards):
            shard_path = os.path.join(self.model_dir, f"shard_{i}")
            if os.path.exists(shard_path):
                try:
                    self.shards_models[i] = pipeline("sentiment-analysis", model=shard_path, device=0 if torch.cuda.is_available() else -1)
                    logger.info(f"Loaded model for shard {i}")
                except Exception as e:
                    logger.error(f"Failed to load model for shard {i}: {e}")

    def _train_one_shard(self, shard_idx: int, shard_dataset: Dataset):
        # FIX: Explicitly release the old model to prevent file lock errors on Windows.
        if self.shards_models.get(shard_idx) is not None:
            logger.info(f"Releasing model for shard {shard_idx} before retraining.")
            self.shards_models[shard_idx] = None
            gc.collect()

        logger.info(f"Starting fine-tuning for shard {shard_idx}...")
        shard_path = os.path.join(self.model_dir, f"shard_{shard_idx}")
        if os.path.exists(shard_path): shutil.rmtree(shard_path)

        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
        
        def tokenize(batch):
            return self.tokenizer(batch["text"], padding="max_length", truncation=True)

        tokenized_dataset = shard_dataset.map(tokenize, batched=True)
        training_args = TrainingArguments(output_dir=shard_path, num_train_epochs=1, per_device_train_batch_size=4, report_to="none")
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
        trainer.train()
        
        self.shards_models[shard_idx] = pipeline("sentiment-analysis", model=trainer.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
        trainer.save_model(shard_path)
        logger.info(f"Finished fine-tuning for shard {shard_idx}.")

    def train_from_dataframe(self, df: pd.DataFrame):
        logger.info("Training with ground truth, creating benchmark set.")
        df['label'] = df['ground_truth'].apply(lambda x: 1 if x == 'POSITIVE' else 0)
        
        train_df, benchmark_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df['label'])
        self.benchmark_dataset = benchmark_df.copy()
        logger.info(f"Training set: {len(train_df)}, Benchmark set: {len(self.benchmark_dataset)}")

        for i in range(self.n_shards):
            shard_df = train_df[train_df['account_number'].apply(lambda x: self._get_shard_index(str(x)) == i)]
            if not shard_df.empty:
                self.shards_data[i] = shard_df.to_dict('records')
                shard_dataset = Dataset.from_pandas(shard_df[['feedback_text', 'label']].rename(columns={'feedback_text': 'text'}))
                self._train_one_shard(i, shard_dataset)
        logger.info("All shards have been trained.")

    def benchmark_accuracy(self):
        if self.benchmark_dataset is None:
            return {"message": "Benchmark dataset not available."}
        
        true_labels = self.benchmark_dataset['label'].tolist()
        predictions = []
        for _, row in self.benchmark_dataset.iterrows():
            shard_idx = self._get_shard_index(row['account_number'])
            shard_model = self.shards_models.get(shard_idx)
            if shard_model:
                pred = shard_model(row['feedback_text'])[0]
                pred_label = 1 if pred['label'] == 'POSITIVE' else 0
                predictions.append(pred_label)
            else:
                predictions.append(-1)

        accuracy = accuracy_score(true_labels, predictions)
        logger.info(f"Benchmark accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}

    def unlearn_and_retrain(self, record_id: str):
        start_time = time.time()
        shard_idx = self._get_shard_index(record_id)
        
        original_data = self.shards_data.get(shard_idx, [])
        data_to_keep = [d for d in original_data if d.get('account_number') != record_id]
        
        if len(data_to_keep) == len(original_data):
            return 0.0

        self.shards_data[shard_idx] = data_to_keep
        
        if not data_to_keep:
            shard_path = os.path.join(self.model_dir, f"shard_{shard_idx}")
            if os.path.exists(shard_path): shutil.rmtree(shard_path)
            self.shards_models[shard_idx] = None
        else:
            retrain_df = pd.DataFrame(data_to_keep)
            retrain_dataset = Dataset.from_pandas(retrain_df[['feedback_text', 'label']].rename(columns={'feedback_text': 'text'}))
            self._train_one_shard(shard_idx, retrain_dataset)
        
        duration = time.time() - start_time
        logger.info(f"Shard {shard_idx} retrained in {duration:.2f}s for record {record_id}.")
        return duration

sisa_model = SISA_Transformer_Model()

@app.post("/train")
def train_model_endpoint(request: TrainRequest, background_tasks: BackgroundTasks):
    df = pd.DataFrame(request.data)
    if 'ground_truth' not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'ground_truth' column in data.")
    background_tasks.add_task(sisa_model.train_from_dataframe, df)
    return {"message": "Model training & benchmark creation initiated."}

@app.get("/benchmark")
def get_benchmark_endpoint():
    return sisa_model.benchmark_accuracy()

@app.post("/unlearn_subset")
def unlearn_subset_endpoint(request: list[UnlearnRequest]):
    certificates = []
    total_duration = 0
    unlearned_count = 0

    for item in request:
        duration = sisa_model.unlearn_and_retrain(item.id)
        if duration > 0:
            total_duration += duration
            unlearned_count += 1
            
            # Create certificate data
            cert_data = {
                "certification_timestamp": datetime.now().isoformat(),
                "certified_unlearned_id": item.id,
                "unlearning_duration_seconds": round(duration, 4),
                "model_state": f"Retrained Transformer shard {sisa_model._get_shard_index(item.id)}."
            }
            
            # Upload to IPFS
            try:
                files = {'file': ('certification.json', json.dumps(cert_data))}
                response = requests.post(f"{IPFS_API_URL}/add", files=files, timeout=20)
                response.raise_for_status()
                cid = response.json().get('Hash')
                cert_data['ipfs_cid'] = cid
            except Exception as e:
                logger.error(f"IPFS upload failed for {item.id}: {e}")
                cert_data['ipfs_cid'] = "UPLOAD_FAILED"
            
            certificates.append(cert_data)

    avg_time = (total_duration / unlearned_count) if unlearned_count > 0 else 0
    
    return {
        "certificates": certificates,
        "average_unlearning_time": round(avg_time, 4),
        "total_records_unlearned": unlearned_count
    }