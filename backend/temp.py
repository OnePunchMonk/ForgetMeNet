# import ipfshttpclient
# import torch
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from transformers import pipeline
# from sklearn.feature_extraction.text import HashingVectorizer
# from sklearn.linear_model import SGDClassifier
# import numpy as np
# import logging
# from typing import Dict, Any

# # --- Setup ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="Unlearning Certification System API",
#     description="API for sentiment analysis and SISA-based unlearning.",
# )

# # --- Model Definitions ---
# class UnlearnRequest(BaseModel):
#     id: str  # Corresponds to account_number
#     text: str # Corresponds to feedback_text
#     full_record: Dict[str, Any] # The full original row from the CSV

# class UnlearnResponse(BaseModel):
#     id: str
#     status: str
#     sentiment: dict
#     cid: str | None = None
#     message: str

# # --- Global Components ---
# try:
#     # Initialize sentiment analysis pipeline from Hugging Face
#     sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
#     logger.info("Sentiment analysis model loaded successfully.")
# except Exception as e:
#     logger.error(f"Failed to load sentiment model: {e}")
#     sentiment_pipeline = None

# try:
#     # Connect to the local IPFS daemon
#     ipfs_client = ipfshttpclient.connect()
#     logger.info("Successfully connected to IPFS daemon.")
# except Exception as e:
#     logger.error(f"Could not connect to IPFS daemon. Please ensure it's running. Error: {e}")
#     ipfs_client = None

# # --- SISA (Sharded, Isolated, Sliced, and Aggregated) Unlearning Model ---
# class SISAModel:
#     """
#     A simplified implementation of a sharded model for efficient unlearning.
#     """
#     def __init__(self, n_shards=10):
#         self.n_shards = n_shards
#         self.vectorizer = HashingVectorizer(n_features=2**10)
#         self._initialize_shards()
#         logger.info(f"SISA model initialized with {n_shards} shards.")

#     def _initialize_shards(self):
#         """Initializes or re-initializes all model shards and data stores."""
#         self.shards_data = {i: [] for i in range(self.n_shards)}
#         self.shards_models = {i: SGDClassifier(loss='log_loss') for i in range(self.n_shards)}
#         self.is_fitted = {i: False for i in range(self.n_shards)}

#     def _get_shard_index(self, text_id: str) -> int:
#         """Determines the shard for a given piece of data."""
#         return hash(text_id) % self.n_shards

#     def add_data(self, text_id: str, text: str, label: int):
#         """Adds data to the appropriate shard and trains that shard."""
#         shard_idx = self._get_shard_index(text_id)
#         # Avoid adding duplicate data
#         if any(d['id'] == text_id for d in self.shards_data[shard_idx]):
#             return
            
#         self.shards_data[shard_idx].append({'id': text_id, 'text': text, 'label': label})
        
#         # Incremental training on the specific shard
#         X_shard = self.vectorizer.transform([d['text'] for d in self.shards_data[shard_idx]])
#         y_shard = np.array([d['label'] for d in self.shards_data[shard_idx]])
        
#         # We need at least two classes to train.
#         if len(np.unique(y_shard)) > 1:
#             self.shards_models[shard_idx].partial_fit(X_shard, y_shard, classes=np.array([0, 1]))
#             self.is_fitted[shard_idx] = True

#     def unlearn_and_retrain(self, text_id: str):
#         """
#         Performs the unlearning operation by retraining a single shard.
#         """
#         shard_idx = self._get_shard_index(text_id)
#         logger.info(f"Unlearning request for ID '{text_id}' in shard {shard_idx}.")

#         # Find and remove the data point
#         data_to_keep = [d for d in self.shards_data[shard_idx] if d['id'] != text_id]

#         if len(data_to_keep) == len(self.shards_data[shard_idx]):
#             logger.warning(f"Data with ID '{text_id}' not found in shard {shard_idx}. Cannot unlearn.")
#             return

#         # Retrain the shard from scratch with the remaining data
#         self.shards_data[shard_idx] = data_to_keep
#         self.shards_models[shard_idx] = SGDClassifier(loss='log_loss') # Re-initialize model
        
#         if not self.shards_data[shard_idx]:
#             self.is_fitted[shard_idx] = False
#             logger.info(f"Shard {shard_idx} is now empty after unlearning.")
#             return

#         X_shard = self.vectorizer.transform([d['text'] for d in self.shards_data[shard_idx]])
#         y_shard = np.array([d['label'] for d in self.shards_data[shard_idx]])

#         if len(np.unique(y_shard)) > 1:
#             self.shards_models[shard_idx].fit(X_shard, y_shard)
#             self.is_fitted[shard_idx] = True
#             logger.info(f"Shard {shard_idx} retrained successfully after unlearning.")
#         else:
#             self.is_fitted[shard_idx] = False # Not enough class diversity to train
#             logger.info(f"Shard {shard_idx} has only one class after unlearning, model is not fitted.")


# # Initialize the global SISA model instance
# sisa_model = SISAModel()

# # --- API Endpoints ---
# @app.get("/status")
# def get_status():
#     """Checks the status of backend services like IPFS."""
#     ipfs_online = False
#     if ipfs_client:
#         try:
#             ipfs_client.id()
#             ipfs_online = True
#         except Exception:
#             ipfs_online = False
    
#     return {
#         "status": "online",
#         "services": {
#             "ipfs_daemon": "online" if ipfs_online else "offline",
#             "sentiment_model": "loaded" if sentiment_pipeline else "error"
#         }
#     }

# @app.post("/certify-unlearning", response_model=UnlearnResponse)
# async def certify_unlearning(request: UnlearnRequest):
#     """
#     Processes an unlearning request: analyzes sentiment, unlearns if necessary,
#     and uploads a certificate to IPFS.
#     """
#     if not sentiment_pipeline or not ipfs_client:
#         raise HTTPException(status_code=503, detail="A backend service is unavailable.")

#     # Step 1: Sentiment Analysis
#     try:
#         sentiment_result = sentiment_pipeline(request.text)[0]
#         # Label: 1 for POSITIVE, 0 for NEGATIVE/NEUTRAL
#         label = 1 if sentiment_result['label'] == 'POSITIVE' else 0
#         sisa_model.add_data(request.id, request.text, label)

#     except Exception as e:
#         logger.error(f"Error during sentiment analysis for ID {request.id}: {e}")
#         raise HTTPException(status_code=500, detail="Sentiment analysis failed.")

#     # Step 2: Unlearning Decision
#     # We unlearn if the sentiment is NOT positive.
#     if sentiment_result['label'] != 'POSITIVE':
#         try:
#             # Trigger the unlearning process in our SISA model
#             sisa_model.unlearn_and_retrain(request.id)

#             # Step 3: Create and Upload Certification to IPFS
#             certification_data = {
#                 "certification_timestamp": torch.cuda.is_available(),
#                 "certified_unlearned_id": request.id,
#                 "original_record": request.full_record, # Store the whole original record
#                 "reason": f"Sentiment was '{sentiment_result['label']}', which triggered unlearning.",
#                 "model_state": f"Retrained shard {sisa_model._get_shard_index(request.id)}."
#             }
            
#             # Add to IPFS
#             res = ipfs_client.add_json(certification_data)
#             logger.info(f"Certified unlearning for ID {request.id}. IPFS CID: {res}")

#             return UnlearnResponse(
#                 id=request.id,
#                 status="Certified",
#                 sentiment=sentiment_result,
#                 cid=res,
#                 message="Data unlearned and certificate stored on IPFS."
#             )
#         except Exception as e:
#             logger.error(f"Error during unlearning/IPFS upload for ID {request.id}: {e}")
#             return UnlearnResponse(
#                 id=request.id,
#                 status="Failed",
#                 sentiment=sentiment_result,
#                 message=f"Unlearning or IPFS upload failed: {e}"
#             )
#     else:
#         # If sentiment is POSITIVE, we don't unlearn
#         return UnlearnResponse(
#             id=request.id,
#             status="Not Applicable",
#             sentiment=sentiment_result,
#             message="Positive sentiment; unlearning not required."
#         )
import ipfs_kubo_client as kubo_client
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
import logging
from typing import Dict, Any

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unlearning Certification System API",
    description="API for sentiment analysis and SISA-based unlearning.",
)

# --- Model Definitions ---
class UnlearnRequest(BaseModel):
    id: str  # Corresponds to account_number
    text: str # Corresponds to feedback_text
    full_record: Dict[str, Any] # The full original row from the CSV

class UnlearnResponse(BaseModel):
    id: str
    status: str
    sentiment: dict
    cid: str | None = None
    message: str

# --- Global Components ---
try:
    # Initialize sentiment analysis pipeline from Hugging Face
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    logger.info("Sentiment analysis model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {e}")
    sentiment_pipeline = None

try:
    # Connect to the local IPFS daemon using the new kubo-client
    ipfs_client = kubo_client.Client()
    logger.info("Successfully connected to IPFS daemon.")
except Exception as e:
    logger.error(f"Could not connect to IPFS daemon. Please ensure it's running. Error: {e}")
    ipfs_client = None

# --- SISA (Sharded, Isolated, Sliced, and Aggregated) Unlearning Model ---
class SISAModel:
    """
    A simplified implementation of a sharded model for efficient unlearning.
    """
    def __init__(self, n_shards=10):
        self.n_shards = n_shards
        self.vectorizer = HashingVectorizer(n_features=2**10)
        self._initialize_shards()
        logger.info(f"SISA model initialized with {n_shards} shards.")

    def _initialize_shards(self):
        """Initializes or re-initializes all model shards and data stores."""
        self.shards_data = {i: [] for i in range(self.n_shards)}
        self.shards_models = {i: SGDClassifier(loss='log_loss') for i in range(self.n_shards)}
        self.is_fitted = {i: False for i in range(self.n_shards)}

    def _get_shard_index(self, text_id: str) -> int:
        """Determines the shard for a given piece of data."""
        return hash(text_id) % self.n_shards

    def add_data(self, text_id: str, text: str, label: int):
        """Adds data to the appropriate shard and trains that shard."""
        shard_idx = self._get_shard_index(text_id)
        # Avoid adding duplicate data
        if any(d['id'] == text_id for d in self.shards_data[shard_idx]):
            return
            
        self.shards_data[shard_idx].append({'id': text_id, 'text': text, 'label': label})
        
        # Incremental training on the specific shard
        X_shard = self.vectorizer.transform([d['text'] for d in self.shards_data[shard_idx]])
        y_shard = np.array([d['label'] for d in self.shards_data[shard_idx]])
        
        # We need at least two classes to train.
        if len(np.unique(y_shard)) > 1:
            self.shards_models[shard_idx].partial_fit(X_shard, y_shard, classes=np.array([0, 1]))
            self.is_fitted[shard_idx] = True

    def unlearn_and_retrain(self, text_id: str):
        """
        Performs the unlearning operation by retraining a single shard.
        """
        shard_idx = self._get_shard_index(text_id)
        logger.info(f"Unlearning request for ID '{text_id}' in shard {shard_idx}.")

        # Find and remove the data point
        data_to_keep = [d for d in self.shards_data[shard_idx] if d['id'] != text_id]

        if len(data_to_keep) == len(self.shards_data[shard_idx]):
            logger.warning(f"Data with ID '{text_id}' not found in shard {shard_idx}. Cannot unlearn.")
            return

        # Retrain the shard from scratch with the remaining data
        self.shards_data[shard_idx] = data_to_keep
        self.shards_models[shard_idx] = SGDClassifier(loss='log_loss') # Re-initialize model
        
        if not self.shards_data[shard_idx]:
            self.is_fitted[shard_idx] = False
            logger.info(f"Shard {shard_idx} is now empty after unlearning.")
            return

        X_shard = self.vectorizer.transform([d['text'] for d in self.shards_data[shard_idx]])
        y_shard = np.array([d['label'] for d in self.shards_data[shard_idx]])

        if len(np.unique(y_shard)) > 1:
            self.shards_models[shard_idx].fit(X_shard, y_shard)
            self.is_fitted[shard_idx] = True
            logger.info(f"Shard {shard_idx} retrained successfully after unlearning.")
        else:
            self.is_fitted[shard_idx] = False # Not enough class diversity to train
            logger.info(f"Shard {shard_idx} has only one class after unlearning, model is not fitted.")


# Initialize the global SISA model instance
sisa_model = SISAModel()

# --- API Endpoints ---
@app.get("/status")
def get_status():
    """Checks the status of backend services like IPFS."""
    ipfs_online = False
    if ipfs_client:
        try:
            ipfs_client.id()
            ipfs_online = True
        except Exception:
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
    """
    Processes an unlearning request: analyzes sentiment, unlearns if necessary,
    and uploads a certificate to IPFS.
    """
    if not sentiment_pipeline or not ipfs_client:
        raise HTTPException(status_code=503, detail="A backend service is unavailable.")

    # Step 1: Sentiment Analysis
    try:
        sentiment_result = sentiment_pipeline(request.text)[0]
        # Label: 1 for POSITIVE, 0 for NEGATIVE/NEUTRAL
        label = 1 if sentiment_result['label'] == 'POSITIVE' else 0
        sisa_model.add_data(request.id, request.text, label)

    except Exception as e:
        logger.error(f"Error during sentiment analysis for ID {request.id}: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed.")

    # Step 2: Unlearning Decision
    # We unlearn if the sentiment is NOT positive.
    if sentiment_result['label'] != 'POSITIVE':
        try:
            # Trigger the unlearning process in our SISA model
            sisa_model.unlearn_and_retrain(request.id)

            # Step 3: Create and Upload Certification to IPFS
            certification_data = {
                "certification_timestamp": torch.cuda.is_available(),
                "certified_unlearned_id": request.id,
                "original_record": request.full_record, # Store the whole original record
                "reason": f"Sentiment was '{sentiment_result['label']}', which triggered unlearning.",
                "model_state": f"Retrained shard {sisa_model._get_shard_index(request.id)}."
            }
            
            # Add to IPFS
            res = ipfs_client.add_json(certification_data)
            logger.info(f"Certified unlearning for ID {request.id}. IPFS CID: {res}")

            return UnlearnResponse(
                id=request.id,
                status="Certified",
                sentiment=sentiment_result,
                cid=res,
                message="Data unlearned and certificate stored on IPFS."
            )
        except Exception as e:
            logger.error(f"Error during unlearning/IPFS upload for ID {request.id}: {e}")
            return UnlearnResponse(
                id=request.id,
                status="Failed",
                sentiment=sentiment_result,
                message=f"Unlearning or IPFS upload failed: {e}"
            )
    else:
        # If sentiment is POSITIVE, we don't unlearn
        return UnlearnResponse(
            id=request.id,
            status="Not Applicable",
            sentiment=sentiment_result,
            message="Positive sentiment; unlearning not required."
        )
