# app/certifier.py
import hashlib
import json
import os
from app.ipfs_utils import upload_json

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def certify_unlearning(sample_text, baseline_prob, new_prob, threshold=0.25):
    delta = abs(baseline_prob - new_prob)
    log = {
        "sample_text": sample_text,
        "text_hash": compute_hash(sample_text),
        "baseline_prob": baseline_prob,
        "new_prob": new_prob,
        "delta": delta,
        "certified_forgotten": delta > threshold
    }

    log_path = f"logs/certs/{log['text_hash']}.json"
    os.makedirs("logs/certs", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    ipfs_hash = upload_json(log)
    log["ipfs_hash"] = ipfs_hash
    return log
