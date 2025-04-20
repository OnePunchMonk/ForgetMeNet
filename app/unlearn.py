# app/unlearn.py
import pandas as pd
from app.model import SentimentModel

def forget_sample(dataset_path: str, forget_hash: str):
    df = pd.read_csv(dataset_path)
    df["text_hash"] = df["text"].apply(lambda t: hashlib.sha256(t.encode()).hexdigest())
    filtered_df = df[df["text_hash"] != forget_hash]
    
    model = SentimentModel(model_dir="model_unlearned")
    model.train(filtered_df)
    return model
