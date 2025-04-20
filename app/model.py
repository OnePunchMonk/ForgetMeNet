# app/model.py
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

class SentimentModel:
    def __init__(self, model_dir="model"):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model_dir = model_dir
        self.model = None

    def tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    def train(self, df):
        dataset = Dataset.from_pandas(df[["text", "label"]])
        dataset = dataset.map(self.tokenize, batched=True)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        args = TrainingArguments(
            output_dir=self.model_dir,
            evaluation_strategy="no",
            per_device_train_batch_size=8,
            num_train_epochs=2,
            logging_steps=10,
            save_steps=10,
            save_total_limit=1
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset
        )

        trainer.train()
        self.model.save_pretrained(self.model_dir)

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_dir)

    def predict_prob(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return float(probs[0][1])  # probability of positive
