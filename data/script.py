from datasets import load_dataset

dataset = load_dataset("imdb")
df = dataset["train"].to_pandas().sample(10000)
df.to_csv("imdb_subset.csv", index=False)