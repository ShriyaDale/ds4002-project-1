import math
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

def add_sentiment_column(input_file, output_file, batch_size=32):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    if 'Reviews' not in df.columns:
        raise ValueError("Error: 'Reviews' column not found in the CSV.")

    model_name = "MarieAngeA13/Sentiment-Analysis-BERT"
    print(f"Loading model: {model_name}...")

    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("Using GPU")
    else:
        print("Using CPU")

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
        truncation=True,
        max_length=512
    )

    print("Analyzing sentiments...")

    reviews = df['Reviews'].astype(str).tolist()
    n = len(reviews)
    num_batches = math.ceil(n / batch_size)

    sentiments = []

    for i in tqdm(range(num_batches), desc="Batches", unit="batch"):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_texts = reviews[start:end]
        batch_results = sentiment_pipeline(batch_texts)
        sentiments.extend([r["label"] for r in batch_results])

    if len(sentiments) != n:
        raise RuntimeError("Sentiment list length does not match number of rows.")

    df["sentiment"] = sentiments

    print(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    input_csv = "data/airlines_reviews.csv"
    output_csv = "data/airlines_reviews_with_sentiment.csv"
    add_sentiment_column(input_csv, output_csv, batch_size=32)
