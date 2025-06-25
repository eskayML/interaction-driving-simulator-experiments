import io
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore")


def plot_word_count_histogram(csv_path):
    """Plot a histogram of word counts per 5-second interval and return as image."""
    df = pd.read_csv(csv_path)
    max_time = df["end_time"].max()
    bucket_size = 5
    buckets = np.arange(0, max_time + bucket_size, bucket_size)
    word_counts = {bucket: 0 for bucket in buckets[:-1]}
    for _, row in df.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]
        text = str(row["text"])
        words = text.split()
        num_words = len(words)
        if num_words == 0:
            continue
        duration = end_time - start_time
        time_per_word = duration / num_words
        for i in range(num_words):
            word_start_time = start_time + i * time_per_word
            bucket_index = int(word_start_time // bucket_size)
            bucket_start = bucket_index * bucket_size
            if bucket_start in word_counts:
                word_counts[bucket_start] += 1
    wc_df = pd.DataFrame(
        {"Time": list(word_counts.keys()), "Word_Count": list(word_counts.values())}
    )
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=wc_df,
        x="Time",
        weights="Word_Count",
        bins=buckets,
        color="teal",
        edgecolor="black",
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Word Count")
    plt.title("Word Count per 5-second Interval")
    plt.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def plot_sentiment_over_time(csv_path):
    """Plot sentiment scores over time and return as image."""
    df = pd.read_csv(csv_path)
    df["Sentiment_Category"] = pd.cut(
        df["sentiment_score"],
        bins=[-1, -0.33, 0.33, 1],
        labels=["Negative", "Neutral", "Positive"],
        include_lowest=True,
    )
    
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="start_time",
        y="sentiment_score",
        hue="Sentiment_Category",
        palette={"Negative": "red", "Neutral": "gray", "Positive": "blue"},
        data=df,
        s=100,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    plt.axhline(y=-0.33, color="gray", linestyle="--", alpha=0.2)
    plt.axhline(y=0.33, color="gray", linestyle="--", alpha=0.2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Sentiment Score (-1 to +1)")
    plt.title("Sentiment Over Time")
    plt.legend(title="Sentiment Category")
    plt.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def plot_tone_intensity(csv_path):
    """Plot tone intensity over time and return as image."""
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x="start_time",
        y="tone_intensity",
        data=df,
        marker="o",
        color="skyblue",
        linewidth=2.5,
        markersize=8,
        label="Tone Intensity",
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Tone Intensity (0 to 1)")
    plt.title("Tone Intensity Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf)
