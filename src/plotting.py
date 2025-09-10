import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_string_dtype
from PIL import Image


def plot_speaker_charts(df):
    """
    Generate four charts based on speaker diarization data using seaborn for a refined look:
    1. Horizontal bar chart of word count per speaker.
    2. Timeline chart (Gantt style) of speaker activity.
    3. Stacked bar chart of sentiment (positive, negative, neutral) per speaker.
    4. Horizontal bar chart of tone intensity per speaker.

    Parameters:
    df (pd.DataFrame): DataFrame with speaker diarization data.
    
    Returns:
    PIL.Image.Image: The generated chart as a PIL image.
    """
    df = df.copy()

    # --- Data Pre-processing ---
    if is_string_dtype(df["start_time"]):
        df["start_time"] = pd.to_numeric(df["start_time"].str.replace("s", ""), errors="coerce")
    else:
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")

    if is_string_dtype(df["end_time"]):
        df["end_time"] = pd.to_numeric(df["end_time"].str.replace("s", ""), errors="coerce")
    else:
        df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")

    df["word_count"] = df["transcribed_content"].apply(lambda x: len(str(x).split()))
    word_counts = df.groupby("speaker_id")["word_count"].sum().reset_index()

    def categorize_sentiment(score):
        if score > 0.1:
            return "Positive"
        elif score < -0.1:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment_category"] = df["sentiment_score"].apply(categorize_sentiment)
    sentiment_distribution = df.groupby(["speaker_id", "sentiment_category"]).size().unstack(fill_value=0)

    intensity_scores = df.groupby("speaker_id")["sentiment_score"].apply(lambda x: x.abs().mean()).reset_index()
    intensity_scores.columns = ["speaker_id", "intensity"]

    # --- Plotting ---
    sns.set_theme(style="whitegrid", palette="viridis")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Speaker Analysis Dashboard", fontsize=20)

    # 1. Word Count per Speaker (Horizontal)
    sns.barplot(data=word_counts, y="speaker_id", x="word_count", ax=axes[0, 0], orient='h')
    axes[0, 0].set_title("Word Count per Speaker")
    axes[0, 0].set_xlabel("Total Words Spoken")
    axes[0, 0].set_ylabel("Speaker ID")

    # 2. Speaker Timeline (Gantt Chart)
    speaker_colors = {speaker: color for speaker, color in zip(df['speaker_id'].unique(), sns.color_palette())}
    for _, row in df.iterrows():
        axes[0, 1].barh(row["speaker_id"], width=row["end_time"] - row["start_time"], left=row["start_time"], color=speaker_colors[row["speaker_id"]])
    axes[0, 1].set_title("Speaker Timeline")
    axes[0, 1].set_xlabel("Time (seconds)")
    axes[0, 1].set_ylabel("Speaker ID")

    # 3. Sentiment Distribution (Stacked Bar)
    sentiment_distribution.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap="RdYlGn")
    axes[1, 0].set_title("Sentiment Distribution per Speaker")
    axes[1, 0].set_xlabel("Speaker ID")
    axes[1, 0].set_ylabel("Number of Segments")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend(title="Sentiment")

    # 4. Tone Intensity (Horizontal)
    sns.barplot(data=intensity_scores, y="speaker_id", x="intensity", ax=axes[1, 1], orient='h')
    axes[1, 1].set_title("Average Tone Intensity per Speaker")
    axes[1, 1].set_xlabel("Average Intensity Score")
    axes[1, 1].set_ylabel("Speaker ID")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return img