import io

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def plot_word_count_per_speaker(df: pd.DataFrame) -> Image.Image:
    """
    Plots the number of words spoken by each speaker as a bar chart.
    Assumes the DataFrame has columns 'speaker_id' and 'transcribed_content'.
    Returns a PIL Image.
    """
    if "speaker_id" not in df.columns or "transcribed_content" not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Required columns not found.", ha="center", va="center")
        ax.axis("off")
    else:
        speaker_word_counts = df.groupby("speaker_id")["transcribed_content"].apply(
            lambda x: x.fillna("").apply(lambda s: len(str(s).split())).sum()
        )
        speakers = speaker_word_counts.index.tolist()
        color_map = plt.get_cmap("tab10")
        colors = [color_map(i % 10) for i in range(len(speakers))]
        fig, ax = plt.subplots()
        speaker_word_counts.plot(kind="bar", ax=ax, color=colors)
        ax.set_xlabel("Speaker")
        ax.set_ylabel("Total Words Spoken")
        ax.set_title("Number of Words Spoken by Each Speaker")
        plt.xticks(rotation=45)
        plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_speaker_timeline(df: pd.DataFrame) -> Image.Image:
    """
    Plots a timeline visualization of speaker segments.
    Assumes the DataFrame has columns: 'speaker_id', 'start_time', 'end_time'.
    Returns a PIL Image.
    """
    if not all(col in df.columns for col in ["speaker_id", "start_time", "end_time"]):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Required columns not found.", ha="center", va="center")
        ax.axis("off")
    else:
        speakers = sorted(df["speaker_id"].unique())
        color_map = plt.get_cmap("tab10")
        speaker_to_color = {
            speaker: color_map(i % 10) for i, speaker in enumerate(speakers)
        }
        fig, ax = plt.subplots(figsize=(10, max(2, len(speakers) * 0.5)))
        for idx, speaker in enumerate(speakers):
            speaker_df = df[df["speaker_id"] == speaker]
            for _, row in speaker_df.iterrows():
                ax.plot(
                    [row["start_time"], row["end_time"]],
                    [idx, idx],
                    lw=8,
                    color=speaker_to_color[speaker],
                )
        ax.set_yticks(range(len(speakers)))
        ax.set_yticklabels(speakers)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speaker")
        ax.set_title("Speaker Timeline Visualization")
        plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img
