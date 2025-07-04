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
        fig, ax = plt.subplots()
        speaker_word_counts.plot(kind="bar", ax=ax, color="skyblue")
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
