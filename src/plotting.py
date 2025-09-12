import io

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_string_dtype
from PIL import Image


def plot_speaker_interaction_network(df, ax):
    """
    Generate a network chart of speaker interactions.
    An interaction is defined as one speaker speaking immediately after another.
    """
    df = df.copy().sort_values(by="start_time")
    interactions = []
    if not df.empty:
        last_speaker = df.iloc[0]["speaker_id"]
        for i in range(1, len(df)):
            current_speaker = df.iloc[i]["speaker_id"]
            if current_speaker != last_speaker:
                interactions.append((last_speaker, current_speaker))
            last_speaker = current_speaker

    if not interactions:
        ax.text(
            0.5,
            0.5,
            "No speaker interactions detected.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("Speaker Interaction Network")
        ax.axis("off")
        return

    G = nx.DiGraph()
    for source, target in interactions:
        if G.has_edge(source, target):
            G[source][target]["weight"] += 1
        else:
            G.add_edge(source, target, weight=1)

    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    unique_speakers = sorted(list(set(df["speaker_id"])))
    colors = sns.color_palette("viridis", n_colors=len(unique_speakers))
    speaker_color_map = {
        speaker: color for speaker, color in zip(unique_speakers, colors)
    }
    node_colors = [speaker_color_map[node] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        width=[w * 0.5 for w in weights],
        edge_color="grey",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        ax=ax,
    )

    ax.set_title("Speaker Interaction Network")
    ax.axis("off")


def plot_speaker_charts(df):
    """
    Generate four charts based on speaker diarization data using seaborn for a refined look:
    1. Horizontal bar chart of word count per speaker.
    2. Timeline chart (Gantt style) of speaker activity.
    3. Stacked bar chart of sentiment (positive, negative, neutral) per speaker.
    4. Horizontal bar chart of tone intensity per speaker.
    5. Network chart of speaker interactions.

    Parameters:
    df (pd.DataFrame): DataFrame with speaker diarization data.

    Returns:
    PIL.Image.Image: The generated chart as a PIL image.
    """
    df = df.copy()

    # --- Data Pre-processing ---
    if is_string_dtype(df["start_time"]):
        df["start_time"] = pd.to_numeric(
            df["start_time"].str.replace("s", ""), errors="coerce"
        )
    else:
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")

    if is_string_dtype(df["end_time"]):
        df["end_time"] = pd.to_numeric(
            df["end_time"].str.replace("s", ""), errors="coerce"
        )
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
    sentiment_distribution = (
        df.groupby(["speaker_id", "sentiment_category"])
        .size()
        .unstack(fill_value=0)
    )

    intensity_scores = (
        df.groupby("speaker_id")["sentiment_score"]
        .apply(lambda x: x.abs().mean())
        .reset_index()
    )
    intensity_scores.columns = ["speaker_id", "intensity"]

    # --- Plotting ---
    sns.set_theme(style="whitegrid", palette="viridis")
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("Speaker Analysis Dashboard", fontsize=24)

    # Define the grid
    ax1 = plt.subplot2grid((3, 2), (0, 0))  # Word Count
    ax2 = plt.subplot2grid((3, 2), (0, 1))  # Timeline
    ax3 = plt.subplot2grid((3, 2), (1, 0))  # Sentiment
    ax4 = plt.subplot2grid((3, 2), (1, 1))  # Intensity
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)  # Network Chart

    # 1. Word Count per Speaker (Horizontal)
    sns.barplot(
        data=word_counts, y="speaker_id", x="word_count", ax=ax1, orient="h"
    )
    ax1.set_title("Word Count per Speaker")
    ax1.set_xlabel("Total Words Spoken")
    ax1.set_ylabel("Speaker ID")

    # 2. Speaker Timeline (Gantt Chart)
    speaker_colors = {
        speaker: color
        for speaker, color in zip(
            df["speaker_id"].unique(), sns.color_palette()
        )
    }
    for _, row in df.iterrows():
        ax2.barh(
            row["speaker_id"],
            width=row["end_time"] - row["start_time"],
            left=row["start_time"],
            color=speaker_colors[row["speaker_id"]],
        )
    ax2.set_title("Speaker Timeline")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Speaker ID")

    # 3. Sentiment Distribution (Stacked Bar)
    sentiment_distribution.plot(
        kind="bar", stacked=True, ax=ax3, colormap="RdYlGn"
    )
    ax3.set_title("Sentiment Distribution per Speaker")
    ax3.set_xlabel("Speaker ID")
    ax3.set_ylabel("Number of Segments")
    ax3.tick_params(axis="x", rotation=45)
    ax3.legend(title="Sentiment")

    # 4. Tone Intensity (Horizontal)
    sns.barplot(
        data=intensity_scores, y="speaker_id", x="intensity", ax=ax4, orient="h"
    )
    ax4.set_title("Average Tone Intensity per Speaker")
    ax4.set_xlabel("Average Intensity Score")
    ax4.set_ylabel("Speaker ID")

    # 5. Speaker Interaction Network
    plot_speaker_interaction_network(df, ax5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return img
