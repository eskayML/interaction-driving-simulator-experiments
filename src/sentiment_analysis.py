import os
import warnings

os.environ["TORCH_HOME"] = "./model_weights/torch"
os.environ["HF_HOME"] = "./model_weights/huggingface"
os.environ["PYANNOTE_CACHE"] = "./model_weights/torch/pyannote"

from transformers import pipeline

warnings.filterwarnings("ignore")


def analyze_sentiment(text):
    """Perform sentiment analysis and return a composite score from -1 to +1."""
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
        return_all_scores=True,
    )
    result = sentiment_pipeline(text)[0]
    scores = {item["label"]: item["score"] for item in result}
    negative = scores.get("LABEL_0", 0)
    neutral = scores.get("LABEL_1", 0)
    positive = scores.get("LABEL_2", 0)
    sentiment_score = (positive * 1) + (neutral * 0) + (negative * -1)
    return sentiment_score
