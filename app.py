import streamlit as st
import torch
torch.classes.__path__ = []
from models.model_definitions import DistilBertFinetuneOnWeightedMSE  # your class import
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from pyprojroot import here
import lightning as L

# ------------------------ Initializations ------------------------
torch.serialization.add_safe_globals({"DistilBertFinetuneOnWeightedMSE": DistilBertFinetuneOnWeightedMSE})

COLUMNS = ['sadness', 'unclear',
           'love', 'gratitude', 'disapproval',
           'amusement', 'disappointment', 'disgust',
           'admiration', 'realization', 'annoyance',
           'confusion', 'optimism', 'curiosity',
           'excitement', 'caring', 'joy',
           'remorse', 'approval', 'nervousness',
           'embarrassment', 'surprise', 'anger',
           'grief', 'pride', 'desire', 'relief',
           'fear']

EMOJI_MAP = {
    "sadness": "😢", "love": "❤️", "gratitude": "🙏", "disapproval": "👎",
    "amusement": "😂", "disappointment": "😞", "disgust": "🤢",
    "admiration": "👏", "realization": "💡", "annoyance": "😒",
    "confusion": "😕", "optimism": "🌈", "curiosity": "🤔",
    "excitement": "🤩", "caring": "🤗", "joy": "😊",
    "remorse": "😔", "approval": "👍", "nervousness": "😬",
    "embarrassment": "😳", "surprise": "😮", "anger": "😡",
    "grief": "😭", "pride": "🏆", "desire": "🔥", "relief": "😌",
    "fear": "😨", "unclear": "❓"
}

# ------------------------ Caching Loader ------------------------

@st.cache_resource
def load_model() -> DistilBertFinetuneOnWeightedMSE:
    """
    Load the pretrained model from checkpoint.
    Cached to avoid reloading on every run.
    """
    model = DistilBertFinetuneOnWeightedMSE.load_from_checkpoint(
        "models/mse_28.ckpt",
        n_emotions=len(COLUMNS)
    )
    model.eval()
    return model

model = load_model()

# ------------------------ Functions ------------------------

@torch.no_grad()
def predict(text: str):
    return model([text])

def get_emotions(text: str) -> str:
    """
    Get emotion probabilities/scores for each emotion label.
    Returns a dict {emotion_label: score}.
    """
    output = predict(text)
    output_tensor = output[0] # get the tensor from the list
    values = output_tensor.squeeze().tolist()  # Remove tensor nesting and convert to list
    scores = {label: float(score) for label, score in zip(COLUMNS, values)}
    return scores

def get_emotion_max(text: str) -> tuple[str, float]:
    """
    Return the emotion label with the highest score and the score itself.
    """
    emotion_scores = get_emotions(text)
    top_emotion = max(emotion_scores, key=emotion_scores.get)
    confidence = emotion_scores[top_emotion]
    return top_emotion, confidence

def get_top_emotion(scores: dict) -> tuple[str, float, float, bool]:
    """
    Returns (top_emotion, top_confidence, unclear_confidence)
    If 'unclear' is highest, returns the next highest emotion.
    """
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    unclear_score = scores.get("unclear", 0.0)

    if sorted_scores[0][0] == "unclear":
        # Find the next best emotion
        for label, score in sorted_scores[1:]:
            if label != "unclear":
                return label, score, unclear_score, True
    return sorted_scores[0][0], sorted_scores[0][1], unclear_score, False


# ------------------------ UI --------------------------------
import pandas as pd
import altair as alt

st.title("🧠 Emotion Prediction")
st.subheader("Detect emotions in text")

raw_text = st.text_area("Enter your text:")
input_text = raw_text.strip()

UNCLEAR_THRESHOLD = 0.30

if st.button("Predict Emotion"):
    if not input_text:
        st.warning("Please enter some text.")
    else:        
        with st.spinner("Predicting..."):
            emotion_scores = get_emotions(input_text)
            prediction, confidence, unclear_prob, is_unclear_top = get_top_emotion(emotion_scores)

        st.success("Prediction")
        col1, col2 = st.columns(2)        

        with col1:
            emoji = EMOJI_MAP.get(prediction, "")
            st.markdown(
                f"<h2 style='margin-top: 0;'>{prediction.capitalize()} {emoji}</h2>"
                f"<p style='font-size: 18px;'>Confidence: {confidence:.2%}</p>",
                unsafe_allow_html=True
            )
        
            unclear_color = "red" if is_unclear_top else "gray"
            st.markdown(
                f"<h4 style='margin-bottom: 0;'>Unclear Confidence</h4>"
                f"<p style='color:{unclear_color}; font-size: 18px;'>{unclear_prob:.2%} ❓</p>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Probability Distribution")

        proba_df = pd.DataFrame([
            (f"{EMOJI_MAP.get(emotion, '')} {emotion}", score)
            for emotion, score in emotion_scores.items()
        ], columns=["emotions", "confidence"])

        fig = alt.Chart(proba_df).mark_bar().encode(
            x=alt.X('emotions', sort=None),
            y='confidence',
            color='emotions',
            tooltip=['emotions', alt.Tooltip('confidence', format='.2%')]
        ).properties(width=600, height=400)

        st.altair_chart(fig, use_container_width=True)