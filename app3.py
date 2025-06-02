import streamlit as st
import torch
torch.classes.__path__ = []
from models.model_definitions import DistilBertFinetuneOnWeightedMSE
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from pyprojroot import here
import lightning as L
import pandas as pd
import altair as alt

# ------------------------ Setup ------------------------

torch.serialization.add_safe_globals({"DistilBertFinetuneOnWeightedMSE": DistilBertFinetuneOnWeightedMSE})

EMOTION_SETS = {
    3: ['positive', 'negative', 'unclear'],
    7: ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'unclear'],
    28: ['sadness', 'unclear', 'love', 'gratitude', 'disapproval', 'amusement', 'disappointment', 'disgust',
         'admiration', 'realization', 'annoyance', 'confusion', 'optimism', 'curiosity', 'excitement', 'caring', 'joy',
         'remorse', 'approval', 'nervousness', 'embarrassment', 'surprise', 'anger', 'grief', 'pride', 'desire',
         'relief', 'fear']
}

EMOJI_MAP = {
    "sadness": "😢", "love": "❤️", "gratitude": "🙏", "disapproval": "👎",
    "amusement": "😂", "disappointment": "😞", "disgust": "🤢", "admiration": "👏", "realization": "💡",
    "annoyance": "😒", "confusion": "😕", "optimism": "🌈", "curiosity": "🤔", "excitement": "🤩", "caring": "🤗",
    "joy": "😊", "remorse": "😔", "approval": "👍", "nervousness": "😬", "embarrassment": "😳", "surprise": "😮",
    "anger": "😡", "grief": "😭", "pride": "🏆", "desire": "🔥", "relief": "😌", "fear": "😨", "unclear": "❓", "positive": "✅", 
    "negative": "❌"
}

# ------------------------ Caching Loader ------------------------

@st.cache_resource
def load_all_models():
    """
    Load and cache all emotion models.
    Returns a dictionary: {3: model3, 6: model6, 27: model27}
    """
    return {
        3: DistilBertFinetuneOnWeightedMSE.load_from_checkpoint(
            "models/mse_3.ckpt", n_emotions=3
        ).eval(),
        7: DistilBertFinetuneOnWeightedMSE.load_from_checkpoint(
            "models/mse_7.ckpt", n_emotions=7  # trained with 7 classes incl. 'unclear'
        ).eval(),
        28: DistilBertFinetuneOnWeightedMSE.load_from_checkpoint(
            "models/mse_28.ckpt", n_emotions=28
        ).eval()
    }

# ------------------------ UI ------------------------

st.title("🧠 Emotion Prediction")
st.subheader("Detect emotions in text")

# emotion_count = st.selectbox("Select number of emotion categories", options=[3, 7, 28])
# emotion_count = st.selectbox("Select number of emotions", [3, 7, 28], index=1)

label_to_value = {
    "Positive/Negative": 3,
    "6 Emotions": 7,
    "27 Emotions": 28
}

selected_label = st.selectbox("Select taxonomy", list(label_to_value.keys()))
emotion_count = label_to_value[selected_label]


all_models = load_all_models()
model = all_models[emotion_count]

COLUMNS = EMOTION_SETS[emotion_count]
# model = load_model(emotion_count)

raw_text = st.text_area("Enter your text:")
input_text = raw_text.strip()


# ------------------------ Inference Functions ------------------------

@torch.no_grad()
def predict(text: str):
    return model([text])

def get_emotions(text: str) -> dict:
    output = predict(text)
    output_tensor = output[0]
    values = output_tensor.squeeze().tolist()
    return {label: float(score) for label, score in zip(COLUMNS, values)}

def get_top_emotion(scores: dict) -> tuple[str, float, float, bool]:
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    unclear_score = scores.get("unclear", 0.0)
    if sorted_scores[0][0] == "unclear":
        for label, score in sorted_scores[1:]:
            if label != "unclear":
                return label, score, unclear_score, True
    return sorted_scores[0][0], sorted_scores[0][1], unclear_score, False

# ------------------------ UI ------------------------

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
                f"<p style='color:{unclear_color}; font-size: 18px;'>Unclear Confidence {unclear_prob:.2%} ❓</p>",
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
