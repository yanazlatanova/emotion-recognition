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

# ------------------------ UI --------------------------------
import pandas as pd
import altair as alt

st.title("🧠 Emotion Prediction")
st.subheader("Detect emotions in text")

raw_text = st.text_area("Enter your text:")
input_text = raw_text.strip()

if st.button("Predict Emotion"):
    if not input_text:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Predicting..."):
            emotion_scores = get_emotions(input_text)
            prediction, confidence = get_emotion_max(input_text)

        col1, col2 = st.columns(2)
        with col1:
            st.success("Original Text")
            st.write(input_text)
        with col2:
            st.success("Prediction")
            st.write(prediction)
            st.write(f"Confidence: {confidence:.2%}")

        proba_df = pd.DataFrame(list(emotion_scores.items()), columns=["emotions", "confidence"])
        fig = alt.Chart(proba_df).mark_bar().encode(
            x=alt.X('emotions', sort=None),
            y='confidence',
            color='emotions',
            tooltip=['emotions', alt.Tooltip('confidence', format='.2%')]
        ).properties(width=600, height=400)

        st.altair_chart(fig, use_container_width=True)

# TO DO: add the emotions, switch different emotion number    
# bar cdom 0 to 1
# show unclear % for every predition seperatly 

# output the main emotion and how unclear it is try out with "cool"