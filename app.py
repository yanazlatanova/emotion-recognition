import streamlit as st
import torch
torch.classes.__path__ = []
from models.model_definition import DistilBertFinetune  # your class import
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from pyprojroot import here
import lightning as L

# ------------------------ Initializations ------------------------
torch.serialization.add_safe_globals({"DistilBertFinetune": DistilBertFinetune})
LABEL_MAP = {0: "positive", 1: "negative", 2: "unclear"}


# ------------------------ Caching Loader ------------------------
@st.cache_resource
def load_model():
    distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    model = DistilBertFinetune.load_from_checkpoint(
        "models/dummy_ligh.ckpt",
        distilbert_model=distilbert_model,
        tokenizer=tokenizer,
        n_emotions=3
    )

    trainer = L.Trainer(
        default_root_dir=here("cache/lightning"),
        logger=False,
        enable_checkpointing=False
    )
    return model, trainer

model, trainer = load_model()

# ------------------------ Functions ------------------------

@torch.no_grad()
def predict(text: str):
    return trainer.predict(model, [text])

def get_emotion_max(text: str) -> str:
    output = predict(text)
    output_tensor = output[0] # get the tensor from the list
    pred = torch.argmax(output_tensor, dim=1).item()
    return LABEL_MAP.get(pred, "unknown")


# ------------------------ UI --------------------------------
st.title("🧠 Emotion Prediction")
text_input = st.text_area("Enter your text:")

if st.button("Predict Emotion"):
    if text_input.strip():
        st.success(f"Predicted Emotion: {get_emotion_max(text_input)}")
    else:
        st.warning("Please enter some text.")
