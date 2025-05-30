import torch
import torch.nn as nn
import os
from pyprojroot import here
os.environ["HF_HOME"] = str(here("cache/HF/"))
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import lightning as L
import torchmetrics
from icecream import ic
import torchmetrics.classification
import torchmetrics.regression
from sklearn.model_selection import train_test_split
import numpy as np
from lightning.pytorch.callbacks import Callback


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MetricsCallback(Callback):
  def __init__(self):
    self.val_losses = []
    self.epochs = []
    
  def on_validation_epoch_end(self, trainer, pl_module):
    self.val_losses.append(trainer.callback_metrics["val_loss"].item())
    self.epochs.append(trainer.current_epoch)


class GoEmotionsDataset(torch.utils.data.Dataset):
  def __init__(self, dataframe):
    self.dataframe = dataframe

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    text = self.dataframe.iloc[idx].text
    # Select all columns that start with "emotion_"
    emotions = self.dataframe.iloc[idx].filter(like="emotion_").values
    return text, torch.tensor(np.array(emotions, dtype=np.float32), dtype=torch.float32).to(DEVICE)

class GoEmotionsDataModule(L.LightningDataModule):
  def __init__(self, dataframe, batch_size=64):
    super().__init__()
    self.dataframe = dataframe
    self.batch_size = batch_size
  
  def prepare_data(self):
    self.train_df, temp_df = train_test_split(self.dataframe, test_size=0.1)
    self.val_df, self.test_df = train_test_split(temp_df, test_size=0.8)

  def setup(self, stage=None):
    self.train_dataset = GoEmotionsDataset(self.train_df)
    self.val_dataset = GoEmotionsDataset(self.val_df)
    self.test_dataset = GoEmotionsDataset(self.test_df)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

  def test_dataloader(self):
    return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class DummyTokenAndSentiment(nn.Module):
  def __init__(self):
    super(DummyTokenAndSentiment, self).__init__()
    self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    self.encoder = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    self.classifier = nn.Linear(self.encoder.config.hidden_size, 3)  # 3 sentiment classes

  def forward(self, text):
    tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = self.encoder(**tokens)
    cls_rep = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
    logits = self.classifier(cls_rep)
    return torch.sigmoid(logits)

class DistilBertFinetune(L.LightningModule):
  def __init__(self, n_emotions=3):
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    self.model = AutoModelForSequenceClassification.from_pretrained(
      "distilbert-base-uncased",
      num_labels=n_emotions,
      problem_type="multi_label_classification"
    ).to(DEVICE)
    self.model.train()
    self.model.classifier = torch.nn.Linear(in_features=768, out_features=n_emotions, bias=True).to(DEVICE)
    # Freeze all layers except classifier and pre-classifier
    for param in self.model.parameters():
      param.requires_grad = False
    for param in self.model.classifier.parameters():
      param.requires_grad = True
    for param in self.model.pre_classifier.parameters():
      param.requires_grad = True
    self.sigmoid = torch.nn.Sigmoid()
    self.sig_loss = torch.nn.BCEWithLogitsLoss()
    self.f1 = torchmetrics.classification.MultilabelF1Score(num_labels=n_emotions, average="macro") # macro is average of f1s, micro is global f1
    self.rmse = torchmetrics.regression.MeanSquaredError(squared=False)
  
  def training_step(self, batch):
    x, target = batch
    tokens = self.tokenizer(
      x,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    target = target.to(DEVICE)
    logits = self.model(
      input_ids=tokens["input_ids"],
      attention_mask=tokens["attention_mask"]
    )
    loss = self.sig_loss(
      logits.logits,
      target
    )
    return loss

  def validation_step(self, batch):
    x, target = batch
    tokens = self.tokenizer(
      x,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    target = target.to(DEVICE)
    logits = self.model(
      input_ids=tokens["input_ids"],
      attention_mask=tokens["attention_mask"]
    )
    loss = self.sig_loss(
      logits.logits,
      target
    )
    y = self.sigmoid(logits.logits)
    self.log_dict({
      "val_loss": loss, 
      "val_rmse": self.rmse(y, target),
    }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def test_step(self, batch):
    x, target = batch
    tokens = self.tokenizer(
      x,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    target = target.to(DEVICE)
    logits = self.model(
      input_ids=tokens["input_ids"],
      attention_mask=tokens["attention_mask"]
    )
    loss = self.sig_loss(
      logits.logits,
      target
    )
    y = self.sigmoid(logits.logits)
    rmse = self.rmse(y, target)
    # if target > 0 then 1
    # if target == 0 then 0
    y = (y > 0.5).int() # thresholding at 0.5 # TODO possibly change this later
    target = (target > 0.01).int()
    f1 = self.f1(y, target)
    self.log_dict({
      "test_loss": loss,
      "test_f1": f1,
      "test_rmse": rmse
    }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def predict_step(self, batch):
    x = batch
    tokens = self.tokenizer(
      x,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    logits = self.model(
      input_ids=tokens["input_ids"],
      attention_mask=tokens["attention_mask"]
    )
    y = self.sigmoid(logits.logits)
    return y
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
  
  def foward(self, x):
    tokens = self.tokenizer(
      x,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}  # Move tokens to device
    logits = self.model(
      input_ids=tokens["input_ids"],
      attention_mask=tokens["attention_mask"]
    )
    y = self.sigmoid(logits.logits)
    return y