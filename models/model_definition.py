import torch
import torch.nn as nn
import os
from pyprojroot import here
os.environ["HF_HOME"] = str(here("cache/HF/"))
from transformers import AutoTokenizer, AutoModel

class TokenizationAndSentimentModel(nn.Module):
  def __init__(self):
    super(TokenizationAndSentimentModel, self).__init__()
    self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    self.encoder = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    self.classifier = nn.Linear(self.encoder.config.hidden_size, 3)  # 3 sentiment classes

  def forward(self, text):
    tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = self.encoder(**tokens)
    cls_rep = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
    logits = self.classifier(cls_rep)
    return torch.sigmoid(logits)


import lightning as L
import torchmetrics
from icecream import ic
import torchmetrics.classification
import torchmetrics.regression
import torchmetrics.text

class DistilBertFinetune(L.LightningModule):
  def __init__(self, distilbert_model, tokenizer, n_emotions=3):
    super().__init__()
    self.tokenizer = tokenizer
    self.model = distilbert_model
    self.model.classifier = torch.nn.Linear(in_features=768, out_features=n_emotions, bias=True)
    self.sigmoid = torch.nn.Sigmoid()
    self.sig_loss = torch.nn.BCEWithLogitsLoss()
    # self.preplexity = torchmetrics.text.Perplexity()
    self.f1 = torchmetrics.classification.MultilabelF1Score(num_labels=n_emotions, average="macro") # macro is average of f1s, micro is global f1
    self.rmse = torchmetrics.regression.MeanSquaredError(squared=False)
  
  def foward(self, x):
    tokens = self.tokenizer(
      x,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512
    )
    logits = self.model(
      input_ids=tokens["input_ids"],
      attention_mask=tokens["attention_mask"]
    )
    y = self.sigmoid(logits.logits)
    return y

  def training_step(self, batch):
    x, target = batch
    tokens = self.tokenizer(
      x,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512
    )
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
      # "val_perplexity": self.preplexity(preds=y, target=target), # ValueError: Input tensor `preds` is expected to have 3 dimensions, [batch_size, seq_len, vocab_size], but got 2.
      "val_rmse": self.rmse(y, target),
    }, on_step=False, on_epoch=True, prog_bar=True, logger=False)
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
    logits = self.model(
      input_ids=tokens["input_ids"],
      attention_mask=tokens["attention_mask"]
    )
    loss = self.sig_loss(
      logits.logits,
      target
    )
    y = self.sigmoid(logits.logits)
    # preplexity = self.preplexity(preds=y, target=target)
    rmse = self.rmse(y, target)
    # if target > 0 then 1
    # if target == 0 then 0
    y = (y > 0.5).int() # thresholding at 0.5 # TODO possibly change this later
    target = (target > 0.01).int()
    f1 = self.f1(y, target)
    self.log_dict({
      "test_loss": loss,
      # "test_perplexity": preplexity,
      "test_f1": f1,
      "test_rmse": rmse
    }, on_step=False, on_epoch=True, prog_bar=True, logger=False)
    return loss
  
  def predict_step(self, batch):
    x = batch # TODO check if this is right
    tokens = self.tokenizer(
      x,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512
    )
    logits = self.model(
      input_ids=tokens["input_ids"],
      attention_mask=tokens["attention_mask"]
    )
    y = self.sigmoid(logits.logits)
    return y
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer