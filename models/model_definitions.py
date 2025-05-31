import torch
import torch.nn as nn
import os
from pyprojroot import here
os.environ["HF_HOME"] = str(here("cache/HF/"))
import torchmetrics.retrieval
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
    
    # F1 standard will be a postivie emotion if theres at least one rater who rated it as positive
    # F1 interesting will be a positive emotion for every emotion > 0.8, but if there isn't at least one, the highest emotion will be considered positive
    self.f1_stand = torchmetrics.classification.MultilabelF1Score(num_labels=n_emotions, average="macro") # macro is average of f1s, micro is global f1
    self.f1_interest = torchmetrics.classification.MultilabelF1Score(num_labels=n_emotions, average="macro") # macro is average of f1s, micro is global f1
    self.rmse = torchmetrics.regression.MeanSquaredError(squared=False)
    self.nDGC = NDCG(k=None, dist_sync_on_step=False)
    self.approx_nDGC = ApproxNDCGLoss(tau=1)
    # changed: removed self.expected_nDGC

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
    cross_entropy = self.sig_loss(
      logits.logits,
      target
    )
    y = self.sigmoid(logits.logits)
    self.log_dict({
      "val_cross_entropy": cross_entropy, 
      "val_rmse": self.rmse(y, target),
      "val_nDGC": self.nDGC(preds=y, target=target),
      "val_approxNDCG": self.approx_nDGC(logits.logits, target),
    }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return cross_entropy
  
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
    cross_entropy = self.sig_loss(
      logits.logits,
      target
    )
    y = self.sigmoid(logits.logits)
    rmse = self.rmse(y, target)
    ndcg = self.nDGC(preds=y, target=target)
    approx_ndcg = self.approx_nDGC(logits.logits, target)
    y = (y > 0.5).int()
    target_stand = (target > 0.01).int()
    target_interest = (target > 0.8).int()
    if target_interest.sum() == 0:
      # If no emotion is above 0.8, take the highest emotion
      max_emotion = target.argmax(dim=1, keepdim=True)
      target_interest = torch.zeros_like(target, dtype=torch.int)
      target_interest.scatter_(1, max_emotion, 1)
    f1_stand = self.f1_stand(y, target_stand)
    f1_interest = self.f1_interest(y, target_interest)
    
    self.log_dict({
      "test_cross_entropy": cross_entropy,
      "test_f1_stand": f1_stand,
      "test_f1_interest": f1_interest,
      "test_rmse": rmse,
      "test_nDGC": ndcg,
      "test_approxNDCG": approx_ndcg
    }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return cross_entropy
  
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

class NDCG(torchmetrics.Metric):
    def __init__(self, k: int = None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k  # Compute NDCG@k (None = full)

        self.add_state("sum_ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _dcg(self, rel_sorted):
        # rel_sorted: [batch_size, list_size]
        positions = torch.arange(1, rel_sorted.size(1) + 1, device=rel_sorted.device).float()
        discounts = torch.log2(positions + 1.0)
        gains = 2 ** rel_sorted - 1
        return (gains / discounts).sum(dim=1)  # [batch_size]

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_size, list_size = preds.shape
        k = self.k if self.k is not None else list_size

        # Get indices that would sort predictions descending
        sorted_indices = torch.argsort(preds, dim=1, descending=True)
        topk_indices = sorted_indices[:, :k]  # [batch_size, k]

        # Gather relevance scores in sorted order
        rel_sorted = torch.gather(target, dim=1, index=topk_indices)

        # Compute DCG
        dcg = self._dcg(rel_sorted)

        # Compute IDCG from ideal sorting of ground truth
        sorted_rel_gt, _ = torch.sort(target, dim=1, descending=True)
        ideal_rel_topk = sorted_rel_gt[:, :k]
        idcg = self._dcg(ideal_rel_topk)

        # Avoid division by zero
        ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))

        self.sum_ndcg += ndcg.sum()
        self.total += batch_size

    def compute(self):
        return self.sum_ndcg / self.total

class SoftRankNDCGLoss(nn.Module):
    def __init__(self, sigma=1.0, k=None):
        super().__init__()
        self.sigma = sigma
        self.k = k

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: Predicted scores, shape [batch_size, list_size]
            target: Ground truth relevance labels, shape [batch_size, list_size]
        Returns:
            A scalar loss: negative Expected NDCG
        """
        batch_size, list_size = preds.shape

        # Step 1: Compute pairwise probabilities
        diff = preds.unsqueeze(2) - preds.unsqueeze(1)  # [B, L, L]
        normal = torch.distributions.normal.Normal(0, self.sigma * (2 ** 0.5))
        p = normal.cdf(diff)  # [B, L, L]
        diag_mask = torch.eye(list_size, device=preds.device).bool()
        p = p.masked_fill(diag_mask.unsqueeze(0), 0)

        # Step 2: Compute expected ranks
        expected_ranks = 1 + p.sum(dim=2)  # [B, L]

        # Step 3: Truncate to top-k if needed
        if self.k is not None:
            topk_idx = torch.topk(target, self.k, dim=1).indices
            target = torch.gather(target, dim=1, index=topk_idx)
            expected_ranks = torch.gather(expected_ranks, dim=1, index=topk_idx)

        # Step 4: Compute DCG and IDCG
        gains = 2 ** target - 1
        discounts = torch.log2(expected_ranks + 1)
        dcg = (gains / discounts).sum(dim=1)  # [B]

        ideal_target, _ = torch.sort(target, descending=True, dim=1)
        if self.k is not None:
            ideal_target = ideal_target[:, :self.k]
        ideal_gains = 2 ** ideal_target - 1
        ideal_discounts = torch.log2(
            torch.arange(1, ideal_target.size(1) + 1, device=preds.device).float() + 1
        )
        idcg = (ideal_gains / ideal_discounts).sum(dim=1)  # [B]

        # Step 5: Compute NDCG and loss
        ndcg = dcg / (idcg + 1e-10)
        loss = -ndcg.mean()  # negative expected NDCG

        return loss

class DistilBertFinetuneOnDCG(L.LightningModule):
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
    self.nDGC = NDCG(k=None, dist_sync_on_step=False)
    self.approx_nDGC = ApproxNDCGLoss(tau=1)
    # F1 standard will be a postivie emotion if theres at least one rater who rated it as positive
    # F1 interesting will be a positive emotion for every emotion > 0.8, but if there isn't at least one, the highest emotion will be considered positive
    self.f1_stand = torchmetrics.classification.MultilabelF1Score(num_labels=n_emotions, average="macro") # macro is average of f1s, micro is global f1
    self.f1_interest = torchmetrics.classification.MultilabelF1Score(num_labels=n_emotions, average="macro") # macro is average of f1s, micro is global f1
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
    y = self.sigmoid(logits.logits)
    loss = self.approx_nDGC(logits.logits, target)
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
    y = self.sigmoid(logits.logits)
    approx_nDGC = self.approx_nDGC(logits.logits, target)
    self.log_dict({
      "val_approxNDCG": 1-approx_nDGC, 
      "val_cross_entropy": self.sig_loss(logits.logits, target),
      "val_nDGC": self.nDGC(preds=y, target=target),
      "val_rmse": self.rmse(y, target),
    }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return approx_nDGC
  
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
    cross_entropy = self.sig_loss(
      logits.logits,
      target
    )
    # note: some of therse can be moved to the setup fucntion to increase speed but it's fast enough to be here
    y = self.sigmoid(logits.logits)
    rmse = self.rmse(y, target)
    ndcg = self.nDGC(preds=y, target=target)
    approx_ndcg = loss = self.approx_nDGC(logits.logits, target)
    y_bin = (y > 0.5).int()
    target_stand = (target > 0.01).int()
    target_interest = (target > 0.8).int()
    if target_interest.sum() == 0:
      max_emotion = target.argmax(dim=1, keepdim=True)
      target_interest = torch.zeros_like(target, dtype=torch.int)
      target_interest.scatter_(1, max_emotion, 1)
    f1_stand = self.f1_stand(y_bin, target_stand)
    f1_interest = self.f1_interest(y_bin, target_interest)

    self.log_dict({
      "test_cross_entropy": cross_entropy,
      "test_f1_stand": f1_stand,
      "test_f1_interest": f1_interest,
      "test_rmse": rmse,
      "test_nDGC": ndcg,
      "test_approxNDCG": 1-approx_ndcg
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

class NDCG(torchmetrics.Metric):
  def __init__(self, k: int = None, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.k = k  # Compute NDCG@k (None = full)

    self.add_state("sum_ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

  def _dcg(self, rel_sorted):
    # rel_sorted: [batch_size, list_size]
    positions = torch.arange(1, rel_sorted.size(1) + 1, device=rel_sorted.device).float()
    discounts = torch.log2(positions + 1.0)
    gains = 2 ** rel_sorted - 1
    return (gains / discounts).sum(dim=1)  # [batch_size]

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    batch_size, list_size = preds.shape
    k = self.k if self.k is not None else list_size

    # Get indices that would sort predictions descending
    sorted_indices = torch.argsort(preds, dim=1, descending=True)
    topk_indices = sorted_indices[:, :k]  # [batch_size, k]

    # Gather relevance scores in sorted order
    rel_sorted = torch.gather(target, dim=1, index=topk_indices)

    # Compute DCG
    dcg = self._dcg(rel_sorted)

    # Compute IDCG from ideal sorting of ground truth
    sorted_rel_gt, _ = torch.sort(target, dim=1, descending=True)
    ideal_rel_topk = sorted_rel_gt[:, :k]
    idcg = self._dcg(ideal_rel_topk)

    # Avoid division by zero
    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))

    self.sum_ndcg += ndcg.sum()
    self.total += batch_size

  def compute(self):
    return self.sum_ndcg / self.total

class SoftRankNDCGLoss(nn.Module):
  def __init__(self, sigma=1.0, k=None):
    super().__init__()
    self.sigma = sigma
    self.k = k

  def forward(self, preds: torch.Tensor, target: torch.Tensor):
    """
    Args:
        preds: Predicted scores, shape [batch_size, list_size]
        target: Ground truth relevance labels, shape [batch_size, list_size]
    Returns:
        A scalar loss: negative Expected NDCG
    """
    batch_size, list_size = preds.shape

    # Step 1: Compute pairwise probabilities
    diff = preds.unsqueeze(2) - preds.unsqueeze(1)  # [B, L, L]
    normal = torch.distributions.normal.Normal(0, self.sigma * (2 ** 0.5))
    p = normal.cdf(diff)  # [B, L, L]
    diag_mask = torch.eye(list_size, device=preds.device).bool()
    p = p.masked_fill(diag_mask.unsqueeze(0), 0)

    # Step 2: Compute expected ranks
    expected_ranks = 1 + p.sum(dim=2)  # [B, L]

    # Step 3: Truncate to top-k if needed
    if self.k is not None:
        topk_idx = torch.topk(target, self.k, dim=1).indices
        target = torch.gather(target, dim=1, index=topk_idx)
        expected_ranks = torch.gather(expected_ranks, dim=1, index=topk_idx)

    # Step 4: Compute DCG and IDCG
    gains = 2 ** target - 1
    discounts = torch.log2(expected_ranks + 1)
    dcg = (gains / discounts).sum(dim=1)  # [B]

    ideal_target, _ = torch.sort(target, descending=True, dim=1)
    if self.k is not None:
        ideal_target = ideal_target[:, :self.k]
    ideal_gains = 2 ** ideal_target - 1
    ideal_discounts = torch.log2(
        torch.arange(1, ideal_target.size(1) + 1, device=preds.device).float() + 1
    )
    idcg = (ideal_gains / ideal_discounts).sum(dim=1)  # [B]

    # Step 5: Compute NDCG and loss
    ndcg = dcg / (idcg + 1e-10)
    loss = -ndcg.mean()  # negative expected NDCG

    return loss

class DistilBertFinetuneOnDCG(L.LightningModule):
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
    self.nDGC = NDCG(k=None, dist_sync_on_step=False)
    self.approx_nDGC = ApproxNDCGLoss(tau=1)
    # F1 standard will be a postivie emotion if theres at least one rater who rated it as positive
    # F1 interesting will be a positive emotion for every emotion > 0.8, but if there isn't at least one, the highest emotion will be considered positive
    self.f1_stand = torchmetrics.classification.MultilabelF1Score(num_labels=n_emotions, average="macro") # macro is average of f1s, micro is global f1
    self.f1_interest = torchmetrics.classification.MultilabelF1Score(num_labels=n_emotions, average="macro") # macro is average of f1s, micro is global f1
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
    # y = self.sigmoid(logits.logits)
    loss = self.approx_nDGC(logits.logits, target)
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
    y = self.sigmoid(logits.logits)
    approx_nDGC = self.approx_nDGC(logits.logits, target)
    self.log_dict({
      "val_approxNDCG": 1-approx_nDGC, 
      "val_cross_entropy": self.sig_loss(logits.logits, target),
      "val_nDGC": self.nDGC(preds=y, target=target),
      "val_rmse": self.rmse(y, target),
    }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return approx_nDGC
  
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
    cross_entropy = self.sig_loss(
      logits.logits,
      target
    )
    # note: some of therse can be moved to the setup fucntion to increase speed but it's fast enough to be here
    y = self.sigmoid(logits.logits)
    rmse = self.rmse(y, target)
    ndcg = self.nDGC(preds=y, target=target)
    approx_ndcg = loss = self.approx_nDGC(logits.logits, target)
    y_bin = (y > 0.5).int()
    target_stand = (target > 0.01).int()
    target_interest = (target > 0.8).int()
    if target_interest.sum() == 0:
      max_emotion = target.argmax(dim=1, keepdim=True)
      target_interest = torch.zeros_like(target, dtype=torch.int)
      target_interest.scatter_(1, max_emotion, 1)
    f1_stand = self.f1_stand(y_bin, target_stand)
    f1_interest = self.f1_interest(y_bin, target_interest)

    self.log_dict({
      "test_cross_entropy": cross_entropy,
      "test_f1_stand": f1_stand,
      "test_f1_interest": f1_interest,
      "test_rmse": rmse,
      "test_nDGC": ndcg,
      "test_approxNDCG": 1-approx_ndcg
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

class ApproxNDCGLoss(nn.Module):
    """
    Differentiable approximation to NDCG, suitable as a loss function.
    The loss is defined as (1 - ApproxNDCG), so minimizing the loss maximizes ranking quality.
    """
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        scores: predicted relevance scores (float) — shape (batch_size, n)
        labels: true relevance scores (float, in [0, 1]) — shape (batch_size, n)
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            labels = labels.unsqueeze(0)
        batch_size, n = scores.size()
        device = scores.device

        losses = []
        for i in range(batch_size):
            s = scores[i]
            l = labels[i]
            # Compute pairwise comparisons
            score_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [n, n]
            P = torch.sigmoid(score_diff / self.tau)      # [n, n]
            soft_rank = torch.sum(P, dim=1)               # [n]

            # Compute DCG
            gains = torch.pow(2.0, l) - 1.0
            discounts = torch.log2(1.0 + soft_rank)
            dcg = torch.sum(gains / discounts)

            # Compute ideal DCG (sorted by label)
            sorted_labels, _ = torch.sort(l, descending=True)
            ideal_gains = torch.pow(2.0, sorted_labels) - 1.0
            ideal_discounts = torch.log2(torch.arange(2, 2 + n, device=device).float())
            idcg = torch.sum(ideal_gains / ideal_discounts)

            ndcg = dcg / idcg.clamp(min=1e-8)
            loss = 1.0 - ndcg  # we want to maximize NDCG ⇒ minimize (1 - NDCG)
            losses.append(loss)
        return torch.stack(losses).mean()