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


# ================== CALLBACKS ==================
class MetricsCallback(Callback):
  def __init__(self):
    self.val_losses = []
    self.epochs = []

  def on_validation_epoch_end(self, trainer, pl_module):
    self.val_losses.append(trainer.callback_metrics["val_loss"].item())
    self.epochs.append(trainer.current_epoch)


# ================== DATA MODULES ==================
class GoEmotionsDataset(torch.utils.data.Dataset):
  def __init__(self, dataframe):
    self.dataframe = dataframe

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    text = self.dataframe.iloc[idx].text
    emotions = self.dataframe.iloc[idx].filter(like="emotion_").values
    return text, torch.tensor(
      np.array(emotions, dtype=np.float32), dtype=torch.float32
    ).to(DEVICE)


class GoEmotionsDataModule(L.LightningDataModule):
  def __init__(self, dataframe, batch_size=64, split_seed=1):
    super().__init__()
    self.dataframe = dataframe
    self.batch_size = batch_size
    self.split_seed = split_seed
    
  def prepare_data(self):
    L.seed_everything(self.split_seed, workers=True)
    self.train_df, temp_df = train_test_split(self.dataframe, test_size=0.3)
    self.val_df, self.test_df = train_test_split(temp_df, test_size=0.5)

  def setup(self, stage=None):
    self.train_dataset = GoEmotionsDataset(self.train_df)
    self.val_dataset = GoEmotionsDataset(self.val_df)
    self.test_dataset = GoEmotionsDataset(self.test_df)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(
      self.train_dataset, batch_size=self.batch_size, shuffle=True
    )

  def val_dataloader(self):
    return torch.utils.data.DataLoader(
      self.val_dataset, batch_size=self.batch_size, shuffle=False
    )

  def test_dataloader(self):
    return torch.utils.data.DataLoader(
      self.test_dataset, batch_size=self.batch_size, shuffle=False
    )


# ================== CUSTOM METRICS ==================
class NDCG(torchmetrics.Metric):
  def __init__(self, k: int = None, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.k = k
    self.add_state("sum_ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

  def _dcg(self, rel_sorted):
    positions = torch.arange(
      1, rel_sorted.size(1) + 1, device=rel_sorted.device
    ).float()
    discounts = torch.log2(positions + 1.0)
    gains = 2**rel_sorted - 1
    return (gains / discounts).sum(dim=1)

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    batch_size, list_size = preds.shape
    k = self.k if self.k is not None else list_size

    sorted_indices = torch.argsort(preds, dim=1, descending=True)
    topk_indices = sorted_indices[:, :k]
    rel_sorted = torch.gather(target, dim=1, index=topk_indices)
    dcg = self._dcg(rel_sorted)

    sorted_rel_gt, _ = torch.sort(target, dim=1, descending=True)
    ideal_rel_topk = sorted_rel_gt[:, :k]
    idcg = self._dcg(ideal_rel_topk)

    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
    self.sum_ndcg += ndcg.sum()
    self.total += batch_size

  def compute(self):
    return self.sum_ndcg / self.total


class SoftRankExpectedNDCG(torchmetrics.Metric):
  def __init__(self, sigma=1.0, k=None, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.sigma = sigma
    self.k = k
    self.add_state("sum_ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

  def _expected_rank(self, scores):
    batch_size, list_size = scores.shape
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    normal = torch.distributions.normal.Normal(0, self.sigma * (2**0.5))
    p = normal.cdf(diff)
    diag_mask = torch.eye(list_size, device=scores.device).bool()
    p = p.masked_fill(diag_mask.unsqueeze(0), 0)
    expected_ranks = 1 + p.sum(dim=2)
    return expected_ranks

  def _dcg(self, rel, ranks):
    if self.k is not None:
      rel = rel[:, : self.k]
      ranks = ranks[:, : self.k]
    gains = 2**rel - 1
    discounts = torch.log2(ranks + 1)
    return (gains / discounts).sum(dim=1)

  def _idcg(self, rel):
    sorted_rel, _ = torch.sort(rel, descending=True, dim=1)
    if self.k is not None:
      sorted_rel = sorted_rel[:, : self.k]
    gains = 2**sorted_rel - 1
    discounts = torch.log2(
      torch.arange(1, sorted_rel.size(1) + 1, device=rel.device).float() + 1
    )
    idcg = (gains / discounts).sum(dim=1)
    return idcg

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    expected_ranks = self._expected_rank(preds)
    dcg = self._dcg(target, expected_ranks)
    idcg = self._idcg(target)
    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
    self.sum_ndcg += ndcg.sum()
    self.total += ndcg.size(0)

  def compute(self):
    return self.sum_ndcg / self.total


# ================== LOSS FUNCTIONS ==================
class ApproxNDCGLoss(nn.Module):
  def __init__(self, tau=1.0):
    super().__init__()
    self.tau = tau

  def forward(self, scores, targets):
    batch_size, list_size = scores.size()
    ideal_targets, _ = torch.sort(targets, descending=True, dim=1)
    gains_ideal = torch.pow(2.0, ideal_targets) - 1.0
    discounts = torch.log2(
      torch.arange(list_size, dtype=torch.float32, device=scores.device) + 2.0
    )
    ideal_dcg = torch.sum(gains_ideal / discounts, dim=1)

    scaled_scores = scores / self.tau
    probs = torch.softmax(scaled_scores, dim=1)
    gains = torch.pow(2.0, targets) - 1.0
    expected_dcg = torch.sum(probs * gains / discounts, dim=1)
    approx_ndcg = expected_dcg / (ideal_dcg + 1e-8)
    loss = 1.0 - approx_ndcg
    return loss.mean()


class SoftRankNDCGLoss(nn.Module):
  def __init__(self, sigma=1.0, k=None):
    super().__init__()
    self.sigma = sigma
    self.k = k

  def forward(self, preds: torch.Tensor, target: torch.Tensor):
    batch_size, list_size = preds.shape
    diff = preds.unsqueeze(2) - preds.unsqueeze(1)
    normal = torch.distributions.normal.Normal(0, self.sigma * (2**0.5))
    p = normal.cdf(diff)
    diag_mask = torch.eye(list_size, device=preds.device).bool()
    p = p.masked_fill(diag_mask.unsqueeze(0), 0)
    expected_ranks = 1 + p.sum(dim=2)

    if self.k is not None:
      topk_idx = torch.topk(target, self.k, dim=1).indices
      target = torch.gather(target, dim=1, index=topk_idx)
      expected_ranks = torch.gather(expected_ranks, dim=1, index=topk_idx)

    gains = 2**target - 1
    discounts = torch.log2(expected_ranks + 1)
    dcg = (gains / discounts).sum(dim=1)

    ideal_target, _ = torch.sort(target, descending=True, dim=1)
    if self.k is not None:
      ideal_target = ideal_target[:, : self.k]
    ideal_gains = 2**ideal_target - 1
    ideal_discounts = torch.log2(
      torch.arange(1, ideal_target.size(1) + 1, device=preds.device).float() + 1
    )
    idcg = (ideal_gains / ideal_discounts).sum(dim=1)
    ndcg = dcg / (idcg + 1e-10)
    loss = ndcg.mean()
    return 1 - loss


class ExponentialWeightedMSELoss(nn.Module):
  def __init__(self, alpha=1.0):
    super().__init__()
    self.alpha = alpha

  def forward(self, y_pred, y_true):
    y_pred = y_pred.float()
    y_true = y_true.float()
    """
    y	   exp(y)  2^y
    0.0	 1.00	   1.00
    0.2	 1.22	   1.15
    0.4	 1.49	   1.32
    0.6	 1.82	   1.52
    0.8	 2.22	   1.78
    1.0	 2.72	   2.00
    """
    
    weights = torch.exp(self.alpha * y_true)
    loss = weights * (y_true - y_pred) ** 2
    return loss.mean()


# ================== BASE MODEL CLASS ==================
class BaseDistilBertModule(L.LightningModule):
  """Base class with common functionality for DistilBERT-based models"""

  def __init__(self, n_emotions=3, learning_rate=1e-3):
    super().__init__()
    self.n_emotions = n_emotions
    self.learning_rate = learning_rate

    # Initialize tokenizer and model
    self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    self.model = AutoModelForSequenceClassification.from_pretrained(
      "distilbert-base-uncased",
      num_labels=n_emotions,
      problem_type="multi_label_classification",
    ).to(DEVICE)

    # Replace classifier layer
    self.model.classifier = torch.nn.Linear(
      in_features=768, out_features=n_emotions, bias=True
    ).to(DEVICE)

    # Freeze layers except classifier and pre-classifier
    self._freeze_layers()

    # Common components
    self.sigmoid = torch.nn.Sigmoid()
    self.sig_loss = torch.nn.BCEWithLogitsLoss()

    # Initialize metrics
    self._init_metrics()

  def _freeze_layers(self):
    """Freeze all layers except classifier and pre-classifier"""
    for param in self.model.parameters():
      param.requires_grad = False
    for param in self.model.classifier.parameters():
      param.requires_grad = True
    for param in self.model.pre_classifier.parameters():
      param.requires_grad = True

  def _init_metrics(self):
    """Initialize common metrics"""
    self.f1_standard = torchmetrics.classification.MultilabelF1Score(
      num_labels=self.n_emotions, average="macro"
    )
    self.f1_interesting = torchmetrics.classification.MultilabelF1Score(
      num_labels=self.n_emotions, average="macro"
    )
    self.rmse = torchmetrics.regression.MeanSquaredError(squared=False)
    self.ndcg = NDCG(k=None, dist_sync_on_step=False)
    self.expected_ndcg = SoftRankExpectedNDCG(sigma=0.05)
    self.weighted_mse = ExponentialWeightedMSELoss(alpha=1.0)
  def _tokenize_batch(self, texts):
    """Common tokenization logic"""
    tokens = self.tokenizer(
      texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    return {k: v.to(DEVICE) for k, v in tokens.items()}

  def _get_model_output(self, tokens, get_predictions=True) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Get model logits and sigmoid predictions"""
    logits = self.model(
      input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
    )
    if get_predictions:
      predictions = self.sigmoid(logits.logits)
    return logits.logits, predictions if get_predictions else None

  def _compute_f1_metrics(self, predictions, targets):
    """
      Compute F1 metrics with standard and interesting target processing
      
      F1 standard will be a postivie emotion if theres at least one rater who rated it as positive
      F1 interesting will be a positive emotion for every emotion > 0.8, but if there isn't at least one, the highest emotion will be considered positive
    """
    y_bin = (predictions > 0.5).int()
    target_standard = (targets > 0.01).int()
    target_interesting = (targets > 0.8).int()

    # If no emotion is above 0.8, take the highest emotion
    if target_interesting.sum() == 0:
      max_emotion = targets.argmax(dim=1, keepdim=True)
      target_interesting = torch.zeros_like(targets, dtype=torch.int)
      target_interesting.scatter_(1, max_emotion, 1)

    f1_standard = self.f1_standard(y_bin, target_standard)
    f1_interesting = self.f1_interesting(y_bin, target_interesting)
    return f1_standard, f1_interesting

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

  def forward(self, texts):
    """Forward method for inference"""
    tokens = self._tokenize_batch(texts)
    _, predictions = self._get_model_output(tokens)
    return predictions

  def predict_step(self, batch):
    texts = batch
    return self.forward(texts)


# ================== CONCRETE MODEL IMPLEMENTATIONS ==================
class DistilBertFinetune(BaseDistilBertModule):
  """DistilBERT fine-tuned with BCE loss"""

  def training_step(self, batch):
    texts, targets = batch
    tokens = self._tokenize_batch(texts)
    targets = targets.to(DEVICE)
    logits, _ = self._get_model_output(tokens, get_predictions=False)
    loss = self.sig_loss(logits, targets)
    return loss

  def validation_step(self, batch):
    texts, targets = batch
    tokens = self._tokenize_batch(texts)
    targets = targets.to(DEVICE)
    logits, predictions = self._get_model_output(tokens)
    cross_entropy = self.sig_loss(logits, targets)
    rmse = self.rmse(predictions, targets)
    ndcg = self.ndcg(preds=predictions, target=targets)
    expected_ndcg = self.expected_ndcg(preds=predictions, target=targets)
    f1_standard, f1_interesting = self._compute_f1_metrics(predictions, targets)
    weighted_mse = self.weighted_mse(predictions, targets)

    self.log_dict(
      {
        "val_cross_entropy": cross_entropy,
        "val_rmse": rmse,
        "val_ndcg": ndcg,
        "val_expected_ndcg": expected_ndcg,
        "val_f1_standard": f1_standard,
        "val_f1_interesting": f1_interesting,
        "val_weighted_mse": weighted_mse,
      },
      on_step=False,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )

    return cross_entropy

  def test_step(self, batch):
    texts, targets = batch
    tokens = self._tokenize_batch(texts)
    targets = targets.to(DEVICE)
    logits, predictions = self._get_model_output(tokens)
    cross_entropy = self.sig_loss(logits, targets)
    rmse = self.rmse(predictions, targets)
    ndcg = self.ndcg(preds=predictions, target=targets)
    expected_ndcg = self.expected_ndcg(preds=predictions, target=targets)
    f1_standard, f1_interesting = self._compute_f1_metrics(predictions, targets)
    weighted_mse = self.weighted_mse(predictions, targets)

    self.log_dict(
      {
        "test_cross_entropy": cross_entropy,
        "test_rmse": rmse,
        "test_ndcg": ndcg,
        "test_expected_ndcg": expected_ndcg,
        "test_f1_standard": f1_standard,
        "test_f1_interesting": f1_interesting,
        "test_weighted_mse": weighted_mse,
      },
      on_step=False,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )

    return cross_entropy


class DistilBertFinetuneOnDCG(BaseDistilBertModule):
  """DistilBERT fine-tuned with Expected NDCG loss"""

  def training_step(self, batch):
    texts, targets = batch
    tokens = self._tokenize_batch(texts)
    targets = targets.to(DEVICE)
    _, predictions = self._get_model_output(tokens)
    loss = self.expected_ndcg(preds=predictions, target=targets)
    return loss

  def validation_step(self, batch):
    texts, targets = batch
    tokens = self._tokenize_batch(texts)
    targets = targets.to(DEVICE)
    logits, predictions = self._get_model_output(tokens)
    loss = self.expected_ndcg(preds=predictions, target=targets)
    cross_entropy = self.sig_loss(logits, targets)
    rmse = self.rmse(predictions, targets)
    ndcg = self.ndcg(preds=predictions, target=targets)
    f1_standard, f1_interesting = self._compute_f1_metrics(predictions, targets)
    weighted_mse = self.weighted_mse(predictions, targets)

    self.log_dict(
      {
        "val_expected_ndcg": loss,
        "val_cross_entropy": cross_entropy,
        "val_ndcg": ndcg,
        "val_rmse": rmse,
        "val_f1_standard": f1_standard,
        "val_f1_interesting": f1_interesting,
        "val_weighted_mse": weighted_mse,
      },
      on_step=False,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )

    return loss

  def test_step(self, batch):
    texts, targets = batch
    tokens = self._tokenize_batch(texts)
    targets = targets.to(DEVICE)
    logits, predictions = self._get_model_output(tokens)
    cross_entropy = self.sig_loss(logits, targets)
    rmse = self.rmse(predictions, targets)
    ndcg = self.ndcg(preds=predictions, target=targets)
    expected_ndcg = self.expected_ndcg(preds=predictions, target=targets)
    f1_standard, f1_interesting = self._compute_f1_metrics(predictions, targets)
    weighted_mse = self.weighted_mse(predictions, targets)

    self.log_dict(
      {
        "test_cross_entropy": cross_entropy,
        "test_rmse": rmse,
        "test_ndcg": ndcg,
        "test_expected_ndcg": expected_ndcg,
        "test_f1_standard": f1_standard,
        "test_f1_interesting": f1_interesting,
        "test_weighted_mse": weighted_mse,
      },
      on_step=False,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )

    return expected_ndcg


class DistilBertFinetuneOnWeightedMSE(BaseDistilBertModule):
  """DistilBERT fine-tuned with Weighted MSE loss"""

  def training_step(self, batch):
    texts, targets = batch
    tokens = self._tokenize_batch(texts)
    targets = targets.to(DEVICE)
    _, predictions = self._get_model_output(tokens)
    loss = self.weighted_mse(predictions, targets)
    return loss

  def validation_step(self, batch):
    texts, targets = batch
    tokens = self._tokenize_batch(texts)
    targets = targets.to(DEVICE)
    logits, predictions = self._get_model_output(tokens)
    loss = self.weighted_mse(predictions, targets)
    cross_entropy = self.sig_loss(logits, targets)
    rmse = self.rmse(predictions, targets)
    ndcg = self.ndcg(preds=predictions, target=targets)
    expected_ndcg = self.expected_ndcg(preds=predictions, target=targets)
    f1_standard, f1_interesting = self._compute_f1_metrics(predictions, targets)

    self.log_dict(
      {
        "val_weighted_mse": loss,
        "val_cross_entropy": cross_entropy,
        "val_rmse": rmse,
        "val_ndcg": ndcg,
        "val_expected_ndcg": expected_ndcg,
        "val_f1_standard": f1_standard,
        "val_f1_interesting": f1_interesting,
      },
      on_step=False,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )

    return loss

  def test_step(self, batch):
    texts, targets = batch
    tokens = self._tokenize_batch(texts)
    targets = targets.to(DEVICE)
    logits, predictions = self._get_model_output(tokens)
    cross_entropy = self.sig_loss(logits, targets)
    rmse = self.rmse(predictions, targets)
    ndcg = self.ndcg(preds=predictions, target=targets)
    expected_ndcg = self.expected_ndcg(preds=predictions, target=targets)
    f1_standard, f1_interesting = self._compute_f1_metrics(predictions, targets)
    weighted_mse = self.weighted_mse(predictions, targets)

    self.log_dict(
      {
        "test_cross_entropy": cross_entropy,
        "test_rmse": rmse,
        "test_ndcg": ndcg,
        "test_expected_ndcg": expected_ndcg,
        "test_f1_standard": f1_standard,
        "test_f1_interesting": f1_interesting,
        "test_weighted_mse": weighted_mse,
      },
      on_step=False,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )

    return weighted_mse


# ================== LEGACY MODELS ==================
class DummyTokenAndSentiment(nn.Module):
  """Legacy dummy model for comparison"""

  def __init__(self):
    super(DummyTokenAndSentiment, self).__init__()
    self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    self.encoder = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    self.classifier = nn.Linear(self.encoder.config.hidden_size, 3)

  def forward(self, text):
    tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = self.encoder(**tokens)
    cls_rep = outputs.last_hidden_state[:, 0, :]
    logits = self.classifier(cls_rep)
    return torch.sigmoid(logits)
