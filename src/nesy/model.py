# Class and method signatures and docstrings: provided by the course Learning and Reasoning
# (KU Leuven, 2025-2026). Network architectures and all method bodies: Magnus De Maerteleire.

from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn

from nesy.evaluator import Evaluator
from nesy.logic import LogicEngine
from nesy.semantics import Semantics
from nesy.term import Clause


# Define neural network architectures
class NeuralBaseline(nn.Module):
    """
    A small CNN that consumes n input images (stacked as channels) of size 28x28
    and outputs m logits (or probabilities if softmax=True).

    Input shape:
      - (..., n, 28, 28)

    Output shape:
      - (..., m)
    """

    def __init__(self, n_inputs, m_outputs, softmax=True):
        super().__init__()
        self.n_inputs = n_inputs
        self.m_outputs = m_outputs
        self.softmax = softmax

        self.conv1 = nn.Conv2d(
            self.n_inputs, 32, kernel_size=3, padding=1
        )  # 28x28 -> pool -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> pool -> 7x7
        self.full1 = nn.Linear(64 * 7 * 7, 128)  # 7x7 after 2 pooling operations
        self.full2 = nn.Linear(128, 64)
        self.full3 = nn.Linear(64, self.m_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.full1(x))
        x = F.relu(self.full2(x))
        logits = self.full3(x)

        if self.softmax:
            return torch.softmax(logits, dim=-1)
        return logits


class SymbolCNN(nn.Module):
    """
    Small CNN for classifying 28×28 grayscale symbol images into n classes.

    Args:
        n (int): Number of output classes.
        output_encoding (dict, optional): Mapping from symbol string to output index.
            Defaults to {str(i): i for i in range(n)}.
        softmax (bool, optional): If True, apply softmax to the output logits.
    """

    def __init__(self, n, output_encoding=None, softmax=True):
        super().__init__()
        self.n = n
        self.output_encoding = output_encoding or {str(i): i for i in range(n)}
        self.softmax = softmax

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.full1 = nn.Linear(64 * 7 * 7, 128)
        self.full2 = nn.Linear(128, 64)
        self.full3 = nn.Linear(64, n)

    def forward(self, x):
        """
        Accepts (B,1,28,28) or (1,28,28) or flattened (B,784).
        Returns (B,n) probabilities by default, or logits if softmax=False.
        """

        # Normalize to (B,1,28,28)
        if x.ndim == 2 and x.shape[-1] == 784:
            x = x.view(-1, 1, 28, 28)
        elif x.ndim == 3 and x.shape[-2:] == (28, 28):
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            pass
        else:
            # best-effort reshape
            x = x.view(-1, 1, 28, 28)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.full1(x))
        x = F.relu(self.full2(x))
        logits = self.full3(x)

        return torch.softmax(logits, dim=-1) if self.softmax else logits


class FeatureExtractor(nn.Module):
    """
    CNN feature extractor that maps 28×28 grayscale images to a feature vector
    of length size_output_vector.

    Args:
        size_output_vector (int, optional): Length of the output feature vector.
            Defaults to 128.
    """

    def __init__(self, size_output_vector=128):
        super().__init__()
        self.size_output_vector = size_output_vector
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # keep output dimension configurable
        self.fc1 = nn.Linear(64 * 7 * 7, self.size_output_vector)

    def forward(self, x):
        """
        Accepts:
          - (B,1,28,28)
          - (B,784)
          - (...,1,28,28) or (...,784)

        Returns:
          - (..., size_output_vector)
        """
        orig_shape = x.shape

        # Flatten leading dims into batch
        if x.ndim >= 3 and orig_shape[-2:] == (28, 28):
            # (..., C, 28, 28) where C should be 1
            x = x.view(-1, 1, 28, 28)
            leading = orig_shape[:-3]
        elif x.ndim >= 2 and orig_shape[-1] == 784:
            x = x.view(-1, 1, 28, 28)
            leading = orig_shape[:-1]
        else:
            # best-effort
            x = x.view(-1, 1, 28, 28)
            leading = orig_shape[:-3] if x.ndim >= 4 else orig_shape[:-1]

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        feats = F.relu(self.fc1(x))

        return feats.view(*leading, self.size_output_vector)


class NetworkHeadLarge(nn.Module):
    """
    MLP head that consumes features from multiple grayscale images and outputs n logits.

    The input consists of `nr_images` grayscale images (28×28), whose features are
    extracted and concatenated before being passed through an MLP.

    Args:
        feature_extractor (nn.Module): Module that maps each 28×28 grayscale image
            to a feature vector of length `feature_extractor.size_output_vector`.
        nr_images (int): Number of input grayscale images whose features are concatenated.
        n (int): Number of output classes.
        output_encoding (dict, optional): Mapping from symbol string to output index.
            Defaults to {str(i): i for i in range(n)}.
    """

    def __init__(self, feature_extractor, nr_images, n, output_encoding=None, softmax=True):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.nr_images = nr_images
        self.n = n
        self.output_encoding = output_encoding or {str(i): i for i in range(n)}
        self.softmax = softmax

        self.full1 = nn.Linear(self.feature_extractor.size_output_vector * self.nr_images, 128)
        self.full2 = nn.Linear(128, 64)
        self.full3 = nn.Linear(64, n)

    def forward(self, x):
        """
        Accepts:
          - (B, nr_images, 1, 28, 28)  (batched)
          - (nr_images, 1, 28, 28)     (single example)

        Returns:
          - (B, n) probabilities (default) or logits if softmax=False
        """
        if x.ndim == 4:
            # (nr_images,1,28,28) -> (1,nr_images,1,28,28)
            x = x.unsqueeze(0)

        if x.ndim != 5:
            raise ValueError(f"Expected input with 4 or 5 dims, got {x.shape}")

        b, k, c, h, w = x.shape
        if k != self.nr_images:
            raise ValueError(f"Expected nr_images={self.nr_images} but got {k}")

        # Extract features for all images in one pass:
        feats = self.feature_extractor(x.view(b * k, c, h, w))  # (B*K, F)
        feats = feats.view(b, k * self.feature_extractor.size_output_vector)  # (B, K*F)

        x = F.relu(self.full1(feats))
        x = F.relu(self.full2(x))
        logits = self.full3(x)

        return torch.softmax(logits, dim=-1) if self.softmax else logits


class NetworkHeadSmall(nn.Module):
    """
    A smaller head that consumes features from multiple grayscale images and outputs n logits.

    The input consists of `nr_images` grayscale images (28×28), whose features are
    extracted and concatenated before being passed through a single linear layer.

    Args:
        feature_extractor (nn.Module): Module that maps each 28×28 grayscale image
            to a feature vector of length `feature_extractor.size_output_vector`.
        nr_images (int): Number of input grayscale images whose features are concatenated.
        n (int): Number of output classes.
        output_encoding (dict, optional): Mapping from symbol string to output index.
            Defaults to {str(i): i for i in range(n)}.
    """

    def __init__(self, feature_extractor, nr_images, n, output_encoding=None, softmax=True):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.nr_images = nr_images
        self.n = n
        self.output_encoding = output_encoding or {str(i): i for i in range(n)}
        self.softmax = softmax
        self.full1 = nn.Linear(self.feature_extractor.size_output_vector * self.nr_images, n)

    def forward(self, x):
        """
        Accepts:
          - (B, nr_images, 1, 28, 28)  (batched)
          - (nr_images, 1, 28, 28)     (single example)

        Returns:
          - (B, n) probabilities (default) or logits if softmax=False
        """
        if x.ndim == 4:
            x = x.unsqueeze(0)

        if x.ndim != 5:
            raise ValueError(f"Expected input with 4 or 5 dims, got {x.shape}")

        b, k, c, h, w = x.shape
        if k != self.nr_images:
            raise ValueError(f"Expected nr_images={self.nr_images} but got {k}")

        feats = self.feature_extractor(x.view(b * k, c, h, w))  # (B*K, F)
        feats = feats.view(b, k * self.feature_extractor.size_output_vector)  # (B, K*F)

        logits = self.full1(feats)
        return torch.softmax(logits, dim=-1) if self.softmax else logits


# Define NeSy models and baseline as lightning modules
class NeuralModel(pl.LightningModule):
    """
    Lightning wrapper for neural baseline.
    """

    def __init__(self, neural_network: nn.Module = None, learning_rate=0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images):
        """
        Expects images: torch.Tensor as batched input to the neural network
        """
        return self.neural_network(images)

    def _step(self, batch, stage):
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()

        if stage == "train":
            return loss

        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class NeSyModelLogicInArchitecture(pl.LightningModule):
    """
    Lightning module for the neurosymbolic logic in the architecture model.
    """

    def __init__(
        self,
        program: List[Clause],
        neural_predicates: torch.nn.ModuleDict,
        logic_engine: LogicEngine,
        label_semantics: Semantics,
        learning_rate=0.001,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.neural_predicates = neural_predicates
        self.logic_engine = logic_engine
        self.label_semantics = label_semantics
        self.program = tuple(program)
        self.learning_rate = learning_rate
        self.bce = torch.nn.BCELoss()
        self.evaluator = Evaluator(
            neural_predicates=neural_predicates, label_semantics=label_semantics
        )
        self.and_or_trees = self.logic_engine.reason(self.program, None)

    def forward(self, tensor_sources, queries):
        """
        Expects:
            tensor_sources: Dict[str, torch.Tensor] for logic evaluation
            queries: List[Term] for logic evaluation
        """
        return self.evaluator.evaluate(tensor_sources, self.and_or_trees, queries)

    def training_step(self, batch, batch_idx):
        tensor_sources, queries, query_probabilities = batch

        y_pred = torch.clamp(self.forward(tensor_sources, queries), 0.0, 1.0).squeeze()
        y_true = query_probabilities.float().squeeze()
        loss = self.bce(y_pred, y_true)
        return loss

    def validation_step(self, batch, batch_idx):
        tensor_sources, queries_lists, targets = batch
        b = next(iter(tensor_sources.values())).shape[0]

        # Evaluate queries per sample to avoid inefficient expansion
        with torch.no_grad():
            scores_list = []
            for i in range(b):
                # Get queries and tensor sources for this sample
                sample_queries = queries_lists[i]
                sample_sources = {k: v[i : i + 1] for k, v in tensor_sources.items()}
                # Evaluate all queries for this sample at once
                sample_scores = self.forward(sample_sources, sample_queries).squeeze(0)  # (n,)
                scores_list.append(sample_scores)

            scores = torch.stack(scores_list, dim=0)  # (b, n)
            preds = scores.argmax(dim=-1).cpu().numpy()

        y_true = targets.cpu().numpy()
        acc = accuracy_score(y_true, preds)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        tensor_sources, queries_lists, targets = batch
        b = next(iter(tensor_sources.values())).shape[0]

        # Evaluate queries per sample to avoid inefficient expansion
        with torch.no_grad():
            scores_list = []
            for i in range(b):
                # Get queries and tensor sources for this sample
                sample_queries = queries_lists[i]
                sample_sources = {k: v[i : i + 1] for k, v in tensor_sources.items()}
                # Evaluate all queries for this sample at once
                sample_scores = self.forward(sample_sources, sample_queries).squeeze(0)  # (n,)
                scores_list.append(sample_scores)

            scores = torch.stack(scores_list, dim=0)  # (b, n)
            preds = scores.argmax(dim=-1).cpu().numpy()

        y_true = targets.cpu().numpy()
        acc = accuracy_score(y_true, preds)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class NeSyModelLogicInLoss(pl.LightningModule):
    """
    Lightning module for the neurosymbolic logic in the loss model.
    """

    def __init__(
        self,
        program: List[Clause],
        neural_predicates: torch.nn.ModuleDict,
        logic_engine: LogicEngine,
        label_semantics: Semantics,
        learning_rate=0.001,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.program = tuple(program)
        self.neural_predicates = neural_predicates
        self.logic_engine = logic_engine
        self.label_semantics = label_semantics
        self.learning_rate = learning_rate
        self.bce = torch.nn.BCELoss()
        self.evaluator = Evaluator(
            neural_predicates=neural_predicates, label_semantics=label_semantics
        )
        self.criterion = nn.CrossEntropyLoss()

        self.and_or_trees = self.logic_engine.reason(self.program, None)

    def forward(self, images):
        """
        Expects images: torch.Tensor as batched input to the neural network

        Returns a dict with batched outputs for known heads.
        """
        out: Dict[str, torch.Tensor] = {}
        # Only compute heads that can consume the full image-tuple directly (typically result_head).
        if "result_head" in self.neural_predicates:
            out["result_head"] = self.neural_predicates["result_head"](images)  # probs by default
        return out

    def training_step(self, batch, batch_idx):
        tensor_sources, queries, query_probabilities, images, targets = batch

        # Logic constraint satisfaction probability
        y_logic = self.evaluator.evaluate(tensor_sources, self.and_or_trees, queries)
        y_logic = torch.clamp(y_logic, 0.0, 1.0).squeeze()
        logic_loss = self.bce(y_logic, query_probabilities.float().squeeze())

        # Supervised head(s)
        if isinstance(images, list):
            # Convert list of tensors to batched tensor: (B, nr_images, 1, 28, 28)
            images_batched = torch.stack(images, dim=0)
        else:
            images_batched = images

        ce_loss = 0.0
        for head, y_true in targets.items():
            # Get the head module and temporarily set softmax=False for efficiency
            head_module = self.neural_predicates[head]
            original_softmax = getattr(head_module, "softmax", True)

            if original_softmax:
                head_module.softmax = False
                logits = head_module(images_batched)
                head_module.softmax = original_softmax  # Restore softmax
            else:
                logits = head_module(images_batched)

            ce_loss = ce_loss + self.criterion(logits, y_true)

        loss = ce_loss + logic_loss
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Handle both list of tensors and batched tensor
        if isinstance(images, list):
            images_batched = torch.stack(images, dim=0)
        else:
            images_batched = images

        accs = []
        for head, y_true in targets.items():
            logits = self.neural_predicates[head](images_batched)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy().flatten()
            accs.append(accuracy_score(y_true_np, preds))

        acc = float(sum(accs) / max(1, len(accs)))
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        images, targets = batch

        # Handle both list of tensors and batched tensor
        if isinstance(images, list):
            images_batched = torch.stack(images, dim=0)
        else:
            images_batched = images

        accs = []
        for head, y_true in targets.items():
            logits = self.neural_predicates[head](images_batched)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy().flatten()
            accs.append(accuracy_score(y_true_np, preds))

        acc = float(sum(accs) / max(1, len(accs)))
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
