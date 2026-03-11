"""IB-LoRA: Information Bottleneck Low-Rank Adaptation.

This module implements IB-LoRA, a parameter-efficient fine-tuning method that
extends LoRA with an information bottleneck to suppress spurious token correlations.

Mathematical background
-----------------------
Standard LoRA adds a low-rank update to frozen weights:
    h = W₀x + ΔWx = W₀x + BAx

IB-LoRA inserts a variational bottleneck at the rank-dimensional intermediate
representation produced by A:

    μ(x)  = Ax           (LoRA intermediate, shape: [rank])
    σ      = exp(log_σ)  (learned per-rank noise scale)
    Z      = μ + σ·ε,   ε ~ N(0, I)   [training]
    Z      = μ            [inference]
    h      = scaling · BZ + W₀x

The training objective is:
    L = E[log p(Y|Z)] - β · KL[q(Z|X) || r(Z)]
      = task_loss - β · KL_loss

where the closed-form KL divergence is:
    KL = -0.5 · Σ(1 + log σ² - μ² - σ²)

β controls the robustness–performance tradeoff:
    β = 0  →  standard LoRA (no compression)
    β → ∞  →  maximum compression, spurious signals suppressed

Reference: "IB-LoRA: Robust Fine-Tuning via Information Bottleneck"
"""

import math
import os
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Core adapter layer
# ---------------------------------------------------------------------------


class IBLoRALayer(nn.Module):
    """Low-rank adapter with an Information Bottleneck bottleneck.

    The layer implements:
        μ  = Ax              (lora_A projection to rank-dim space)
        Z  = μ + σ·ε        (stochastic bottleneck, training only)
        out = scaling · BZ  (lora_B projects back to out_features)

    KL divergence against N(0, I) prior is stored in ``kl_loss`` after every
    forward call and should be added to the task loss by the training loop.

    Parameters
    ----------
    in_features:
        Input dimensionality (matches the frozen Linear layer's in_features).
    out_features:
        Output dimensionality (matches the frozen Linear layer's out_features).
    rank:
        Bottleneck rank ``r``.  Smaller → more compression.
    alpha:
        LoRA scaling numerator.  ``scaling = alpha / rank``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling: float = alpha / rank

        # LoRA matrices -------------------------------------------------
        # lora_A: projects input → rank-dimensional space (μ producer)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        # lora_B: projects bottleneck → output space
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Bottleneck noise scale ----------------------------------------
        # log_sigma of shape (rank,) — one per rank dimension.
        # Initialised to 0 → σ = 1 at the start of training.
        self.log_sigma = nn.Parameter(torch.zeros(rank))

        # Stores the KL divergence computed during the last forward pass.
        self.kl_loss: Tensor = torch.tensor(0.0)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise weights following standard LoRA conventions.

        lora_A: kaiming uniform (same as nn.Linear default)
        lora_B: zeros  →  adapter output is zero at initialisation so
                          training starts from the pretrained checkpoint.
        log_sigma: zeros  →  σ = 1, KL ≈ 0.5·rank at init but decreases
                             rapidly as the model learns a useful bottleneck.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.zeros_(self.log_sigma)

    # ------------------------------------------------------------------
    # Reparameterisation trick
    # ------------------------------------------------------------------

    def reparameterize(self, mu: Tensor) -> Tensor:
        """Apply the reparameterisation trick to obtain the bottleneck sample Z.

        During **training** injects Gaussian noise scaled by σ = exp(log_σ):
            Z = μ + σ · ε,  ε ~ N(0, I)

        During **evaluation** returns the deterministic mean:
            Z = μ

        This preserves the standard expectation-under-noise semantics during
        training while making inference fully deterministic and reproducible.

        Parameters
        ----------
        mu:
            Mean tensor produced by lora_A, shape (..., rank).

        Returns
        -------
        Tensor of the same shape as ``mu``.
        """
        if self.training:
            sigma = torch.exp(self.log_sigma)          # (rank,)
            eps = torch.randn_like(mu)                 # (..., rank)
            return mu + sigma * eps
        return mu

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    def compute_kl_loss(self, mu: Tensor) -> Tensor:
        """Compute the closed-form KL divergence KL[q(Z|X) || N(0, I)].

        For a diagonal Gaussian q = N(μ, σ²I) and prior r = N(0, I):
            KL = -0.5 · Σ_d (1 + log σ²_d - μ²_d - σ²_d)

        Note that μ here is the *batch-averaged* μ (mean over the batch and
        sequence dimensions) to produce a scalar KL that is comparable across
        different batch sizes.  σ is shared across all positions (it is a
        per-rank learned scalar), so it is summed over rank dimensions only.

        Parameters
        ----------
        mu:
            Mean activations of shape (batch, seq_len, rank) or (batch, rank).

        Returns
        -------
        Scalar tensor (0-dim).
        """
        sigma_sq = torch.exp(2.0 * self.log_sigma)  # (rank,)

        # Per-position KL: -0.5*(1 + log σ² - μ² - σ²) averaged over batch/seq
        # mu_sq_mean has shape (rank,) after mean over leading dims
        mu_sq_mean = (mu ** 2).mean(dim=list(range(mu.dim() - 1)))  # (rank,)

        kl = -0.5 * (1.0 + 2.0 * self.log_sigma - mu_sq_mean - sigma_sq)
        return kl.sum()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Compute the IB-LoRA adapter output and store the KL divergence.

        Steps
        -----
        1. Project input to rank space:      μ = x @ lora_A.T
        2. Bottleneck sample:                Z = reparameterize(μ)
        3. Project back to output space:     out = Z @ lora_B.T · scaling
        4. Compute and store KL divergence.

        Parameters
        ----------
        x:
            Input tensor, shape (batch, seq_len, in_features) or
            (batch, in_features).

        Returns
        -------
        Adapter output, same leading shape as ``x`` but last dim = out_features.
        """
        mu = F.linear(x, self.lora_A)          # (..., rank)
        z = self.reparameterize(mu)             # (..., rank)
        out = F.linear(z, self.lora_B) * self.scaling  # (..., out_features)

        # Store KL for retrieval in the training loop
        self.kl_loss = self.compute_kl_loss(mu)

        return out


# ---------------------------------------------------------------------------
# Wrapper: frozen pretrained Linear + IB-LoRA adapter
# ---------------------------------------------------------------------------


class IBLoRALinear(nn.Module):
    """Wraps a frozen pretrained ``nn.Linear`` with an ``IBLoRALayer`` adapter.

    Forward pass:
        h = pretrained(x) + ib_lora(x)

    The pretrained weights are frozen; only lora_A, lora_B, and log_sigma are
    updated during training.

    Parameters
    ----------
    linear:
        The pretrained ``nn.Linear`` layer to wrap.  Its weights are frozen
        in-place.
    rank:
        Adapter bottleneck rank.
    alpha:
        LoRA scaling factor.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.pretrained = linear
        # Freeze the pretrained weights
        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.ib_lora = IBLoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
        )

    # ------------------------------------------------------------------

    @property
    def kl_loss(self) -> Tensor:
        """KL divergence from the last forward pass of the IB-LoRA adapter."""
        return self.ib_lora.kl_loss

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Compute pretrained output plus adapter output.

        Parameters
        ----------
        x:
        Input tensor.

        Returns
        -------
        Combined output tensor.
        """
        return self.pretrained(x) + self.ib_lora(x)

    # ------------------------------------------------------------------

    def merge_and_unload(self) -> nn.Linear:
        """Merge the learned LoRA weights into the pretrained weights.

        Produces a standard ``nn.Linear`` with no IB-LoRA overhead, suitable
        for efficient inference deployment.  The merge uses the *deterministic*
        (mean) path: ΔW = scaling · B @ A (no noise).

        Returns
        -------
        ``nn.Linear`` whose weight = W₀ + scaling · B @ A.
        """
        # Compute the low-rank update: ΔW = scaling * B @ A
        # lora_A: (rank, in_features), lora_B: (out_features, rank)
        delta_w = (
            self.ib_lora.scaling
            * self.ib_lora.lora_B.data
            @ self.ib_lora.lora_A.data
        )  # (out_features, in_features)

        merged = nn.Linear(
            self.pretrained.in_features,
            self.pretrained.out_features,
            bias=self.pretrained.bias is not None,
        )
        merged.weight = nn.Parameter(
            (self.pretrained.weight + delta_w).detach().clone()
        )
        if self.pretrained.bias is not None:
            merged.bias = nn.Parameter(self.pretrained.bias.detach().clone())

        return merged


# ---------------------------------------------------------------------------
# Model-level utilities
# ---------------------------------------------------------------------------


def apply_ib_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Recursively replace target ``nn.Linear`` layers with ``IBLoRALinear``.

    Parameters
    ----------
    model:
        The pretrained model to adapt.
    rank:
        LoRA rank for all inserted adapters.
    alpha:
        LoRA alpha scaling for all inserted adapters.
    target_modules:
        List of module *name substrings* to target (e.g. ``["query", "value"]``).
        A layer is replaced if any substring appears in its full dotted name.
        If ``None``, **all** ``nn.Linear`` layers are replaced.

    Returns
    -------
    The model with replaced modules (modified in-place and returned).
    """
    # Collect (name, parent_module, attr_name) triples for matching linears
    replacements: List[Tuple[str, nn.Module, str]] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None:
            if not any(t in full_name for t in target_modules):
                continue
        # Navigate to parent
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        replacements.append((full_name, parent, parts[-1]))

    for full_name, parent, attr in replacements:
        original: nn.Linear = getattr(parent, attr)
        setattr(parent, attr, IBLoRALinear(original, rank=rank, alpha=alpha))

    return model


def collect_kl_loss(model: nn.Module) -> Tensor:
    """Sum all KL losses from every ``IBLoRALinear`` in the model.

    Should be called *after* the forward pass so that all adapters have
    populated their ``kl_loss`` field.

    Parameters
    ----------
    model:
        A model that has been wrapped with ``apply_ib_lora``.

    Returns
    -------
    Scalar tensor = Σ kl_loss over all ``IBLoRALinear`` layers.
    """
    total = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, IBLoRALinear):
            total = total + module.kl_loss
    return total


def get_trainable_parameters(model: nn.Module) -> Dict[str, object]:
    """Return counts and fraction of trainable parameters.

    Parameters
    ----------
    model:
        Any ``nn.Module``, typically after ``apply_ib_lora``.

    Returns
    -------
    dict with keys:
        ``trainable``  – int, number of trainable parameters.
        ``total``      – int, total number of parameters.
        ``fraction``   – float, trainable / total.
        ``percent``    – str, human-readable percentage string.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    fraction = trainable / total if total > 0 else 0.0
    return {
        "trainable": trainable,
        "total": total,
        "fraction": fraction,
        "percent": f"{100.0 * fraction:.4f}%",
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def train_step(
    model: nn.Module,
    batch: Dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    beta: float,
) -> Dict[str, float]:
    """Execute one IB-LoRA training step.

    Objective:
        L = task_loss + β · kl_loss
        (written as subtraction in the paper since task_loss = -log p(Y|Z))

    The ``beta`` argument is intentionally decoupled from the per-layer beta so
    that it can be swept during ablations (e.g. via ``BetaScheduler``) without
    reinitialising the model.

    When ``beta = 0`` the KL term vanishes and the objective is identical to
    standard LoRA (cross-entropy only).

    Parameters
    ----------
    model:
        Model with IB-LoRA adapters applied via ``apply_ib_lora``.
    batch:
        Dictionary containing at least ``"input_ids"`` and ``"labels"``.
        May optionally contain ``"attention_mask"``.  Labels should use -100
        for positions that should not contribute to the loss (HuggingFace
        convention).
    optimizer:
        An already-constructed optimiser (e.g. AdamW) over trainable params.
    beta:
        Current IB weight.  Pass 0 to recover standard LoRA exactly.

    Returns
    -------
    dict with keys ``task_loss``, ``kl_loss``, ``total_loss`` (all floats).
    """
    model.train()
    optimizer.zero_grad()

    input_ids: Tensor = batch["input_ids"]
    labels: Tensor = batch["labels"]
    attention_mask: Optional[Tensor] = batch.get("attention_mask", None)

    # Forward pass
    kwargs: Dict[str, Tensor] = {"input_ids": input_ids, "labels": labels}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    outputs = model(**kwargs)
    task_loss: Tensor = outputs.loss

    # Collect KL after forward (all adapters have populated kl_loss)
    kl: Tensor = collect_kl_loss(model)

    total_loss: Tensor = task_loss + beta * kl

    total_loss.backward()
    optimizer.step()

    return {
        "task_loss": task_loss.item(),
        "kl_loss": kl.item(),
        "total_loss": total_loss.item(),
    }


# ---------------------------------------------------------------------------
# Beta warmup scheduler
# ---------------------------------------------------------------------------


class BetaScheduler:
    """Warm up β from 0 to ``target_beta`` over ``warmup_steps`` steps.

    Starting with a large β destabilises training (the KL penalty dominates
    before the model has learnt anything useful).  This scheduler linearly or
    cosine-anneals β from 0 → target_beta so the bottleneck is introduced
    gradually.

    Parameters
    ----------
    target_beta:
        Final value of β after warmup is complete.
    warmup_steps:
        Number of steps over which to ramp β.
    schedule:
        ``"linear"`` or ``"cosine"``.

    Examples
    --------
    >>> scheduler = BetaScheduler(target_beta=0.01, warmup_steps=1000)
    >>> for step, batch in enumerate(dataloader):
    ...     beta = scheduler.get_beta(step)
    ...     losses = train_step(model, batch, optimizer, beta=beta)
    """

    def __init__(
        self,
        target_beta: float,
        warmup_steps: int,
        schedule: str = "linear",
    ) -> None:
        if schedule not in ("linear", "cosine"):
            raise ValueError(f"schedule must be 'linear' or 'cosine', got {schedule!r}")
        self.target_beta = target_beta
        self.warmup_steps = warmup_steps
        self.schedule = schedule

    def get_beta(self, step: int) -> float:
        """Return the β value for the given training step.

        Parameters
        ----------
        step:
            Current training step (0-indexed).

        Returns
        -------
        β value in [0, target_beta].
        """
        if self.warmup_steps <= 0:
            return self.target_beta

        if step >= self.warmup_steps:
            return self.target_beta

        progress = step / self.warmup_steps  # in [0, 1)

        if self.schedule == "linear":
            return self.target_beta * progress
        else:  # cosine
            # cosine schedule: 0 at step=0, target at step=warmup_steps
            return self.target_beta * 0.5 * (1.0 - math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Save / load utilities
# ---------------------------------------------------------------------------


def save_ib_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the trainable IB-LoRA parameters to disk.

    Saves lora_A, lora_B, and log_sigma for every ``IBLoRALinear`` module.
    Pretrained (frozen) weights are not saved, keeping checkpoint size small.

    Parameters
    ----------
    model:
        Model with IB-LoRA adapters.
    path:
        File path for the ``.pt`` checkpoint (e.g. ``"ib_lora_weights.pt"``).
    """
    state: Dict[str, Tensor] = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(state, path)


def load_ib_lora_weights(model: nn.Module, path: str) -> None:
    """Load IB-LoRA parameters from a checkpoint saved by ``save_ib_lora_weights``.

    Parameters
    ----------
    model:
        Model with the **same** IB-LoRA architecture as when it was saved.
    path:
        Path to the ``.pt`` checkpoint.
    """
    state: Dict[str, Tensor] = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_trainable = [
        k for k in unexpected if k in {n for n, _ in model.named_parameters()}
    ]
    if unexpected_trainable:
        raise RuntimeError(
            f"Unexpected trainable keys in checkpoint: {unexpected_trainable}"
        )


# ---------------------------------------------------------------------------
# Demo / main block
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """Demonstrate applying IB-LoRA to bert-base-uncased."""
    try:
        from transformers import AutoModel
    except ImportError:
        raise SystemExit("transformers is required for the demo: pip install transformers")

    print("Loading bert-base-uncased …")
    model = AutoModel.from_pretrained("bert-base-uncased")

    before = get_trainable_parameters(model)
    print(f"Before IB-LoRA — trainable: {before['trainable']:,}  "
          f"({before['percent']})")

    apply_ib_lora(
        model,
        rank=8,
        alpha=16.0,
        target_modules=["query", "value"],
    )

    after = get_trainable_parameters(model)
    print(f"After  IB-LoRA — trainable: {after['trainable']:,}  "
          f"({after['percent']})  /  total: {after['total']:,}")

    # Count how many IBLoRALinear layers were inserted
    n_adapters = sum(1 for m in model.modules() if isinstance(m, IBLoRALinear))
    print(f"Inserted {n_adapters} IBLoRALinear adapters "
          f"(target_modules=[\"query\", \"value\"])")
