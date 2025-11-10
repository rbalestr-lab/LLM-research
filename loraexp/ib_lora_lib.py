import numpy as np
import torch
import torch.nn as nn
import math
from typing import Any, Optional
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.integrations import dequantize_bnb_weight, gather_params_ctx
from peft.utils.other import transpose
from transformers.pytorch_utils import Conv1D
from peft.tuners.lora.layer import LoraLayer
from transformers import Trainer


# torch implementation
class OnlineCovarianceEstimatorWelford(nn.Module):
    """
    Online covariance with batched Welford
    Lazy-initialize buffers on first update if num_features is None.
    """
    def __init__(self, num_features: int | None = None, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = float(eps)

        # Counters and stats as buffers (persist & move with the model)
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", None)     # [d]
        self.register_buffer("scatter", None)  # [d, d]

        # If d is known, allocate now; otherwise lazy-init on first batch
        if num_features is not None:
            d = int(num_features)
            self.mean = torch.zeros(d, dtype=torch.float32)
            self.scatter = torch.zeros(d, d, dtype=torch.float32)

    def _lazy_init(self, x: torch.Tensor):
        if self.mean is not None:
            return
        d = x.shape[-1]
        device = x.device
        self.mean = torch.zeros(d, dtype=torch.float32, device=device)
        self.scatter = torch.zeros(d, d, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        """
        batch: [..., d] or [N, d], flatten all leading dims into N
        """
        if batch is None or batch.numel() == 0:
            return

        # Flatten to [N, d]
        if batch.dim() > 2:
            batch = batch.view(-1, batch.shape[-1])

        x = batch.detach().to(torch.float32)
        N_b = x.shape[0]
        if N_b == 0:
            return

        # Lazy allocate buffers if needed
        self._lazy_init(x)

        # Batch stats computed around the *batch mean*
        mean_b = x.mean(dim=0)                      # [d]
        xc = x - mean_b                             # [N, d]
        S_b = xc.T @ xc                             # [d, d]  (scatter within the batch)

        n_old = int(self.count.item())
        if n_old == 0:
            # Initialize from first batch
            self.mean.copy_(mean_b)
            self.scatter.copy_(S_b)
            self.count.fill_(N_b)
            return

        n_new = n_old + N_b
        dmean = (mean_b - self.mean)                # [d]
        # S_total = S_old + S_batch + n_old * N_b / n_new * >dmean, dmean<
        self.scatter += S_b + (n_old * N_b / n_new) * torch.outer(dmean, dmean)
        self.mean += (N_b / n_new) * dmean
        self.count.fill_(n_new)


    @torch.no_grad()
    def get_covariance(self, unbiased: bool = False) -> torch.Tensor:
        """
        Returns the covariance matrix
        small ridge on the diagonal for PSD
        unbiased uses bessels correction
        """
        if self.mean is None or self.scatter is None:
            raise ValueError("No data yet (tracker not initialized).")
        n = int(self.count.item())
        if unbiased:
            if n <= 1:
                raise ValueError("Need at least 2 samples for unbiased covariance.")
            C = self.scatter / (n - 1.0)
        else:
            if n <= 0:
                raise ValueError("No data yet.")
            C = self.scatter / float(n)

        # PSD ridge
        C = C.clone()
        C.diagonal().add_(self.eps)
        return C



def ib_regularizer_AB(A: torch.Tensor, B: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:

    # symmetrize the covariance matrix
    Sigma = 0.5 * (Sigma + Sigma.T)

    # r×r core
    M = A @ Sigma @ A.T                         # [r, r]
    # middle = (W Σ) W^T = B M B^T
    middle = B @ M @ B.T                        # [d_out, d_out]

    # Solve middle * X = (W Σ) instead of explicit inverse
    # RHS: (W Σ) = B A Σ  (shape [d_out, d_in])
    RHS = B @ A @ Sigma                         # [d_out, d_in]

    # solve for inverse, lstsq for pinv is faster
    # X = torch.linalg.solve(middle, RHS)         # [d_out, d_in]
    X = torch.linalg.lstsq(middle, RHS).solution

    # Σ W^T (WΣW^T)^{-1} W Σ = Σ @ (W^T X)
    BtX = B.T @ X                                # [d_out, d_out]
    AtBtX = A.T @ BtX                            # [d_in, d_out]
    correction = Sigma @ AtBtX                   # [d_in, d_out]

    # symmetrize
    correction = 0.5 * (correction + correction.T)

    denom = Sigma - correction

    # add a small ridge to the denom for PSD
    denom = denom + 1e-6 * torch.eye(denom.shape[0], device=denom.device, dtype=denom.dtype)

    # Check for non-finite/non-positive logdet
    sign_num, logdet_num = torch.slogdet(Sigma)
    sign_den, logdet_den = torch.slogdet(denom)
    if (sign_num <= 0) or (sign_den <= 0) or not torch.isfinite(logdet_num) or not torch.isfinite(logdet_den):
        print("Encountered non-finite/non-positive logdet, returning 0")
        print(f"Sigma: {Sigma}")
        print(f"denom: {denom}")
        print(f"sign_num: {sign_num}")
        print(f"sign_den: {sign_den}")
        print(f"logdet_num: {logdet_num}")
        print(f"logdet_den: {logdet_den}")
        return torch.zeros((), device=Sigma.device, dtype=Sigma.dtype)

    return 0.5 * (logdet_num - logdet_den)


def _is_lora_wrapper(m) -> bool:
    return hasattr(m, "lora_A") and hasattr(m, "lora_B")

def attach_cov_hooks_to_lora_sites(
    model,
    tracker_ctor=lambda: OnlineCovarianceEstimatorWelford(num_features=None, eps=1e-5),
):
    """
    Attach forward_pre_hook to each LoRA base layer to collect X and update tracker.
    Stores tracker at base_layer.cov_tracker (buffers move & save with the model).
    Returns list of hook handles
    """
    handles = []

    @torch.no_grad()
    def pre_hook(mod, inputs):
        (x,) = inputs  # Linear-like signature
        x_flat = x if x.dim() == 2 else x.view(-1, x.shape[-1])
        x_flat = x_flat.detach().to(torch.float32)
        if not hasattr(mod, "cov_tracker") or mod.cov_tracker is None:
            mod.cov_tracker = tracker_ctor().to(x_flat.device)
        mod.cov_tracker.update(x_flat)

    for name, m in model.named_modules():
        if _is_lora_wrapper(m):
            base = getattr(m, "base_layer", m)
            if not hasattr(base, "_cov_hook_attached"):
                base._module_name = name 
                h = base.register_forward_pre_hook(pre_hook, with_kwargs=False)
                base._cov_hook_attached = True
                handles.append(h)

    return handles

def ib_penalty_from_ab(model, include_scaling: bool = False):
    """
    Iterate LoRA modules, pull Σ from each base_layer.cov_tracker, and sum ib_regularizer_AB(A,B,Σ).
    """
    # default device/dtype from model
    try:
        p0 = next(model.parameters())
        device = p0.device
        dtype = p0.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32

    reg = torch.zeros((), device=device, dtype=dtype)
    per_mod_reg = {} # track the regularizer for each module

    for m in model.modules():
        if not _is_lora_wrapper(m):
            continue
        base = getattr(m, "base_layer", m)
        tracker = getattr(base, "cov_tracker", None)
        if tracker is None or tracker.mean is None:
            continue  # not warmed up yet

        Sigma = tracker.get_covariance().to(device=device, dtype=dtype)
        mod_name = getattr(base, "_module_name", "<unnamed>")


        # PEFT keeps per-adapter A/B in dicts; aggregate across all adapter keys
        for key in m.lora_A.keys():
            A = m.lora_A[key].weight  # [r, d_in]
            B = m.lora_B[key].weight  # [d_out, r]
            reg = reg + ib_regularizer_AB(A, B, Sigma)

            reg_key = f"{mod_name}:{key}".replace(".", "_")
            per_mod_reg[reg_key] = reg

    return reg, per_mod_reg

class IBLoraTrainer(Trainer):
    """
    Trainer that attaches input-capture hooks for LoRA sites, and 
    adds regularizer to the loss using cached covariances.
    """

    def __init__(self, *args, ib_lambda: float = 1e-4, include_scaling: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ib_lambda = float(ib_lambda)
        self.include_scaling = include_scaling
        self._cov_hooks_attached = False
        self._cov_handles = []

    def _unwrap_model(self):
        try:
            return self.accelerator.unwrap_model(self.model)
        except Exception:
            return self.model.module if hasattr(self.model, "module") else self.model

    def _maybe_attach_hooks(self):
        if self._cov_hooks_attached:
            return
        base = self._unwrap_model()
        self._cov_handles = attach_cov_hooks_to_lora_sites(base)
        self._cov_hooks_attached = True

    def compute_loss(self, model, inputs, return_outputs=False):
        # ensure hooks are on before the first forward
        self._maybe_attach_hooks()

        outputs = model(**inputs)
        task_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        base = self._unwrap_model()
        ib_reg, per_mod_reg = ib_penalty_from_ab(base, include_scaling=self.include_scaling)

        if self.state.global_step > 0 and (self.state.global_step % self.args.logging_steps == 0):
            self.log({
                "ce_loss": task_loss.detach().item(),
                "ib_penalty_total": ib_reg.detach().item(),
            })
            for key, val in per_mod_reg.items():
                self.log({
                    f"ib_penalty_{key}": val.detach().item(),
                })


        # want to maximize MI, so we should subtract MI from loss
        loss = task_loss - self.ib_lambda * ib_reg
        return (loss, outputs) if return_outputs else loss

    def __del__(self):
        for h in self._cov_handles:
            try:
                h.remove()
            except Exception:
                pass
