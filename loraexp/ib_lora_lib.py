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


# class OnlineCovarianceEstimator:
#     def __init__(self, num_features, decay=0.9):
#         self.num_features = num_features
#         self.decay = decay
#         self.mean = torch.zeros(num_features)
#         self.covariance = torch.zeros((num_features, num_features))
#         self.count = 0
#     def update(self, batch):
#         batch_size = batch.shape[0]
#         batch_mean = torch.mean(batch, axis=0)
        
#         # Update the mean using a moving average
#         self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
        
#         # Compute the batch covariance
#         centered_batch = batch - self.mean
#         # centered_batch = batch - batch_mean
#         batch_covariance = np.dot(centered_batch.T, centered_batch) / batch_size
        
#         # Update the covariance matrix using a moving average
#         self.covariance = self.decay * self.covariance + (1 - self.decay) * batch_covariance
        
#         # Update the count of processed samples
#         self.count += batch_size
#     def get_covariance(self):
#         return self.covariance


# class OnlineCovarianceEstimatorWelfordCorrected:
#     def __init__(self, num_features):
#         self.num_features = num_features
#         self.mean = torch.zeros(num_features)
#         self.covariance = torch.zeros((num_features, num_features))
#         self.count = 0


#     def update(self, batch):
#         batch_size = batch.shape[0]
#         batch_mean = torch.mean(batch, axis=0)
#         centered_batch = batch - batch_mean
#         scatter_batch = centered_batch.mT @ centered_batch
        
#         if self.count == 0:
#             self.mean = batch_mean
#             self.scatter = scatter_batch
#             self.count = batch_size
#         else:
#             new_count = self.count + batch_size
#             d_mean = batch_mean - self.mean       
#             self.scatter = (
#                 self.scatter
#                 + scatter_batch
#                 + (self.count * batch_size / new_count) * torch.ger(d_mean, d_mean)
#             )    
#             self.mean = self.mean + (batch_size / new_count) * d_mean
#             self.count = new_count


    # def get_covariance(self, unbiased=False):
    #     n = self.count
    #     if unbiased:
    #         if n <= 1:
    #             raise ValueError("Need at least 2 samples for unbiased covariance.")
    #         return self.scatter / (n - 1.0)
    #     else:
    #         if n <= 0:
    #             raise ValueError("No data yet.")
    #         return self.scatter / n


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
        # S_total = S_old + S_batch + n_old * N_b / n_new * (dmean ⊗ dmean)
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
    """
    Same objective:
      0.5 * log det(Σ) - 0.5 * log det(Σ - Σ W^T (WΣW^T)^{-1} W Σ),  W=BA
    but computed via A,B with solves (no change in math).
    """
    # r×r core
    M = A @ Sigma @ A.T                         # [r, r]
    # middle = (W Σ) W^T = B M B^T
    middle = B @ M @ B.T                        # [d_out, d_out]

    # Solve middle * X = (W Σ) instead of explicit inverse
    # RHS: (W Σ) = B A Σ  (shape [d_out, d_in])
    RHS = B @ A @ Sigma                         # [d_out, d_in]
    X = torch.linalg.solve(middle, RHS)         # [d_out, d_in]

    # Σ W^T (WΣW^T)^{-1} W Σ = Σ @ (W^T X)
    correction = Sigma @ ( (B @ A).T @ X )      # [d_in, d_in]

    # Symmetrize before det to remove tiny skew from FP roundoff
    Sigma_s   = 0.5 * (Sigma + Sigma.T)
    denom_s   = Sigma_s - 0.5 * (correction + correction.T)

    det_num = torch.linalg.det(Sigma_s)
    det_den = torch.linalg.det(denom_s)
    if (det_num <= 0) or (det_den <= 0):
        return torch.zeros((), device=Sigma.device, dtype=Sigma.dtype)
    return 0.5 * torch.log(det_num / det_den)


def ib_regularizer_A_only(A: torch.Tensor, Sigma: torch.Tensor, jitter: float = 1e-6):
    """
    Uses only the r×r core via Cholesky; avoids any d_out solves.
    Assumes B has full column rank (left inverse exists) and cancels
    """
    # Symmetrize Σ to kill tiny FP skew and add a small ridge for PD
    Sigma = 0.5 * (Sigma + Sigma.T)
    d = Sigma.shape[0]
    Sigma = Sigma + jitter * torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)

    # r×r core
    M = A @ Sigma @ A.T                           # [r, r]
    M = 0.5 * (M + M.T) + jitter * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)

    # chol(M) and solve M^{-1}AΣ in r×r
    L = torch.linalg.cholesky(M)                  # SPD (after jitter)
    AS = A @ Sigma                                # [r, d_in]
    Y = torch.cholesky_solve(AS, L)              # [r, d_in]

    correction = Sigma @ (A.T @ Y)               # [d_in, d_in]
    denom = 0.5 * (Sigma + Sigma.T) - 0.5 * (correction + correction.T)

    # Make denom PD with a whisper of ridge (same jitter)
    denom = 0.5 * (denom + denom.T) + jitter * torch.eye(d, device=denom.device, dtype=denom.dtype)

    # Stable log-dets
    sign_num, logdet_num = torch.slogdet(Sigma)
    sign_den, logdet_den = torch.slogdet(denom)

    # If non-PD numerically, default 0
    if (sign_num <= 0) or (sign_den <= 0):
        return torch.zeros((), device=Sigma.device, dtype=Sigma.dtype)

    return 0.5 * (logdet_num - logdet_den)


def ib_regularizer_A_pinv(
    A: torch.Tensor,                 # [r, d_in]
    Sigma: torch.Tensor,             # [d_in, d_in] (symmetric PSD)
    jitter: float = 1e-6,            # small ridge for PD-ness
    rcond: float = 1e-7,             # cutoff for pseudoinverse
    use_float64: bool = False,       # improves stability for logdets
):
    """
    Computes: 0.5 * [ logdet(Σ) - logdet(Σ - Σ A^T (A Σ A^T)^+ A Σ) ].
    Uses MP pseudoinverse on the r×r core M = A Σ A^T.
    This expression is independent of B in W = B A.

    Notes:
      • Σ should be covariance-like (symmetric, PSD). We add a tiny jitter to keep PD.
      • Works even if rank(A) < r (then M is singular and M^+ is used).
      • Differentiable in PyTorch (pinv has gradients).
    """
    dtype = torch.float64 if use_float64 else Sigma.dtype
    device = Sigma.device

    # symmetrize + ridge for slogdet
    Sigma = Sigma.to(dtype)
    Sigma = 0.5 * (Sigma + Sigma.T)
    d = Sigma.shape[0]
    Sigma = Sigma + jitter * torch.eye(d, dtype=dtype, device=device)

    # Core: M = A Σ A^T  (r×r), symmetrize
    A = A.to(dtype)
    M = A @ Sigma @ A.T
    M = 0.5 * (M + M.T)

    M_pinv = torch.linalg.pinv(M, rcond=rcond)

    correction = Sigma @ (A.T @ (M_pinv @ (A @ Sigma)))
    correction = 0.5 * (correction + correction.T)

    # jitter for PD
    denom = 0.5 * (Sigma + Sigma.T) - correction
    denom = 0.5 * (denom + denom.T) + jitter * torch.eye(d, dtype=dtype, device=device)

    sign_num, logdet_num = torch.slogdet(Sigma)
    sign_den, logdet_den = torch.slogdet(denom)

    if (sign_num <= 0) or (sign_den <= 0) or not torch.isfinite(logdet_num + logdet_den):
        return torch.zeros((), device=device, dtype=dtype)

    return 0.5 * (logdet_num - logdet_den)


# class ib_regularizer:
#     def __init__(self):
#         pass

#     def regularize(self, W, sigma): 
#         """
#         W is the LoRA matrix (AB)
#         Sigma is the covariance matrix

#         """
#         # WΣ
#         w_sigma = W @ sigma
#         # calculate the term corresponding to (WΣ)W.T
#         middle_term = w_sigma @ W.T
#         # take the inverse ((WΣ)W.T)^-1 
#         inv_middle = torch.linalg.inv(middle_term)  
#         # would torch.linalg.pinv be safer to use?

#         # calculate the correction term
#         correction = sigma @ W.T @ inv_middle @ W @ sigma
#         denominator = sigma - correction

#         # take the determinants of the numerator and denominator
#         det_numerator = torch.linalg.det(sigma)
#         det_denominator = torch.linalg.det(denominator)

#         # consider stabilizing the computation for denominator like the following
#         # det_denominator = torch.linalg.det(denominator + 1e-6 * torch.eye(sigma.shape[0]))

#         # skip unstable cases
#         if det_numerator <= 0 or det_denominator <= 0:
#             return torch.tensor(0.0, device=sigma.device) 

#         regualrizer = 0.5 * torch.log(det_numerator / det_denominator)
#         return regualrizer


def _is_lora_wrapper(m) -> bool:
    return hasattr(m, "lora_A") and hasattr(m, "lora_B")

def attach_cov_hooks_to_lora_sites(
    model,
    tracker_ctor=lambda: OnlineCovarianceEstimatorWelford(num_features=None, eps=1e-5),
):
    """
    Attach a forward_pre_hook to each LoRA *base layer* to collect X and update a tracker.
    Stores the tracker at base_layer.cov_tracker (buffers move & save with the model).
    Returns a list of hook handles (so you can remove them later if needed).
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
            # B = m.lora_B[key].weight  # [d_out, r]
            if include_scaling and hasattr(m, "scaling"):
                # reg = reg + ib_regularizer_AB(A, B * m.scaling, Sigma)
                reg_A = ib_regularizer_A_only(A, Sigma)
                reg = reg + reg_A
            else:
                # reg = reg + ib_regularizer_AB(A, B, Sigma)
                reg_A = ib_regularizer_A_only(A, Sigma)
                reg = reg + reg_A

            key = f"{mod_name}:{key}".replace(".", "_")
            per_mod_reg[key] = reg_A

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


        # want to maximize MI, so we want to minimize the IB regularizer
        loss = task_loss - self.ib_lambda * ib_reg
        return (loss, outputs) if return_outputs else loss

    def __del__(self):
        for h in self._cov_handles:
            try:
                h.remove()
            except Exception:
                pass
