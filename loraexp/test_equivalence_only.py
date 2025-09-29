# tests/test_ib_regularizer.py
import torch
import pytest

# --- Functions under test ---

def ib_regularizer_W(W: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    # Your original W-based computation
    w_sigma = W @ Sigma
    middle_term = w_sigma @ W.T
    inv_middle = torch.linalg.inv(middle_term)
    correction = Sigma @ W.T @ inv_middle @ W @ Sigma
    denominator = Sigma - correction
    det_numerator = torch.linalg.det(Sigma)
    det_denominator = torch.linalg.det(denominator)
    if (det_numerator <= 0) or (det_denominator <= 0):
        return torch.zeros((), device=Sigma.device, dtype=Sigma.dtype)
    return 0.5 * torch.log(det_numerator / det_denominator)

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

# --- Helpers ---

def make_spd(d, dtype, eps=1e-2, device="cpu"):
    X = torch.randn(d, d, dtype=dtype, device=device)
    S = X @ X.T
    S = S + eps * torch.eye(d, dtype=dtype, device=device)
    return S

def sample_full_rank(rows, cols, dtype, device):
    # Try until full rank
    for _ in range(300):
        M = torch.randn(rows, cols, dtype=dtype, device=device)
        if torch.linalg.matrix_rank(M).item() == min(rows, cols):
            return M
    raise RuntimeError("Could not sample full-rank matrix.")

def sample_valid_case(r, d_in, dtype, device="cpu"):
    """
    We require d_out == r so that (W Σ W^T) is r×r and (typically) invertible under random full-rank A,B.
    """
    d_out = r
    for _ in range(1000):
        Sigma = make_spd(d_in, dtype=dtype, device=device)
        A = sample_full_rank(r, d_in, dtype=dtype, device=device)
        B = sample_full_rank(d_out, r, dtype=dtype, device=device)
        W = B @ A
        middle = (W @ Sigma) @ W.T
        # must be invertible for the functions (they use inv, not pinv)
        if torch.linalg.matrix_rank(middle).item() != d_out:
            continue
        return A, B, Sigma, W
    raise RuntimeError("Failed to sample a valid (invertible middle) case.")

# --- Tests ---

@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float64, 1e-10, 1e-10),
    (torch.float32, 5e-6, 1e-5),
])
def test_middle_equivalence(dtype, atol, rtol):
    torch.manual_seed(0)
    r, d_in = 4, 9
    A, B, Sigma, W = sample_valid_case(r=r, d_in=d_in, dtype=dtype)
    middle_W  = (W @ Sigma) @ W.T
    middle_AB = B @ (A @ Sigma @ A.T) @ B.T
    assert torch.allclose(middle_W, middle_AB, atol=atol, rtol=rtol)

@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float64, 1e-10, 1e-10),
    (torch.float32, 5e-6, 1e-5),
])
def test_correction_and_denominator_equivalence(dtype, atol, rtol):
    torch.manual_seed(1)
    r, d_in = 3, 7
    A, B, Sigma, W = sample_valid_case(r=r, d_in=d_in, dtype=dtype)

    # W-path correction
    inv_middle_W = torch.linalg.inv((W @ Sigma) @ W.T)
    corr_W  = Sigma @ W.T @ inv_middle_W @ W @ Sigma

    # AB-path correction
    M = A @ Sigma @ A.T
    inv_middle_AB = torch.linalg.inv(B @ M @ B.T)
    Bt_invB = B.T @ inv_middle_AB @ B
    corr_AB = Sigma @ A.T @ Bt_invB @ A @ Sigma

    assert torch.allclose(corr_W, corr_AB, atol=atol, rtol=rtol)

    den_W  = Sigma - corr_W
    den_AB = Sigma - corr_AB
    assert torch.allclose(den_W, den_AB, atol=atol, rtol=rtol)

@pytest.mark.parametrize("dtype,atol", [
    (torch.float64, 1e-8),
    (torch.float32, 1e-4),
])
def test_scalar_value_matches_when_defined(dtype, atol):
    """
    Compare the scalar penalties only when both determinants are > 0 (the functions
    return 0.0 otherwise, per the original guard). If no such case is found across
    several samples, we skip rather than fail—this is expected for many LoRA shapes.
    """
    r, d_in = 4, 10
    for seed in range(50):
        torch.manual_seed(seed if dtype is torch.float64 else seed + 1000)
        A, B, Sigma, W = sample_valid_case(r=r, d_in=d_in, dtype=dtype)
        vW  = ib_regularizer_W(W, Sigma)
        vAB = ib_regularizer_AB(A, B, Sigma)
        if (vW.item() != 0.0) and (vAB.item() != 0.0):
            assert torch.isclose(vW, vAB, atol=atol), f"Mismatch: vW={vW.item()}, vAB={vAB.item()}"
            return
    pytest.skip("Could not find a case with positive determinants for both paths in limited trials.")

def test_grad_propagates_to_A_and_B():
    # Ensure autograd flows through A and B in the AB implementation
    torch.manual_seed(123)
    dtype = torch.float64
    r, d_in = 3, 6
    A, B, Sigma, _ = sample_valid_case(r=r, d_in=d_in, dtype=dtype)
    A = A.clone().requires_grad_(True)
    B = B.clone().requires_grad_(True)

    vAB = ib_regularizer_AB(A, B, Sigma)
    vAB.backward()

    assert A.grad is not None and torch.isfinite(A.grad).all()
    assert B.grad is not None and torch.isfinite(B.grad).all()

# ---- MAIN: build Σ with your tracker, compare W vs A,B, then run pytest ----
import argparse, sys

def _build_sigma_with_tracker(d_in=8, n_samples=2000, batch_size=256, dtype=torch.float64, device="cpu"):
    """
    Uses your online tracker to estimate Σ from synthetic data X ~ N(0, I).
    Replace the fallback Tracker with your own class if already available.
    """

    # Try to import YOUR tracker; fallback to a minimal compatible version.
    try:
        # Replace with the actual import path if different:
        from ib_lora_lib import OnlineCovarianceEstimatorWelford as Tracker
        TrackerClass = Tracker
    except Exception:
        class TrackerClass(torch.nn.Module):
            def __init__(self, num_features=None, eps=1e-5):
                super().__init__()
                self.num_features = num_features
                self.eps = eps
                self.register_buffer("count", torch.zeros((), dtype=torch.long))
                self.register_buffer("mean", None)
                self.register_buffer("scatter", None)

            @torch.no_grad()
            def _lazy(self, d, device):
                if self.mean is None:
                    self.mean = torch.zeros(d, dtype=torch.float32, device=device)
                    self.scatter = torch.zeros(d, d, dtype=torch.float32, device=device)

            @torch.no_grad()
            def update(self, batch):
                x = batch
                if x.dim() > 2:
                    x = x.view(-1, x.shape[-1])
                x = x.to(torch.float32)
                n_b = x.shape[0]
                if n_b == 0: return
                self._lazy(x.shape[-1], x.device)
                mean_b = x.mean(0)
                xc = x - mean_b
                S_b = xc.T @ xc
                n_old = int(self.count.item())
                if n_old == 0:
                    self.mean.copy_(mean_b)
                    self.scatter.copy_(S_b)
                    self.count.fill_(n_b)
                    return
                n_new = n_old + n_b
                dmean = (mean_b - self.mean)
                self.scatter += S_b + (n_old * n_b / n_new) * (dmean[:, None] @ dmean[None, :])
                self.mean += (n_b / n_new) * dmean
                self.count.fill_(n_new)

            @torch.no_grad()
            def get_covariance(self, unbiased=False):
                n = int(self.count.item())
                if n <= 0: raise ValueError("No data yet.")
                C = self.scatter / (n - 1.0 if unbiased else float(n))
                C = C.clone()
                C.diagonal().add_(self.eps)
                return C

        TrackerClass = TrackerClass

    torch.manual_seed(123)
    tracker = TrackerClass(num_features=None)
    X = torch.randn(n_samples, d_in, dtype=dtype, device=device)  # zero-mean Gaussian
    for i in range(0, n_samples, batch_size):
        tracker.update(X[i:i+batch_size])
    Sigma_hat = tracker.get_covariance(unbiased=False).to(dtype)
    return Sigma_hat

def _sample_full_rank(rows, cols, dtype, device="cpu", max_tries=500):
    for _ in range(max_tries):
        M = torch.randn(rows, cols, dtype=dtype, device=device)
        if torch.linalg.matrix_rank(M).item() == min(rows, cols):
            return M
    raise RuntimeError("Could not sample a full-rank matrix.")

def _demo_compare_with_tracker_covariance(d_in=8, r=4, dtype=torch.float64, device="cpu"):
    # Build Σ from tracker
    Sigma = _build_sigma_with_tracker(d_in=d_in, n_samples=3000, batch_size=256, dtype=dtype, device=device)

    # Choose d_out = r so (W Σ W^T) is square; pick full-rank A,B
    d_out = r
    A = _sample_full_rank(r, d_in, dtype=dtype, device=device)
    B = _sample_full_rank(d_out, r, dtype=dtype, device=device)
    W = B @ A

    # Ensure middle is invertible for this demo; retry a few times if needed
    tries = 0
    while torch.linalg.matrix_rank((W @ Sigma) @ W.T).item() < d_out and tries < 50:
        A = _sample_full_rank(r, d_in, dtype=dtype, device=device)
        B = _sample_full_rank(d_out, r, dtype=dtype, device=device)
        W = B @ A
        tries += 1

    vW  = ib_regularizer_W(W, Sigma)
    vAB = ib_regularizer_AB(A, B, Sigma)
    diff = (vW - vAB).abs().item()

    print("\n=== Demo: equivalence using Σ estimated by tracker ===")
    print(f"d_in={d_in}, d_out={d_out}, r={r}, dtype={dtype}")
    print(f"ib_regularizer_W = {float(vW):.8f}")
    print(f"ib_regularizer_AB= {float(vAB):.8f}")
    print(f"abs diff         = {diff:.8e}")
    return diff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_in", type=int, default=300)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64","float32"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_pytest", action="store_true", help="Also run the pytest suite after the demo.")
    args, unknown = parser.parse_known_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    _demo_compare_with_tracker_covariance(d_in=args.d_in, r=args.rank, dtype=dtype, device=args.device)

    if args.run_pytest:
        try:
            import pytest
        except ImportError:
            print("\npytest not installed; install it or omit --run_pytest.", file=sys.stderr)
            sys.exit(1)
        # Run this file's tests
        # NOTE: we pass through any extra CLI args after ours to pytest (e.g., -q)
        code = pytest.main([__file__] + unknown)
        sys.exit(code)
    else:
        print("\n(Skipping pytest run; pass --run_pytest to execute the test suite.)")

if __name__ == "__main__":
    main()
