from ib_lora_lib import *
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

def wishart_cov(d=10, n=200, device="cpu"):
    X = torch.randn(n, d, device=device)
    X -= X.mean(0)
    return (X.T @ X) / (n - 1)

def factor_cov(d=50, r=3, sigma2=0.1, device="cpu"):
    B = torch.randn(d, r, device=device)
    return B @ B.T + sigma2 * torch.eye(d, device=device)

def ar1_cov(d=50, rho=0.8, device="cpu"):
    idx = torch.arange(d, device=device)
    return rho ** (idx[:, None] - idx[None, :]).abs()

def spiked_cov(d=50, spikes=(10, 5), bulk=1.0, device="cpu"):
    m = len(spikes)
    lambdas = torch.tensor(list(spikes) + [bulk]*(d - m), device=device)
    Q, _ = torch.linalg.qr(torch.randn(d, d, device=device))
    return Q @ torch.diag(lambdas) @ Q.T

def ill_conditioned_cov(d=50, kappa=1e6, device="cpu"):
    # Create geometric sequence manually since torch.geomspace doesn't exist
    log_kappa = torch.log(torch.tensor(kappa, device=device))
    log_1 = torch.log(torch.tensor(1.0, device=device))
    log_lambdas = torch.linspace(log_kappa, log_1, d, device=device)
    lambdas = torch.exp(log_lambdas)
    Q, _ = torch.linalg.qr(torch.randn(d, d, device=device))
    return Q @ torch.diag(lambdas) @ Q.T

def kernel_rbf_cov(X, ell=1.0, sigma2=1.0):
    # X: n x p
    sq = (X**2).sum(1, keepdim=True)
    D2 = sq + sq.T - 2 * (X @ X.T)
    K = torch.exp(-0.5 * D2 / ell**2)
    return sigma2 * K

def block_diag_cov(blocks):
    return torch.block_diag(*blocks)


def test_online_covariance_estimator(estimator_class, cov_matrix, num_features, batch_size, samples, trials, device="cpu"):
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(num_features), covariance_matrix=cov_matrix)
    error_trajectories = []
    for _ in range(trials):
        estimator = estimator_class(num_features)
        error_trajectory = []
        for _ in range(samples):
            batch = dist.sample(sample_shape=(batch_size,))
            estimator.update(batch)
            # Use relative error for better interpretability
            true_norm = torch.norm(cov_matrix).item()
            error = torch.norm(estimator.get_covariance() - cov_matrix).item() / true_norm
            error_trajectory.append(error)
        error_trajectories.append(error_trajectory)
    return error_trajectories

if __name__ == "__main__":
    num_features = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 10
    samples = 1000
    trials = 10
    cov_matrix = wishart_cov(d=num_features, device=device)
    # cov_matrix = ill_conditioned_cov(d=num_features, device=device)
    # cov_matrix = spiked_cov(d=num_features, device=device)
    
    # Define estimators to test
    estimators = {
        "naive": OnlineCovarianceEstimator,
        # "welford": OnlineCovarianceEstimatorWelford,
        "welford_corrected": OnlineCovarianceEstimatorWelfordCorrected

    }
    
    # Calculate confidence intervals for trajectories
    import numpy as np
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (est_name, est_class) in enumerate(estimators.items()):
        print(f"Testing {est_name} estimator...")
        
        # Test the estimator
        error_trajectories = test_online_covariance_estimator(
            est_class, cov_matrix, num_features, batch_size, samples, trials, device
        )
        
        # Convert trajectories to numpy array for easier computation
        trajectories_array = np.array(error_trajectories)  # shape: (trials, samples)
        
        # Calculate mean and standard error across trials
        mean_trajectory = np.mean(trajectories_array, axis=0)
        std_trajectory = np.std(trajectories_array, axis=0)
        se_trajectory = std_trajectory / np.sqrt(trials)  # standard error
        
        # Calculate 95% confidence interval
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = 1.96  # approximate for large n, or use scipy.stats.t.ppf for exact
        margin_of_error = t_critical * se_trajectory
        
        lower_bound = mean_trajectory - margin_of_error
        upper_bound = mean_trajectory + margin_of_error
        
        color = colors[i % len(colors)]
        
        # Plot individual trajectories (light version of main color)
        for trajectory in error_trajectories:
            plt.plot(trajectory, alpha=0.2, color=color, linewidth=0.3)
        
        # Plot mean trajectory
        plt.plot(mean_trajectory, color=color, linewidth=2, 
                label=f'{est_name.title()} Mean Error')
        
        # Plot confidence interval
        x_vals = np.arange(len(mean_trajectory))
        plt.fill_between(x_vals, lower_bound, upper_bound, 
                         alpha=0.2, color=color, 
                         label=f'{est_name.title()} {confidence_level*100:.0f}% CI')
        
        print(f"{est_name.title()} - Final mean error: {mean_trajectory[-1]:.4f} ± {margin_of_error[-1]:.4f}")
    
    plt.xlabel('Sample Number')
    plt.ylabel('Relative Covariance Estimation Error')
    plt.title('Online Covariance Estimator Error Trajectories with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # save plots
    scratch_path = "/users/rgao48/scratch"
    plt.savefig(f"{scratch_path}/covariance_estimator_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved to {scratch_path}/covariance_estimator_comparison.png")

