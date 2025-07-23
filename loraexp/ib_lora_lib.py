import numpy as np
import torch



class OnlineCovarianceEstimator:
    def __init__(self, num_features, decay=0.9):
        self.num_features = num_features
        self.decay = decay
        self.mean = np.zeros(num_features)
        self.covariance = np.zeros((num_features, num_features))
        self.count = 0
    def update(self, batch):
        batch_size = batch.shape[0]
        batch_mean = np.mean(batch, axis=0)
        
        # Update the mean using a moving average
        self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
        
        # Compute the batch covariance
        centered_batch = batch - batch_mean
        batch_covariance = np.dot(centered_batch.T, centered_batch) / batch_size
        
        # Update the covariance matrix using a moving average
        self.covariance = self.decay * self.covariance + (1 - self.decay) * batch_covariance
        
        # Update the count of processed samples
        self.count += batch_size
    def get_covariance(self):
        return self.covariance

class ib_regularizer:
    def __init__(self):
        pass

    def regularize(self, W, sigma):
        """
        W is the LoRA matrix (AB)
        Sigma is the covariance matrix

        """
        # WΣ
        w_sigma = W @ sigma
        # calculate the term corresponding to (WΣ)W.T
        middle_term = w_sigma @ W.T
        # take the inverse ((WΣ)W.T)^-1 
        inverse = torch.linalg.inv(middle_term)  
        # would torch.linalg.pinv be safer to use?

        # calculate the correction temr 
        correction = sigma @ W.T @ inv_middle @ W @ sigma
        denominator = sigma - correction

        # take the determinants of the numerator and denominator
        det_numerator = torch.linalg.det(sigma)
        det_denominator = torch.linalg.det(denominator)

        # consider stabilizing the computation for denominator like the following
        # det_denominator = torch.linalg.det(denominator + 1e-6 * torch.eye(sigma.shape[0]))

        # skip unstable cases
        if det_numerator <= 0 or det_denominator <= 0:
            return torch.tensor(0.0, device=sigma.device) 

        regualrizer = 0.5 * torch.log(det_numerator / det_denominator)
        return regualrizer


def __main__():
    # Example usage
    num_features = 5
    estimator = OnlineCovarianceEstimator(num_features)
    # Simulate streaming data with mini-batches
    for _ in range(100):
        batch = np.random.randn(10, num_features)  # Replace with your data
        estimator.update(batch)
    covariance_matrix = estimator.get_covariance()
    print("Estimated Covariance Matrix:\n", covariance_matrix) 