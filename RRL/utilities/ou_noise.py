import torch

class OUNoise:
    """Ornstein–Uhlenbeck process in PyTorch."""

    def __init__(self, size, mu=0.0, theta=0.0067, sigma=0.001, device=None, dtype=None):
        """
        Args:
            size (int or tuple): shape of the noise vector.
            mu (float or array-like): long-term mean.
            theta (float): rate of mean reversion.
            sigma (float): scale of the noise.
            device, dtype: torch device and dtype for the tensors.
        """
        self.device = device or torch.device('cuda')
        self.dtype = dtype or torch.float32
        self.mu = torch.full(size, mu, device=self.device, dtype=self.dtype)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.clone()

    def reset(self, env_ids): 
        """Reset the internal state to the mean."""
        # Clone so that mu and state are independent tensors
        self.state[env_ids] = self.mu.clone()[env_ids]

    def sample(self):
        """
        Advance the OU process and return the new noise sample.
        dx = θ (μ – x) + σ ε,  ε ~ N(0,1)
        """
        # Gaussian noise with same shape as state
        noise = torch.randn_like(self.state, device=self.device, dtype=self.dtype)
        dx = self.theta * (self.mu - self.state) + self.sigma * noise
        self.state = self.state + dx
        return self.state


'''
tau = correlation time = 20 s
then theta = dt / tau = 1/30/20 = 0.00167

A = RMS position jitter = 0.005
alpha = 1 - theta = 1 - 0.00167 = 0.99833
var_factor = 1 - alpha*alpha = 1 - 0.99833*0.99833 = 0.0022
sigma = A * math.sqrt(var_factor) = 0.005 * math.sqrt(0.0022) = 0.0005
'''