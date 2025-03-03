import torch

def normalize_depth_01(depth_input, min_depth=0.07, max_depth=2.0):
    """Normalize depth to range [0, 1] for CNN input."""
    depth_input = torch.nan_to_num(depth_input, nan=0.0)
    depth_input = depth_input.reshape(depth_input.shape[0], -1)
    depth_input = torch.clamp(depth_input, min_depth, max_depth)  # Ensure valid depth range
    depth_input = (depth_input - min_depth) / (max_depth - min_depth)
    return depth_input


def add_depth_dependent_noise_torch(depth, base_std=0.005, scale=0.02):
    """Adds Gaussian noise where noise increases with depth distance."""
    noise_std = base_std + scale * depth  # Standard deviation grows with depth
    noise = torch.randn_like(depth) * noise_std  # Generate Gaussian noise
    return torch.clamp(depth + noise, min=0)  # Ensure depth is non-negative

def add_salt_and_pepper_noise_torch(depth, prob=0.02):
    """Adds salt-and-pepper noise by setting some pixels to 0 (missing depth) or max depth."""
    noisy_depth = torch.clone(depth)
    mask = torch.randint(0, 100, depth.shape, device=depth.device) / 100.0  # Random values between 0 and 1

    noisy_depth[mask < (prob / 2)] = 0  # Set some pixels to 0 (black, missing depth)
    noisy_depth[(mask >= (prob / 2)) & (mask < prob)] = depth.max()  # Set some pixels to max depth (white noise)
    
    return noisy_depth

def generate_perlin_noise(shape, scale=0.1, octaves=4, device="cuda"):
    """Generates batched Perlin noise for (num_envs, N, H, W) tensors."""
    num_envs, num_channels, height, width = shape
    perlin_noise = torch.zeros((num_envs, height, width), dtype=torch.float32, device=device)

    # Initialize frequency and amplitude for octaves
    frequency = 1.0
    amplitude = scale

    for _ in range(octaves):
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, frequency, height, device=device),
            torch.linspace(0, frequency, width, device=device),
            indexing="ij"
        )
        grid_x = grid_x.unsqueeze(0).expand(num_envs, -1, -1)  # Expand for batch
        grid_y = grid_y.unsqueeze(0).expand(num_envs, -1, -1)

        # Random gradients for each environment
        gradient = torch.rand((num_envs, height, width), device=device) * 2 - 1  # [-1,1]
        perlin_noise += amplitude * torch.cos(grid_x + grid_y + gradient)

        frequency *= 2  # Increase frequency for finer details
        amplitude *= 0.5  # Reduce amplitude for higher frequencies

    return perlin_noise

def add_perlin_noise(depth_images, scale=0.1, octaves=4, device="cuda"):
    """
    Adds Perlin noise to a batch of depth images.
    
    Args:
        depth_images (torch.Tensor): Tensor of shape (num_envs, H, W).
        scale (float): Scaling factor for noise intensity.
        device (str): Device to run computation on.

    Returns:
        torch.Tensor: Noisy depth images of shape (num_envs, H, W).
    """
    depth_images = depth_images.to(device)  # Move to GPU
    perlin_map = generate_perlin_noise(depth_images.shape, scale, octaves, device=device)
    return depth_images + perlin_map


def simulate_depth_noise(depth):
    """Simulates realistic depth noise including Gaussian noise and salt-and-pepper noise."""
    depth = add_depth_dependent_noise_torch(depth, base_std=0.005, scale=0.01)
    depth = add_salt_and_pepper_noise_torch(depth, prob=0.01)
    # depth = add_perlin_noise(depth, scale=0.05, octaves=4)
    return depth