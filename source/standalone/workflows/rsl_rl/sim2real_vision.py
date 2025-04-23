import numpy as np
import cv2
import matplotlib.pyplot as plt
from RRL.utilities.traj_generator import load_from_txt
import torch
import torch.nn.functional as F

def visualize_depth_mse(depth1: np.ndarray,
                        depth2: np.ndarray,
                        vmin: float,
                        vmax: float,
                        colormap: int = cv2.COLORMAP_JET) -> None:
    """
    Compute and print the average pixel‐level MSE between two depth maps,
    then display them side‐by‐side using the specified OpenCV colormap,
    with a shared normalization range [vmin, vmax].
    
    Args:
        depth1 (np.ndarray): First depth map of shape (H, W).
        depth2 (np.ndarray): Second depth map of shape (H, W).
        vmin (float): Minimum depth value for colormap normalization.
        vmax (float): Maximum depth value for colormap normalization.
        colormap (int): OpenCV colormap flag (e.g. cv2.COLORMAP_JET).
    """
    # 1) Compute MSE on raw floats
    d1 = depth1.astype(np.float32)
    d2 = depth2.astype(np.float32)
    mse = np.mean((d1 - d2)**2)
    print(f"Average pixel‐level MSE: {mse:.6f}")

    # 2) Normalize both with same vmin, vmax to [0,255]
    def to_u8_shared(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax > vmin:
            scaled = 255 * (arr - vmin) / (vmax - vmin)
        else:
            scaled = np.zeros_like(arr)
        return np.clip(scaled, 0, 255).astype(np.uint8)

    d1_u8 = to_u8_shared(d1, vmin, vmax)
    d2_u8 = to_u8_shared(d2, vmin, vmax)

    # 3) Apply colormap (OpenCV uses BGR)
    cmap1 = cv2.applyColorMap(d1_u8, colormap)
    cmap2 = cv2.applyColorMap(d2_u8, colormap)

    # Convert BGR → RGB for matplotlib
    cmap1 = cv2.cvtColor(cmap1, cv2.COLOR_BGR2RGB)
    cmap2 = cv2.cvtColor(cmap2, cv2.COLOR_BGR2RGB)

    # 4) Plot side‑by‑side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(d1_u8)
    axes[0].set_title("Depth Map Sim")
    axes[0].axis("off")
    axes[1].imshow(d2_u8)
    axes[1].set_title("Depth Map Real")
    axes[1].axis("off")
    plt.suptitle(f"Avg. Pixel MSE = {mse:.6f}", fontsize=14)
    plt.tight_layout()
    plt.show()

def compare_tensors(file1: str, file2: str, method: str = 'mse'):
    """
    Compare two tensors using either MSE or L2 norm.

    Args:
        file1 (str): Path to the first .pt file.
        file2 (str): Path to the second .pt file.
        method (str): Comparison method - 'mse' or 'norm'.

    """
    # Load tensors
    t1 = torch.load(file1).to('cuda:0').reshape(120,120)
    t2 = torch.load(file2).to('cuda:0').reshape(120,120)

    depth1_np = t1.detach().cpu().numpy()
    depth2_np = t2.detach().cpu().numpy()

    import pdb; pdb.set_trace()

    # Ensure they are the same shape
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"
    
    # Compute difference
    if method == 'mse':
        err = F.mse_loss(t1, t2).item()
    elif method == 'norm':
        err = torch.norm(t1 - t2, p=2).item()
    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'mse' or 'norm'.")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(depth1_np)
    axes[0].set_title("Depth Sim")
    axes[0].axis("off")
    axes[1].imshow(depth2_np)
    axes[1].set_title("Depth Real")
    axes[1].axis("off")
    plt.suptitle(f"Avg. Pixel MSE = {err:.6f}", fontsize=14)
    plt.tight_layout()
    plt.show()

# depth_sim_raw = load_from_txt("RRL/sim2real/vision_gap/raw/visual_raw_obs_sim_env0.txt", return_type='numpy')  # shape (120,120)
# depth_real_raw = load_from_txt("RRL/sim2real/vision_gap/raw/visual_raw_obs_real_env0.txt", return_type='numpy')  # shape (120,120)
# depth_sim_normalized = load_from_txt("RRL/sim2real/vision_gap/input/visual_input_obs_sim_env0.txt", return_type='numpy')  # shape (120,120)
# depth_real_normalized = load_from_txt("RRL/sim2real/vision_gap/input/visual_input_obs_real_env0.txt", return_type='numpy')  # shape (120,120)

# visualize_depth_mse(depth_sim_raw, depth_real_raw, vmin=0.1, vmax=0.5) # type: ignore
# visualize_depth_mse(depth_sim_normalized, depth_real_normalized, vmin=0.0, vmax=1.0) # type: ignore

compare_tensors("RRL/sim2real/vision_gap/embedded/visual_obs_sim.pt", "RRL/sim2real/vision_gap/embedded/visual_obs_real.pt", method='mse') # type: ignore
# embedding_err = compare_tensors("RRL/sim2real/vision_gap/embedded/embedding_sim.pt", "RRL/sim2real/vision_gap/embedded/embedding_real.pt", method='mse') # type: ignore
