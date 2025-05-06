import torch
import torch.nn.functional as F

import numpy as np
import cv2

# def normalize_depth_01(depth_input, min_depth=0.07, max_depth=2.0):
#     """Normalize depth to range [0, 1] for CNN input."""
#     depth_input = torch.nan_to_num(depth_input, nan=0.0)
#     depth_input = depth_input.reshape(depth_input.shape[0], -1)
#     depth_input = torch.clamp(depth_input, min_depth, max_depth)  # Ensure valid depth range
#     depth_input = (depth_input - min_depth) / (max_depth - min_depth)
#     return depth_input

# import torch

def filter_sim_depth(sim_depth, min_depth=0.10, max_depth=0.45):
    depth_rotated = torch.rot90(sim_depth, k=1, dims=(-2, -1))
    filtered_depth = torch.nan_to_num(depth_rotated, 
                                      nan=0.0,
                                      posinf=max_depth,
                                      neginf=min_depth,
                                      )
    filtered_depth = torch.clamp(filtered_depth, min=min_depth, max=max_depth)

    return filtered_depth.reshape(-1, 120*120)


def normalize_depth_01(depth_input, min_depth=0.10, max_depth=0.5):
    """Normalize depth to range [0, 1] while preserving spatial structure."""
    normalized_depth = (depth_input - min_depth) / (max_depth - min_depth)

    return normalized_depth.reshape(-1,120*120)

def make_box_mask(N: int, H: int, W: int,
                  x_min: int, x_max: int,
                  y_min: int, y_max: int,
                  device=None) -> torch.Tensor:
    """
    Returns a (N, H*W) binary mask with 1 inside the box.
    """
    device = device or torch.device('cuda')
    # create coordinate grids
    xs = torch.arange(W, device=device)[None, None, :].expand(N, H, W)
    ys = torch.arange(H, device=device)[None, :, None].expand(N, H, W)
    mask2d = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
    return mask2d.view(N, H*W).to(device)

def add_noise_in_depth_band(depth: torch.Tensor,
                            min_d: float = 0.110,
                            max_d: float = 0.19,
                            mean: float = 0.0,
                            std: float = 0.01,
                            clip_min: float = 0.1,
                            clip_max: float = 0.45) -> torch.Tensor:
    """
    Add Gaussian noise only to depth values in [min_d, max_d].

    Args:
        depth   (torch.Tensor): shape (N, H*W) in meters.
        min_d   (float): lower depth bound to noise.
        max_d   (float): upper depth bound to noise.
        mean    (float): noise mean (m).
        std     (float): noise stddev (m).
        clip_min(float): min valid depth (after noise).
        clip_max(float): max valid depth (after noise), or None to skip.

    Returns:
        torch.Tensor: same shape, with noise added in that band only.
    """
    # 1) make a mask of where depth ∈ [min_d, max_d]
    mask = (depth >= min_d) & (depth <= max_d)  # bool tensor of shape (N, H*W)

    # 2) sample a full noise map, then zero it outside the band
    noise = torch.randn_like(depth) * std + mean
    noise = noise * mask.to(depth.dtype)

    # 3) add and clamp
    depth_noisy = depth + noise
    if clip_max is not None:
        depth_noisy = torch.clamp(depth_noisy, clip_min, clip_max)
    else:
        depth_noisy = torch.clamp(depth_noisy, clip_min)

    return depth_noisy

def save_tensor_as_txt(tensor, filename_prefix="depth_map"):
    """
    Save a (num_envs, 120*120) PyTorch tensor as separate .txt files.

    Args:
        tensor (torch.Tensor): Tensor of shape (num_envs, 120*120).
        filename_prefix (str): Prefix for the saved file names.
    """
    assert tensor.ndim == 2, "Tensor must have shape (num_envs, 120*120)"
    
    num_envs = tensor.shape[0]
    
    # Move tensor to CPU and convert to NumPy
    tensor_np = tensor.reshape(-1, 120, 120).cpu().numpy()  # Shape (num_envs, 120, 120)

    for i in range(num_envs):
        filename = f"{filename_prefix}_env{i}.txt"
        np.savetxt(filename, np.round(tensor_np[i], 2), fmt="%.2f")
        print(f"Saved: {filename}")

def transform_point_eef_to_cam(p_eef: torch.Tensor, T_cam2eef: torch.Tensor) -> torch.Tensor:
    """
    Transforms a point from the EEF frame to the camera frame.
    
    Args:
        p_eef (torch.Tensor): A point in the EEF frame (shape: (3,) or (N,3)).
        T_eef2cam (torch.Tensor): 4x4 homogeneous transform from EEF to camera frame.
    
    Returns:
        torch.Tensor: The point in the camera frame (shape: (3,) or (N,3)).
    """
    T_eef2cam = T_cam2eef.inverse()  # Inverse transform to go from EEF to camera frame

    if p_eef.dim() == 1:
        p_eef_h = torch.cat([p_eef, torch.tensor([1.0], device=p_eef.device)])
        p_cam_h = T_eef2cam @ p_eef_h
        return p_cam_h[:3] / p_cam_h[3]
    else:
        ones = torch.ones((p_eef.shape[0], 1), device=p_eef.device, dtype=p_eef.dtype)
        p_eef_h = torch.cat([p_eef, ones], dim=1)  # shape (N,4)
        p_cam_h = (T_eef2cam.unsqueeze(0) @ p_eef_h.unsqueeze(-1)).squeeze(-1)  # (N,4)
        return p_cam_h[:, :3] / p_cam_h[:, 3:4]
    

def project_points(points_3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Projects 3D points in the camera frame onto the 2D image plane using the pinhole camera model.

    Args:
        points_3d (torch.Tensor): Tensor of shape (N, 3) representing 3D points in the camera coordinate frame.
        K (torch.Tensor): Camera intrinsics matrix of shape (3, 3) in the form:
                          [[fx, 0,  cx],
                           [0,  fy, cy],
                           [0,  0,  1]].
    
    Returns:
        torch.Tensor: Tensor of shape (N, 2) containing the projected 2D pixel coordinates.
    """
    # Compute the homogeneous projection: shape (N, 3)
    # Multiply each 3D point by K:
    points_hom = (K @ points_3d.T).T  # (N, 3)
    
    # Avoid division by zero: if any Z is zero, we add a small epsilon.
    eps = 1e-6
    Z = points_hom[:, 2:3] + eps  # shape (N, 1)
    
    # Normalize to get pixel coordinates.
    u = points_hom[:, 0:1] / Z
    v = points_hom[:, 1:2] / Z
    
    # Concatenate to shape (N, 2)
    points_2d = torch.cat([u, v], dim=1)
    return points_2d

def visualize_points_on_image(points_2d: torch.Tensor, image: np.ndarray, color=(0, 255, 0), radius=0.3):
    """
    Overlays 2D points onto an image.
    
    Args:
        points_2d (torch.Tensor): Tensor of shape (N, 2) with pixel coordinates.
        image (np.ndarray): Image in BGR format (as from cv2).
        color (tuple): Color for the drawn points (B, G, R).
        radius (int): Radius of the circle to draw.
    
    Returns:
        np.ndarray: Image with overlaid points.
    """
    # Convert the points to CPU numpy int32 array.
    points_np = points_2d.cpu().numpy().astype(np.int32)
    for point in points_np:
        cv2.circle(image, (point[0], point[1]), radius, color, -1) # type: ignore
    return image

def filter_depth_for_visualization(depth: np.ndarray, max_depth: float = 0.5, min_depth: float = 0.1, unit: str = "mm") -> np.ndarray:
    """
    filter depth to be visualized using cv2.
    """

    if unit == "mm":
        depth = depth / 1000.0
    elif unit == "cm":
        depth = depth / 100.0

    # remove NaN and Inf values
    depth_filtered = np.nan_to_num(
        depth,
        nan=0.0,
        posinf=max_depth,
        neginf=min_depth,
    )
    
    # clip depth
    depth = np.clip(depth_filtered, min_depth, max_depth)

    # Normalize to [0, 255] and convert to uint8
    depth_uint8 = (depth * 255.0).astype(np.uint8)
    
    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_vis = cv2.resize(depth_vis, (480, 480))
    
    return depth_vis