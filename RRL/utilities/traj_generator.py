import torch
from omni.isaac.lab.utils.math import sample_uniform, euler_xyz_from_quat, quat_from_euler_xyz, quat_from_matrix, subtract_frame_transforms, quat_mul, matrix_from_quat
import os
from typing import Callable
import math

def abs_to_rel_traj(init_ee, abs_traj):
    """
    Convert absolute trajectory to relative trajectory with respect to the initial end-effector (EE) pose.

    Args:
        abs_traj (torch.Tensor): Absolute trajectory of shape (num_envs, action_dim * traj_length, num_demo).

    Returns:
        torch.Tensor: Relative trajectory of the same shape.
    """
    num_envs, traj_dim, num_demos = abs_traj.shape
    num_actions = traj_dim // 8  # Each action consists of 8 elements

    # Expand init_ee to match the shape of abs_traj
    first_point = init_ee.unsqueeze(-1).expand(-1, -1, num_demos)  # (num_envs, 8, num_demos)
    
    # Reshape to align action steps properly
    first_point = first_point.repeat(1, num_actions, 1)  # (num_envs, 8 * num_actions, num_demos)

    # Compute the relative trajectory
    rel_traj = abs_traj - first_point

    return rel_traj

def export_tensor_to_txt(tensor: torch.Tensor, file_path: str):
    """
    Exports a PyTorch tensor of shape (1, num_lines, values_per_line) to a .txt file.

    Args:
        tensor (torch.Tensor): The input tensor of shape (1, num_lines, values_per_line).
        file_path (str): Path to save the .txt file.
    """
    # Ensure tensor is on CPU before converting to numpy
    tensor = tensor.squeeze(0).cpu()

    # Save tensor to a text file
    with open(file_path, 'w') as f:
        for row in tensor:
            line = ' '.join(map(str, row.tolist()))  # Convert tensor row to space-separated string
            f.write(line + '\n')

def resample_trajectory(trajectory: torch.Tensor, goal_traj_length: int, waypoint_dim=10) -> torch.Tensor:
    """
    Resample a trajectory to a specified length using linear interpolation.

    Args:
        trajectory (torch.Tensor): Input trajectory tensor of shape (num_envs, current_traj_length, waypoint_dim).
        goal_traj_length (int): The desired trajectory length.
        waypoint_dim (int): The dimensionality of each waypoint.

    Returns:
        torch.Tensor: Resampled trajectory of shape (num_envs, goal_traj_length, waypoint_dim).
    """
    num_envs, current_traj_length, action_dim = trajectory.shape

    # Apply linear interpolation along the trajectory length axis
    resampled_trajectory = torch.nn.functional.interpolate(
        trajectory.permute(0, 2, 1),  # Change shape to (num_envs, action_dim, current_traj_length)
        size=goal_traj_length,
        mode='linear',
        align_corners=True
    )

    # Reshape back to (num_envs, goal_traj_length, action_dim)
    resampled_trajectory = resampled_trajectory.permute(0, 2, 1)  # Shape: (num_envs, goal_traj_length, action_dim)

    return resampled_trajectory

def resample_trajectory_10d(trajectory: torch.Tensor, goal_traj_length: int, waypoint_dim=10) -> torch.Tensor:
    """
    Resample a trajectory to a specified length using linear interpolation for position and gripper,
    and SLERP for orientation.

    The input trajectory is assumed to be of shape (num_envs, current_traj_length, waypoint_dim),
    where each waypoint is defined as:
        [position (3), orientation (6), gripper_status (1)]
    The orientation is in the 6D format [r11, r21, r31, r12, r22, r32].

    Args:
        trajectory (torch.Tensor): Input trajectory tensor of shape (num_envs, current_traj_length, waypoint_dim).
        goal_traj_length (int): The desired trajectory length.
        waypoint_dim (int): Dimensionality of each waypoint (default 10).

    Returns:
        torch.Tensor: Resampled trajectory of shape (num_envs, goal_traj_length, waypoint_dim).
    """
    num_envs, current_traj_length, _ = trajectory.shape
    device = trajectory.device
    dtype = trajectory.dtype

    # Create normalized time steps for the original and new trajectories.
    T_orig = torch.linspace(0, 1, steps=current_traj_length, device=device, dtype=dtype)
    T_new = torch.linspace(0, 1, steps=goal_traj_length, device=device, dtype=dtype)

    # ----- Linearly interpolate Position and Gripper Status -----
    # Position: first 3 dims.
    pos_traj = torch.nn.functional.interpolate(
        trajectory[:, :, :3].permute(0, 2, 1),  # shape: (num_envs, 3, current_traj_length)
        size=goal_traj_length,
        mode='linear',
        align_corners=True
    ).permute(0, 2, 1)  # shape: (num_envs, goal_traj_length, 3)

    # Gripper status: last dim.
    grip_traj = torch.nn.functional.interpolate(
        trajectory[:, :, 9:].permute(0, 2, 1),  # shape: (num_envs, 1, current_traj_length)
        size=goal_traj_length,
        mode='linear',
        align_corners=True
    ).permute(0, 2, 1)  # shape: (num_envs, goal_traj_length, 1)

    # ----- Resample Orientation using SLERP -----
    # Extract original 6D orientation (dims 3 to 8)
    orient6d = trajectory[:, :, 3:9]  # shape: (num_envs, current_traj_length, 6)
    # Convert 6D to quaternion: expect output shape (num_envs, current_traj_length, 4)
    Q = quat_from_6d(orient6d.view(-1, 6)).view(num_envs, current_traj_length, 4)


    # For resampling a trajectory of quaternions, we use piecewise SLERP.
    # For each new time in T_new, find the corresponding segment in T_orig.
    indices = torch.searchsorted(T_orig, T_new)
    # Clamp indices to be within valid range.
    indices = torch.clamp(indices, 1, current_traj_length - 1)
    i0 = indices - 1  # shape: (goal_traj_length,)
    i1 = indices      # shape: (goal_traj_length,)

    # Compute local interpolation factors for each new time.
    T_orig_i0 = T_orig[i0]  # shape: (goal_traj_length,)
    T_orig_i1 = T_orig[i1]  # shape: (goal_traj_length,)
    u = (T_new - T_orig_i0) / (T_orig_i1 - T_orig_i0 + 1e-6)  # shape: (goal_traj_length,)
    tol = 1e-3

    # For each new time step, perform SLERP between Q[:, i0, :] and Q[:, i1, :].
    Q_new_list = []
    for j in range(goal_traj_length):
        # Endpoints for the j-th segment: shape (num_envs, 4)
        q1 = Q[:, i0[j], :]
        q2 = Q[:, i1[j], :]
        if torch.allclose(q1, q2, atol=tol):
            Q_new_list.append(q1)
        else:
            # Interpolate using slerp_batch with factor u[j]
            u_j = u[j].unsqueeze(0)  # shape: (1,)
            q_new = slerp_batch(q1, q2, u_j)  # shape: (num_envs, 1, 4)
            Q_new_list.append(q_new.squeeze(1))

    # Stack the interpolated quaternions along time: shape (num_envs, goal_traj_length, 4)
    Q_new = torch.stack(Q_new_list, dim=1)
    # Convert interpolated quaternions back to 6D representation.
    Q_new_flat = Q_new.view(-1, 4)
    orient6d_new_flat = quat_to_6d(Q_new_flat)  # shape: (num_envs * goal_traj_length, 6)
    orient6d_new = orient6d_new_flat.view(num_envs, goal_traj_length, 6)

    # ----- Concatenate all components -----
    # pos_traj: (num_envs, goal_traj_length, 3)
    # orient6d_new: (num_envs, goal_traj_length, 6)
    # grip_traj: (num_envs, goal_traj_length, 1)
    resampled_trajectory = torch.cat([pos_traj, orient6d_new, grip_traj], dim=-1)  # shape: (num_envs, goal_traj_length, 10)
    return resampled_trajectory

def postprocess_real_demo_trajectory_test(w_s2w_r, folder_path):
    """
    Process a folder containing robot end-effector trajectory waypoints stored in .txt files.

    Each file contains the position, rotation matrix, and gripper status, which are extracted and converted
    into a structured tensor.

    Args:
        quat_diff (torch.Tensor): A quaternion adjustment factor applied to the extracted quaternions.
        folder_path (str): Path to the folder containing the trajectory files.

    Returns:
        torch.Tensor: A structured tensor of shape (1, traj_length, waypoint_dim), where each waypoint
                      consists of [position (3), quaternion (4), gripper status (1)].
    """
    import os
    import torch

    waypoints = []
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Extract position
                position = torch.tensor([float(x) for x in lines[0].strip().split()], device='cuda')
                # Apply transformation to position
                p_sim = position#(w_s2w_r @ position)

                p_sim[0] += 0.1
                p_sim[1] += 0.12
                p_sim[2] += 0.16
                
                # Extract rotation matrix and convert to quaternion
                w_r2ee_r = torch.tensor([ # wr2eefr
                    [float(x) for x in lines[1].strip().split()],
                    [float(x) for x in lines[2].strip().split()],
                    [float(x) for x in lines[3].strip().split()]
                ], device='cuda')

                rotation_matrix = w_r2ee_r #w_s2w_r @ w_r2ee_r
                
                orientation_6d = rotation_matrix_to_6d(rotation_matrix)  # Converts to (6,)
                
                # Extract gripper status
                real_gripper_status = float(lines[4].strip().split()[0])
                gripper_status = gripper_real2sim(real_gripper_status)
                
                # Combine into waypoint
                waypoint = torch.cat([p_sim, orientation_6d, torch.tensor([gripper_status], device='cuda')])
                waypoints.append(waypoint)
    
    # Convert list to torch tensor of shape (1, traj_length, waypoint_dim)
    trajectory_tensor = torch.stack(waypoints).unsqueeze(0)  # Add batch dim
    
    return trajectory_tensor  # Shape: (1, traj_length, waypoint_dim)

def postprocess_real_demo_trajectory_6D(w_s2w_r, folder_path):
    """
    Process a folder containing robot end-effector trajectory waypoints stored in .txt files.

    Each file contains the position, rotation matrix, and gripper status, which are extracted and converted
    into a structured tensor.

    Args:
        quat_diff (torch.Tensor): A quaternion adjustment factor applied to the extracted quaternions.
        folder_path (str): Path to the folder containing the trajectory files.

    Returns:
        torch.Tensor: A structured tensor of shape (1, traj_length, waypoint_dim), where each waypoint
                      consists of [position (3), quaternion (4), gripper status (1)].
    """
    import os
    import torch

    waypoints = []
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Extract position
                position = torch.tensor([float(x) for x in lines[0].strip().split()], device='cuda')
                # Apply transformation to position
                p_sim = (w_s2w_r @ position)

                p_sim[0] += 0.1
                p_sim[1] += 0.12
                p_sim[2] += 0.16
                
                # Extract rotation matrix and convert to quaternion
                w_r2ee_r = torch.tensor([ # wr2eefr
                    [float(x) for x in lines[1].strip().split()],
                    [float(x) for x in lines[2].strip().split()],
                    [float(x) for x in lines[3].strip().split()]
                ], device='cuda')

                rotation_matrix = w_s2w_r @ w_r2ee_r
                
                orientation_6d = rotation_matrix_to_6d(rotation_matrix)  # Converts to (6,)
                
                # Extract gripper status
                real_gripper_status = float(lines[8].strip().split()[0])
                gripper_status = gripper_real2sim(real_gripper_status)
                
                # Combine into waypoint
                waypoint = torch.cat([p_sim, orientation_6d, torch.tensor([gripper_status], device='cuda')])
                waypoints.append(waypoint)
    
    # Convert list to torch tensor of shape (1, traj_length, waypoint_dim)
    trajectory_tensor = torch.stack(waypoints).unsqueeze(0)  # Add batch dim
    
    return trajectory_tensor  # Shape: (1, traj_length, waypoint_dim)

def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix to a 6D orientation representation.
    
    The 6D representation is obtained by concatenating the first two columns of the rotation matrix 
    in column-major order: [r11, r21, r31, r12, r22, r32].
    
    Args:
        R (torch.Tensor): Rotation matrix of shape (3, 3) or (batch_size, 3, 3).
    
    Returns:
        torch.Tensor: 6D representation as a tensor of shape (6,) or (batch_size, 6).
    """
    squeeze_output = False
    if R.dim() == 2:  # Single rotation matrix
        R = R.unsqueeze(0)
        squeeze_output = True

    # Extract the first two columns of R: shape (batch_size, 3, 2)
    R_6d = R[..., :2]
    # Permute to get column-major ordering: shape (batch_size, 2, 3)
    R_6d = R_6d.permute(0, 2, 1)
    # Flatten the last two dimensions to get shape (batch_size, 6)
    R_6d = R_6d.reshape(R.shape[0], 6)

    return R_6d.squeeze(0) if squeeze_output else R_6d

def validate_rotation_matrix(R: torch.Tensor, tol=1e-3) -> bool:
    """
    Check whether each rotation matrix in R is orthonormal.
    
    Args:
        R (torch.Tensor): Rotation matrix of shape (..., 3, 3).
        tol (float): Tolerance for the difference from the identity.
        
    Returns:
        bool: True if all matrices are within tolerance of being orthonormal.
    """
    # Create an identity matrix of shape (3, 3) on the same device and dtype.
    I = torch.eye(3, device=R.device, dtype=R.dtype)
    # Compute R R^T and compare to I.
    # If R is batched, this computes the norm over the last two dims.
    err = torch.norm(torch.matmul(R, R.transpose(-2, -1)) - I, dim=(-2, -1))
    return (err < tol).all() # type: ignore

def project_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
    """
    Project each rotation matrix in R onto SO(3) using SVD.
    
    Args:
        R (torch.Tensor): Rotation matrix of shape (..., 3, 3).
    
    Returns:
        torch.Tensor: Projected rotation matrix of the same shape.
    """
    squeeze_output = False
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze_output = True
    U, S, Vh = torch.linalg.svd(R)
    R_proj = U @ Vh
    # Ensure determinant is 1: if det(R_proj) is negative, flip the sign of the last column of U.
    det = torch.det(R_proj)
    mask = (det < 0)
    if mask.any():
        U[mask, :, -1] = -U[mask, :, -1]
        R_proj = U @ Vh
    if squeeze_output:
        R_proj = R_proj.squeeze(0)
    return R_proj

def sixd_to_rotation_matrix(sixd: torch.Tensor, eps=1e-6, collinearity_thresh=0.99, tol=1e-3) -> torch.Tensor:
    """
    Convert a 6D orientation representation to a rotation matrix with checks to avoid degenerate cases.
    
    This function uses a Gram–Schmidt process (with safety checks) to form a rotation matrix.
    After computing the matrix, it checks whether the matrix is orthonormal and, if not, projects it onto SO(3).
    
    Args:
        sixd (torch.Tensor): 6D orientation tensor of shape (6,) or (batch_size, 6).
        eps (float): Small constant to avoid division by zero.
        collinearity_thresh (float): Dot-product threshold to detect near-collinearity.
        tol (float): Tolerance for the validation of the rotation matrix.
    
    Returns:
        torch.Tensor: Rotation matrix of shape (3, 3) or (batch_size, 3, 3).
    """
    squeeze_output = False
    if sixd.dim() == 1:
        sixd = sixd.unsqueeze(0)
        squeeze_output = True

    # Split into two 3D vectors.
    a = sixd[:, :3]  # candidate for first column
    b = sixd[:, 3:6]  # candidate for second column

    # Check norm of first vector.
    a_norm = a.norm(dim=1, keepdim=True)
    if torch.isnan(a_norm).any() or torch.isinf(a_norm).any():
        import pdb; pdb.set_trace()
        raise ValueError("Invalid values in a_norm")
    if (a_norm < eps).any():
        # Replace near-zero vectors with a default direction.
        a = torch.where(a_norm < eps, torch.tensor([1.0, 0.0, 0.0], device=sixd.device).expand_as(a), a)
        a_norm = a.norm(dim=1, keepdim=True)
    r1 = a / (a_norm + eps)

    # Normalize second vector candidate.
    b_norm = b.norm(dim=1, keepdim=True)
    if (b_norm < eps).any():
        b = torch.tensor([0.0, 1.0, 0.0], device=sixd.device).expand_as(b)
        b_norm = b.norm(dim=1, keepdim=True)
    b = b / (b_norm + eps)

    # Check for near-collinearity.
    dot = (r1 * b).sum(dim=1, keepdim=True)
    nearly_collinear = (dot.abs() > collinearity_thresh)
    if nearly_collinear.any():
        # For problematic cases, choose a candidate vector that is guaranteed not to be collinear.
        default = torch.tensor([0.0, 0.0, 1.0], device=sixd.device).expand_as(r1)
        alternative = torch.tensor([0.0, 1.0, 0.0], device=sixd.device).expand_as(r1)
        condition = (r1 - default).norm(dim=1, keepdim=True) < eps
        candidate = torch.where(condition, alternative, default)
        b = torch.cross(r1, candidate, dim=1)
        b = torch.nn.functional.normalize(b, dim=1)
    else:
        # Standard Gram–Schmidt orthogonalization.
        b_proj = dot * r1
        b_ortho = b - b_proj
        b = torch.nn.functional.normalize(b_ortho, dim=1)

    # Compute third column as cross product.
    r3 = torch.cross(r1, b, dim=1)
    R = torch.stack((r1, b, r3), dim=2)

    # Validate the resulting rotation matrix.
    if not validate_rotation_matrix(R, tol=tol):
        R = project_rotation_matrix(R)
    return R.squeeze(0) if squeeze_output else R

def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    R = matrix_from_quat(quat)
    return rotation_matrix_to_6d(R)

def quat_from_6d(sixd: torch.Tensor) -> torch.Tensor:
    R = sixd_to_rotation_matrix(sixd)
    return quat_from_matrix(R)

def postprocess_real_demo_trajectory_quat(quat_diff, folder_path):
    """
    Process a folder containing robot end-effector trajectory waypoints stored in .txt files.

    Each file contains the position, rotation matrix, and gripper status, which are extracted and converted
    into a structured tensor.

    Args:
        quat_diff (torch.Tensor): A quaternion adjustment factor applied to the extracted quaternions.
        folder_path (str): Path to the folder containing the trajectory files.

    Returns:
        torch.Tensor: A structured tensor of shape (1, traj_length, waypoint_dim), where each waypoint
                      consists of [position (3), quaternion (4), gripper status (1)].
    """
    import os
    import torch

    waypoints = []
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Extract position
                position = torch.tensor([float(x) for x in lines[0].strip().split()], device='cuda')
                position[0] += 0.1
                position[1] *= -1
                position[1] += 0.12
                position[2] *= -1
                position[2] += 0.16
                
                # Extract rotation matrix and convert to quaternion
                rotation_matrix = torch.tensor([
                    [float(x) for x in lines[1].strip().split()],
                    [float(x) for x in lines[2].strip().split()],
                    [float(x) for x in lines[3].strip().split()]
                ], device='cuda')
                quaternion = quat_from_matrix(rotation_matrix)  # Converts to (w, x, y, z) format
                # quaternion = quat_mul(quat_diff.squeeze(0), quaternion)
                
                # Extract gripper status
                real_gripper_status = float(lines[8].strip().split()[0])
                gripper_status = gripper_real2sim(real_gripper_status)
                
                # Combine into waypoint
                waypoint = torch.cat([position, quaternion, torch.tensor([gripper_status], device='cuda')])
                waypoints.append(waypoint)
    
    # Convert list to torch tensor of shape (1, traj_length, waypoint_dim)
    trajectory_tensor = torch.stack(waypoints).unsqueeze(0)  # Add batch dim
    
    return trajectory_tensor  # Shape: (1, traj_length, waypoint_dim)

def gripper_real2sim(real_gripper_status: float) -> float:
    sim_gripper_joint = (800 - real_gripper_status) / 800
    if sim_gripper_joint < 0:
        sim_gripper_joint = 0

    return sim_gripper_joint

def interpolate_ee_trajectory(ee_init: torch.Tensor, ee_final: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Generate a trajectory between initial and final end-effector poses across multiple environments
    using linear interpolation.

    Args:
        ee_init (torch.Tensor): Initial end-effector poses for each environment, shape (num_envs, 8).
        ee_final (torch.Tensor): Final end-effector poses for each environment, shape (num_envs, 8).
        num_steps (int): Number of steps in the trajectory.

    Returns:
        torch.Tensor: Interpolated trajectories for each environment, shape (num_envs, num_steps, 8).
    """
    if ee_init.shape != ee_final.shape or ee_init.shape[1] != 8:
        raise ValueError("ee_init and ee_final must have the same shape and must be (num_envs, 8).")

    num_envs = ee_init.shape[0]

    # Generate interpolation factors (t), shape: (num_steps,)
    t = torch.linspace(0, 1, num_steps, device=ee_init.device).unsqueeze(0).unsqueeze(-1)  # Shape: (1, num_steps, 1)

    # Interpolate between ee_init and ee_final
    trajectory = ee_init.unsqueeze(1) + t * (ee_final.unsqueeze(1) - ee_init.unsqueeze(1))  # Shape: (num_envs, num_steps, 8)

    return trajectory  # Shape: (num_envs, num_steps, 8)

def slerp_batch(q1: torch.Tensor, q2: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """
    Vectorized SLERP between two batches of quaternions.
    
    Args:
        q1 (torch.Tensor): Batch of quaternions, shape (B, 4)
        q2 (torch.Tensor): Batch of quaternions, shape (B, 4)
        ts (torch.Tensor): 1D tensor of interpolation factors, shape (num_steps,)
    
    Returns:
        torch.Tensor: Interpolated quaternions, shape (B, num_steps, 4)
    """
    # Normalize input quaternions.
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    q2 = q2 / q2.norm(dim=-1, keepdim=True)
    
    # Dot product (cosine of the angle between quaternions).
    cosTheta = torch.sum(q1 * q2, dim=-1, keepdim=True)  # (B, 1)
    
    # Ensure the shortest path is taken.
    mask = (cosTheta < 0)
    q2 = torch.where(mask, -q2, q2)
    cosTheta = torch.where(mask, -cosTheta, cosTheta)
    
    # Expand q1 and q2 for broadcasting: (B, 1, 4)
    q1_exp = q1.unsqueeze(1)
    q2_exp = q2.unsqueeze(1)
    
    # Reshape ts to (1, num_steps, 1) for broadcasting.
    ts_exp = ts.view(1, -1, 1)
    
    # Compute the angle between quaternions.
    theta = torch.acos(torch.clamp(cosTheta, -1.0, 1.0))  # shape: (B, 1)
    theta = theta.unsqueeze(1)  # Now shape: (B, 1, 1)
    sinTheta = torch.sin(theta) + 1e-6  # (B, 1, 1)
    
    # Compute interpolation factors.
    factor1 = torch.sin((1 - ts_exp) * theta) / sinTheta  # (B, num_steps, 1)
    factor2 = torch.sin(ts_exp * theta) / sinTheta        # (B, num_steps, 1)
    
    q_interp = factor1 * q1_exp + factor2 * q2_exp  # (B, num_steps, 4)
    q_interp = q_interp / q_interp.norm(dim=-1, keepdim=True)
    
    return q_interp

def interpolate_10d_ee_trajectory(ee_init: torch.Tensor, ee_final: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Generate a trajectory between initial and final end-effector poses for multiple environments,
    using linear interpolation for position and gripper status and SLERP (via quaternions)
    for orientation.
    
    Each end-effector pose is defined as:
        [position (3D), orientation (6D), gripper_status (1D)] = total 10D.
    
    Args:
        ee_init (torch.Tensor): Initial poses, shape (num_envs, 10).
        ee_final (torch.Tensor): Final poses, shape (num_envs, 10).
        num_steps (int): Number of interpolation steps.
    
    Returns:
        torch.Tensor: Trajectory of shape (num_envs, num_steps, 10)
    """
    if ee_init.shape != ee_final.shape or ee_init.shape[1] != 10:
        raise ValueError("ee_init and ee_final must have the same shape and be (num_envs, 10).")
    
    num_envs = ee_init.shape[0]
    
    # Create interpolation factors: shape (num_steps,)
    ts = torch.linspace(0, 1, steps=num_steps, device=ee_init.device, dtype=ee_init.dtype)
    
    # ---- Interpolate Position (first 3 dims) ----
    pos_init = ee_init[:, :3]   # shape (num_envs, 3)
    pos_final = ee_final[:, :3] # shape (num_envs, 3)
    # (num_envs, 1, 3) broadcast with (1, num_steps, 3)
    pos_traj = pos_init.unsqueeze(1) * (1 - ts.view(1, num_steps, 1)) + pos_final.unsqueeze(1) * ts.view(1, num_steps, 1)
    
    # ---- Interpolate Orientation (next 6 dims) using SLERP ----
    orient6d_init = ee_init[:, 3:9]   # shape (num_envs, 6)
    orient6d_final = ee_final[:, 3:9]   # shape (num_envs, 6)
    
    # Convert 6D representations to quaternions (assumed functions).
    q_init = quat_from_6d(orient6d_init)   # shape (num_envs, 4)
    q_final = quat_from_6d(orient6d_final)   # shape (num_envs, 4)
    
    # Perform batched SLERP.
    q_traj = slerp_batch(q_init, q_final, ts)  # shape (num_envs, num_steps, 4)
    
    # Convert interpolated quaternions back to 6D.
    B, T, _ = q_traj.shape
    q_traj_flat = q_traj.reshape(B * T, 4)
    orient6d_traj_flat = quat_to_6d(q_traj_flat)  # shape (B*T, 6)
    orient6d_traj = orient6d_traj_flat.reshape(B, T, 6)
    
    # ---- Interpolate Gripper Status (last dim) ----
    grip_init = ee_init[:, 9:]   # shape (num_envs, 1)
    grip_final = ee_final[:, 9:]  # shape (num_envs, 1)
    grip_traj = grip_init.unsqueeze(1) * (1 - ts.view(1, num_steps, 1)) + grip_final.unsqueeze(1) * ts.view(1, num_steps, 1)
    
    # ---- Concatenate all components ----
    # pos_traj: (num_envs, num_steps, 3)
    # orient6d_traj: (num_envs, num_steps, 6)
    # grip_traj: (num_envs, num_steps, 1)
    trajectory = torch.cat([pos_traj, orient6d_traj, grip_traj], dim=-1)  # (num_envs, num_steps, 10)
    
    return trajectory

def interpolate_7d_ee_trajectory(ee_init: torch.Tensor, ee_final: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Generate a trajectory between initial and final end-effector poses for multiple environments,
    using linear interpolation for position and SLERP for quaternions.
    
    Each end-effector pose is defined as:
        [position (3D), quaternion (4D)] = total 7D.
    
    Args:
        ee_init (torch.Tensor): Initial poses, shape (num_envs, 7).
        ee_final (torch.Tensor): Final poses, shape (num_envs, 7).
        num_steps (int): Number of interpolation steps.
    
    Returns:
        torch.Tensor: Trajectory of shape (num_envs, num_steps, 7)
    """
    if ee_init.shape != ee_final.shape or ee_init.shape[1] != 7:
        raise ValueError("ee_init and ee_final must have the same shape and be (num_envs, 7).")
    
    num_envs = ee_init.shape[0]
    
    # Create interpolation factors: shape (num_steps,)
    ts = torch.linspace(0, 1, steps=num_steps, device=ee_init.device, dtype=ee_init.dtype)
    
    # ---- Interpolate Position (first 3 dims) ----
    pos_init = ee_init[:, :3]   # shape (num_envs, 3)
    pos_final = ee_final[:, :3] # shape (num_envs, 3)
    # (num_envs, 1, 3) broadcast with (1, num_steps, 3)
    pos_traj = pos_init.unsqueeze(1) * (1 - ts.view(1, num_steps, 1)) + pos_final.unsqueeze(1) * ts.view(1, num_steps, 1)
    
    # ---- Interpolate quat using SLERP ----
    q_init = ee_init[:, 3:7]    # shape (num_envs, 4)
    q_final = ee_final[:, 3:7]   # shape (num_envs, 4)
    
    # Perform batched SLERP.
    q_traj = slerp_batch(q_init, q_final, ts)  # shape (num_envs, num_steps, 4)
    
    # ---- Concatenate all components ----
    trajectory = torch.cat([pos_traj, q_traj], dim=-1)  # (num_envs, num_steps, 7)
    
    return trajectory

def generate_demo_ee_trajectory(ee_traj_storage: torch.Tensor, env_ids: torch.Tensor, ee_init: torch.Tensor, ee_reach: torch.Tensor, traj_length: int = 400) -> torch.Tensor:
    """
    Generate a demonstration trajectory for specific environments and store them in a pre-allocated tensor.

    Args:
        ee_traj_storage (torch.Tensor): Pre-allocated tensor to store trajectories, shape (num_envs, 8*num_steps).
        env_ids (torch.Tensor): Indices of environments to generate trajectories for, shape (num_selected_envs,).
        ee_init (torch.Tensor): Initial end-effector poses for selected environments, shape (num_selected_envs, 8).
        ee_reach (torch.Tensor): Reach end-effector poses for selected environments, shape (num_selected_envs, 8).
        traj_length (int): Total length of the trajectory.

    Returns:
        torch.Tensor: Updated ee_traj_storage with generated trajectories for specified env_ids.
    """
    # Break the trajectory into parts: reach, lift up, grasping
    reaching_idx = 120
    grasping_idx = 80
    lift_up_idx = traj_length - reaching_idx - grasping_idx

    # Compute intermediate end-effector poses
    # 0.51 for expert, 0.4 for expert_bar
    ee_grasp = ee_reach + torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0.5]], device=ee_reach.device)  # Close gripper
    ee_lift_up = ee_grasp + torch.tensor([[0, 0, 0.3, 0, 0, 0, 0, 0]], device=ee_grasp.device)  # Lift up

    # Generate trajectories for each part
    ee_traj_reach = interpolate_ee_trajectory(ee_init, ee_reach, reaching_idx)  # Shape: (num_selected_envs, reaching_idx * 8)
    ee_traj_grasp = interpolate_ee_trajectory(ee_reach, ee_grasp, grasping_idx)  # Shape: (num_selected_envs, grasping_idx * 8)
    ee_traj_pick = interpolate_ee_trajectory(ee_grasp, ee_lift_up, lift_up_idx)  # Shape: (num_selected_envs, lift_up_idx * 8)

    # Concatenate trajectories
    ee_traj = torch.cat((ee_traj_reach, ee_traj_grasp, ee_traj_pick), dim=1)  # Shape: (num_selected_envs, 8*num_steps)

    # Update the storage tensor for specified environments
    ee_traj_storage[env_ids] = ee_traj

    return ee_traj_storage

def smooth_noisy_trajectory(trajectory, env_ids, step_interval=10, noise_level=0.01, beta_filter=0.7):
    """
    Add noise to a trajectory and smooth it using cubic interpolation for each demo.

    Args:
        trajectory (torch.Tensor): Trajectory of shape (num_envs, num_demo, traj_length, action_dim).
        env_ids (torch.Tensor): Indices of environments to apply the method.
        step_interval (int): Interval at which to add noise.
        noise_level (float): Magnitude of the noise.
        beta_filter (float): Temporal correlation factor for the noise.

    Returns:
        torch.Tensor: Smoothed trajectory of shape (num_envs, num_demo, traj_length, action_dim).
    """
    num_envs, num_demos, traj_length, action_dim = trajectory.shape

    smoothed_trajs = []
    
    # Process each demo independently
    for demo in range(num_demos):
        # Step 1: Add noise
        noised_nodes, noised_steps = add_noise_to_nodes(
            trajectory[:, demo, :, :], env_ids, step_interval, noise_level, beta_filter
        )

        # Step 2: Reconnect using cubic spline interpolation
        smoothed_demo = reconnect_with_cubic_spline(
            trajectory[:, demo, :, :], noised_nodes, noised_steps, traj_length, step_interval, env_ids
        )

        smoothed_trajs.append(smoothed_demo)  # Each smoothed_demo is (num_envs, traj_length, action_dim)

    # Stack across demos
    smoothed_trajectory = torch.stack(smoothed_trajs, dim=1)  # Shape: (num_envs, num_demo, traj_length, action_dim)

    return smoothed_trajectory  # Correct shape

def add_noise_to_nodes(trajectory, env_ids, step_interval=10, noise_level=0.01, beta_filter=0.7):
    """
    Add normal noise to the trajectory at every `step_interval` step for specified environments.

    Args:
        trajectory (torch.Tensor): Original trajectory of shape (num_envs, time_step, action_dim).
        env_ids (torch.Tensor): Tensor containing indices of environments to apply the method.
        step_interval (int): Interval at which to add noise.
        noise_level (float): Magnitude of the noise.
        beta_filter (float): Temporal correlation factor for the noise.

    Returns:
        torch.Tensor: Noised nodes of shape (num_envs, num_noised_steps, action_dim).
        torch.Tensor: Indices of the noised steps.
    """
    num_envs, time_step, action_dim = trajectory.shape
    device = trajectory.device

    # Steps where noise is added
    noised_steps = torch.arange(0, time_step, step_interval, device=device)

    # Extract nodes at noised steps
    noised_nodes = trajectory[:, noised_steps, :].clone() # (num_envs, num_noised_steps, action_dim)

    # Residual noise buffer
    act_residual = torch.zeros((num_envs, action_dim), dtype=trajectory.dtype, device=device)

    # Apply noise to specified environments
    for i in range(len(noised_steps)):
        noise_sample = torch.normal(0, noise_level, (len(env_ids), action_dim), device=device)
        act_residual[env_ids] = beta_filter * noise_sample + (1 - beta_filter) * act_residual[env_ids]
        noised_nodes[env_ids, i, :] += act_residual[env_ids]

    # Constrain specific dimensions (e.g., z-axis for robotic tasks)
    noised_nodes[env_ids, :, 2] = torch.clamp(noised_nodes[env_ids, :, 2], 0.17, 0.4)

    return noised_nodes, noised_steps


def reconnect_with_cubic_spline(original_trajectory, noised_nodes, noised_steps, time_step, step_interval, env_ids):
    """
    Reconnect noised nodes into a smooth trajectory using cubic spline interpolation.

    Args:
        original_trajectory (torch.Tensor): Original trajectory of shape (num_envs, time_step, action_dim).
        noised_nodes (torch.Tensor): Noised nodes of shape (num_envs, num_noised_steps, action_dim).
        noised_steps (torch.Tensor): Indices of noised nodes in the trajectory.
        time_step (int): Total number of steps in the trajectory.
        step_interval (int): Interval at which noise was added.
        env_ids (torch.Tensor): Tensor containing indices of environments to apply the method.

    Returns:
        torch.Tensor: Smoothed trajectory of shape (num_envs, time_step, action_dim).
    """
    num_envs, _, action_dim = original_trajectory.shape
    device = original_trajectory.device

    # Compute cubic spline second derivatives for noised nodes (only for env_ids)
    M = cubic_spline_nd_torch_batched(noised_nodes[env_ids])

    # Query timesteps for interpolation
    query_steps = torch.arange(0, time_step, device=device).unsqueeze(0).repeat(len(env_ids), 1)

    # Normalize steps for cubic spline
    normalized_query_steps = query_steps.float() / step_interval

    # Evaluate the cubic spline
    smoothed_subset = eval_cubic_spline_nd_torch_batched(noised_nodes[env_ids], M, normalized_query_steps)

    # Replace the smoothed environments in the original trajectory
    smoothed_trajectory = original_trajectory.clone()
    smoothed_trajectory[env_ids] = smoothed_subset  # Only modify the specified environments

    return smoothed_trajectory

def cubic_spline_nd_torch_batched(points: torch.Tensor) -> torch.Tensor:
    """Compute second derivatives for a batched natural cubic spline.

    Args:
        points (torch.Tensor): Tensor of shape (B, K, D), where
            - B = batch size,
            - K = number of knots (control points),
            - D = number of dimensions.

    Returns:
        torch.Tensor: Second derivatives (M) of shape (B, K, D).
    """
    B, K, D = points.shape

    if K <= 2:
        return torch.zeros_like(points)

    points_flat = points.permute(0, 2, 1).reshape(-1, K)  # Flatten to (B*D, K)

    M_flat = torch.zeros_like(points_flat)
    alpha = torch.zeros_like(points_flat)

    # Compute alpha for Thomas algorithm
    alpha[:, 1 : K - 1] = 6.0 * (
        points_flat[:, 2:] - 2.0 * points_flat[:, 1:-1] + points_flat[:, :-2]
    )

    # Solve using Thomas algorithm
    l = torch.zeros_like(points_flat)
    mu = torch.zeros_like(points_flat)
    z = torch.zeros_like(points_flat)

    l[:, 0] = 1.0
    for i in range(1, K - 1):
        l[:, i] = 4.0 - mu[:, i - 1]
        mu[:, i] = 1.0 / l[:, i]
        z[:, i] = (alpha[:, i] - z[:, i - 1]) / l[:, i]

    # Back-substitution
    for i in range(K - 2, 0, -1):
        M_flat[:, i] = z[:, i] - mu[:, i] * M_flat[:, i + 1]

    return M_flat.view(B, D, K).permute(0, 2, 1)

def eval_cubic_spline_nd_torch_batched(points: torch.Tensor, M: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """Evaluate a batched cubic spline at given parameters.

    Args:
        points (torch.Tensor): Shape (B, K, D), where K is the number of control points.
        M (torch.Tensor): Second derivatives (B, K, D).
        ts (torch.Tensor): Query times of shape (B, N).

    Returns:
        torch.Tensor: Evaluated spline of shape (B, N, D).
    """
    B, K, D = points.shape

    if K == 1:
        return points[:, 0:1, :].expand(B, ts.shape[1], D)

    i_clamped = torch.clamp(torch.floor(ts).long(), 0, K - 2)
    mu = ts - i_clamped.float()

    gather_idx = i_clamped.unsqueeze(-1).expand(-1, -1, D)
    y_i = torch.gather(points, dim=1, index=gather_idx)
    y_ip1 = torch.gather(points, dim=1, index=gather_idx + 1)

    M_i = torch.gather(M, dim=1, index=gather_idx)
    M_ip1 = torch.gather(M, dim=1, index=gather_idx + 1)

    mu_3d = mu.unsqueeze(-1)
    one_minus_mu_3d = 1.0 - mu_3d

    vals = (
        (one_minus_mu_3d**3 * M_i + mu_3d**3 * M_ip1) / 6
        + (y_i - M_i / 6) * one_minus_mu_3d
        + (y_ip1 - M_ip1 / 6) * mu_3d
    )
    return vals

def cubic_spline_nd_function_torch(points: torch.Tensor,
    ) -> Callable[[float], torch.Tensor]:
    """Create a function to evaluate a natural cubic spline at any parameter t

    Given a set of points in D dimensions, precompute the second derivatives
    for a natural cubic spline, and return a function that can evaluate the spline
    at any parameter t in [0, N-1].

    Args:
        points: (B, K, D) tensor of K points in D dimensions, for B separate spline
        problems.

    Returns:
        A function that takes a parameter t in [0, N-1] and returns the spline value at
    """
    # 1) Precompute second derivatives in each dimension
    M = cubic_spline_nd_torch_batched(points)

    # 2) Return a closure that evaluates at any t in [0, N-1]
    def spline_func(t: float) -> torch.Tensor:
        return eval_cubic_spline_nd_torch_batched(points, M, t) # type: ignore

    return spline_func

def format_tensor(tensor, precision=3):
    if tensor.dim() == 0:  # Handle scalar tensor
        print(f"{tensor.item():.{precision}f}")
    elif tensor.dim() == 1:  # Handle 1D tensor
        formatted = [f"{value:.{precision}f}" for value in tensor.cpu().tolist()]
        print(formatted)
    elif tensor.dim() == 2:  # Handle 2D tensor
        formatted_rows = [
            [f"{value:.{precision}f}" for value in row.cpu().tolist()] for row in tensor
        ]
        for row in formatted_rows:
            print(row)
    else:
        print("Tensor with more than 2 dimensions is not supported.")

def subtract_z_in_baseframe_batch(ee_pose, z_offset=0.15):
    """
    Subtracts a given offset along the base frame's z-axis from the end-effector position
    for multiple environments.

    Args:
        ee_pose (torch.Tensor): End-effector pose (position + quaternion) in shape (num_envs, 7).
        z_offset (float): Distance to subtract along the base frame's z-axis (default -0.15m).

    Returns:
        torch.Tensor: Adjusted end-effector positions after subtracting z-offset, shape (num_envs, 3).
    """
    num_envs = ee_pose.shape[0]

    position = ee_pose[:, :3]
    quaternion = ee_pose[:, 3:]

    R = matrix_from_quat(quaternion)

    displacement_base = torch.tensor([0.0, 0.0, z_offset], dtype=ee_pose.dtype, device=ee_pose.device).expand(num_envs, 3)

    # Transform displacement into each environment’s EE frame
    displacement_ee = torch.bmm(R, displacement_base.unsqueeze(-1)).squeeze(-1)  # (num_envs, 3)

    # Adjust position
    new_position = position + displacement_ee  # (num_envs, 3)
    
    return new_position


def compute_relative_state(prev_state: torch.Tensor, curr_state: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative robot state (current state expressed in the previous state's frame).

    The function supports two state types:
    
    1. 10-dimensional state: [position (3), orientation (6D) (columns 3-8), gripper (1)].
    2. 16-dimensional state: [position (3), orientation (6D) (columns 3-8),
                               linear velocity (3) (columns 9-11),
                               angular velocity (3) (columns 12-14), gripper (1) (column 15)].
    
    For both cases, the relative state is computed as follows:
      - Relative position:  p_rel = R_prev^T (p_curr - p_prev)
      - Relative orientation: R_rel = R_prev^T * R_curr, converted back to a 6D representation.
      - For the 16-dimensional state, the linear and angular velocities are also rotated:
          v_lin_rel = R_prev^T (v_lin_curr - v_lin_prev)
          v_ang_rel = R_prev^T (v_ang_curr - v_ang_prev)
      - The gripper value is taken from the current state.
      
    The output will have the same dimensionality as the input state.

    Args:
        prev_state (torch.Tensor): Previous state, shape (num_envs, 10) or (num_envs, 16).
        curr_state (torch.Tensor): Current state, shape (num_envs, 10) or (num_envs, 16).

    Returns:
        torch.Tensor: Relative state, of the same shape as the input states.
    """
    # Check that both states have the same dimension.
    if prev_state.shape[1] != curr_state.shape[1]:
        raise ValueError("prev_state and curr_state must have the same dimensionality.")
    
    state_dim = prev_state.shape[1]
    if state_dim not in (10, 16):
        raise ValueError("State dimensionality must be either 10 or 16.")

    # Extract common components: position, orientation, gripper.
    prev_pos = prev_state[:, :3]  # (num_envs, 3)
    curr_pos = curr_state[:, :3]  # (num_envs, 3)
    
    # Orientation is stored as 6D (columns 3 to 8).
    prev_R = sixd_to_rotation_matrix(prev_state[:, 3:9])  # (num_envs, 3, 3)
    curr_R = sixd_to_rotation_matrix(curr_state[:, 3:9])  # (num_envs, 3, 3)
    
    # Gripper is the last element.
    prev_gripper = prev_state[:, -1]  # (num_envs,)
    curr_gripper = curr_state[:, -1]  # (num_envs,)

    # Compute the inverse of the previous rotation (transpose, since R is orthogonal).
    prev_R_T = prev_R.transpose(1, 2)  # (num_envs, 3, 3)

    # Compute relative position: rotate the difference into the previous frame.
    delta_pos = curr_pos - prev_pos
    rel_pos = torch.bmm(prev_R_T, delta_pos.unsqueeze(-1)).squeeze(-1)  # (num_envs, 3)

    # Compute relative orientation: R_rel = R_prev^T * R_curr, then convert to 6D.
    rel_R = torch.bmm(prev_R_T, curr_R)  # (num_envs, 3, 3)
    rel_orient = rotation_matrix_to_6d(rel_R)  # (num_envs, 6)

    if state_dim == 16:
        # Extract velocities.
        prev_lin_vel = prev_state[:, 9:12]    # (num_envs, 3)
        prev_ang_vel = prev_state[:, 12:15]    # (num_envs, 3)
        curr_lin_vel = curr_state[:, 9:12]     # (num_envs, 3)
        curr_ang_vel = curr_state[:, 12:15]     # (num_envs, 3)
        
        # Compute relative velocities by rotating the differences.
        delta_lin_vel = curr_lin_vel - prev_lin_vel
        rel_lin_vel = torch.bmm(prev_R_T, delta_lin_vel.unsqueeze(-1)).squeeze(-1)  # (num_envs, 3)
        
        delta_ang_vel = curr_ang_vel - prev_ang_vel
        rel_ang_vel = torch.bmm(prev_R_T, delta_ang_vel.unsqueeze(-1)).squeeze(-1)  # (num_envs, 3)
        
        # Assemble relative state for 16-dimensional input:
        # [rel_pos (3), rel_orient (6), rel_lin_vel (3), rel_ang_vel (3), gripper (1)]
        rel_state = torch.cat([
            rel_pos,
            rel_orient,
            rel_lin_vel,
            rel_ang_vel,
            curr_gripper.unsqueeze(1)
        ], dim=1)  # (num_envs, 16)
    else:
        # Assemble relative state for 10-dimensional input:
        # [rel_pos (3), rel_orient (6), gripper (1)]
        rel_state = torch.cat([
            rel_pos,
            rel_orient,
            curr_gripper.unsqueeze(1)
        ], dim=1)  # (num_envs, 10)

    return rel_state
