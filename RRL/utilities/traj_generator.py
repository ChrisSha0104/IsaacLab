import torch
from omni.isaac.lab.utils.math import sample_uniform, euler_xyz_from_quat, quat_from_euler_xyz, quat_from_matrix, subtract_frame_transforms, quat_mul
import os
from typing import Callable

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


def resample_trajectory(trajectory: torch.Tensor, goal_traj_length: int, waypoint_dim=8) -> torch.Tensor:
    """
    Resample a trajectory to a specified length using linear interpolation.

    Args:
        trajectory (torch.Tensor): Input trajectory tensor of shape (num_envs, current_traj_length * waypoint_dim).
        goal_traj_length (int): The desired trajectory length.
        waypoint_dim (int): The dimensionality of each waypoint.

    Returns:
        torch.Tensor: Resampled trajectory of shape (num_envs, goal_traj_length * waypoint_dim).
    """
    num_envs, total_dim = trajectory.shape
    current_traj_length = total_dim // waypoint_dim  # Compute original trajectory length

    # Reshape into (num_envs, current_traj_length, waypoint_dim)
    trajectory = trajectory.view(num_envs, current_traj_length, waypoint_dim)

    # Apply linear interpolation along the trajectory length axis
    resampled_trajectory = torch.nn.functional.interpolate(
        trajectory.permute(0, 2, 1),  # Change shape to (num_envs, waypoint_dim, current_traj_length)
        size=goal_traj_length,
        mode='linear',
        align_corners=True
    )

    # Reshape back to (num_envs, goal_traj_length, waypoint_dim) and flatten
    resampled_trajectory = resampled_trajectory.permute(0, 2, 1).reshape(num_envs, -1)

    return resampled_trajectory

def postprocess_real_demo_trajectory(quat_diff, folder_path):
    """
    Process a folder containing robot end-effector trajectory waypoints stored in .txt files.

    Each file contains the position, rotation matrix, and gripper status, which are extracted and converted
    into a structured tensor.

    Args:
        folder_path (str): Path to the folder containing the trajectory files.

    Returns:
        torch.Tensor: A concatenated tensor of shape (1, traj_length * waypoint_dim), where each waypoint
                    consists of [position (3), quaternion (4), gripper status (1)].
    """
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
                quaternion = quat_mul(quat_diff.squeeze(0), quaternion)
                
                # Extract gripper status
                real_gripper_status = float(lines[8].strip().split()[0])
                gripper_status = gripper_real2sim(real_gripper_status)
                
                # Combine into waypoint
                waypoint = torch.cat([position, quaternion, torch.tensor([gripper_status], device='cuda')])
                waypoints.append(waypoint)
    
    # Convert list to torch tensor of shape (1, traj_length * waypoint_dim)
    trajectory_tensor = torch.stack(waypoints).view(1, -1)
    
    return trajectory_tensor


def gripper_real2sim(real_gripper_status: float) -> float:
    """
    Maps the real robot's gripper status (800 open, 0 closed) to the simulator's (0.0 open, 0.84 closed)
    using a quadratic function.

    Args:
        real_gripper_status (float): Gripper status from the real robot (range: 800 to 0).

    Returns:
        float: Mapped gripper status for the simulator (range: 0.0 to 0.84).
    """
    a, b, c = -1.3522e-6, 3.1781e-5, 0.84
    return a * real_gripper_status**2 + b * real_gripper_status + c

    # return (800 - real_gripper_status) / 800


def interpolate_ee_trajectory(ee_init: torch.Tensor, ee_final: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Generate a trajectory between initial and final end-effector poses across multiple environments
    using linear interpolation.

    Args:
        ee_init (torch.Tensor): Initial end-effector poses for each environment, shape (num_envs, 8).
        ee_final (torch.Tensor): Final end-effector poses for each environment, shape (num_envs, 8).
        num_steps (int): Number of steps in the trajectory.

    Returns:
        torch.Tensor: Interpolated trajectories for each environment, shape (num_envs, num_steps * 8).
    """
    if ee_init.shape != ee_final.shape or ee_init.shape[1] != 8:
        raise ValueError("ee_init and ee_final must have the same shape and must be (num_envs, 8).")

    num_envs = ee_init.shape[0]

    # Generate interpolation factors (t), shape: (num_steps,)
    t = torch.linspace(0, 1, num_steps, device=ee_init.device).unsqueeze(0)  # Shape: (1, num_steps)

    # Reshape t to allow broadcasting with (num_envs, 8)
    t = t.unsqueeze(-1)  # Shape: (1, num_steps, 1)

    # Interpolate between ee_init and ee_final
    trajectory = ee_init.unsqueeze(1) + t * (ee_final.unsqueeze(1) - ee_init.unsqueeze(1))  # Shape: (num_envs, num_steps, 8)

    # Reshape the trajectory to (num_envs, num_steps * 8)
    trajectory = trajectory.reshape(num_envs, num_steps * 8)

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
        trajectory (torch.Tensor): Trajectory of shape (num_envs, n_look_ahead*action_dim, num_demos).
        env_ids (torch.Tensor): Indices of environments to apply the method.
        step_interval (int): Interval at which to add noise.
        noise_level (float): Magnitude of the noise.
        beta_filter (float): Temporal correlation factor for the noise.

    Returns:
        torch.Tensor: Smoothed trajectory of shape (num_envs, n_look_ahead*action_dim, num_demos).
    """
    num_envs, traj_length_action, num_demos = trajectory.shape
    traj = trajectory.reshape(num_envs, -1, 8, num_demos)  # Shape: (num_envs, n_look_ahead, action_dim, num_demos)

    n_look_ahead = traj.size(1)

    smoothed_trajs = []
    
    # Process each demo independently
    for demo in range(num_demos):
        # Step 1: Add noise
        noised_nodes, noised_steps = add_noise_to_nodes(
            traj[..., demo], env_ids, step_interval, noise_level, beta_filter
        )

        # Step 2: Reconnect using cubic spline interpolation
        smoothed_demo = reconnect_with_cubic_spline(
            traj[..., demo], noised_nodes, noised_steps, n_look_ahead, step_interval, env_ids
        )

        smoothed_trajs.append(smoothed_demo)

    # Stack across demos
    smoothed_trajectory = torch.stack(smoothed_trajs, dim=-1)  # Shape: (num_envs, n_look_ahead, action_dim, num_demos)
    smoothed_trajectory = smoothed_trajectory.reshape(num_envs, traj_length_action, num_demos)

    return smoothed_trajectory


def add_noise_to_nodes(trajectory, env_ids, step_interval=10, noise_level=0.01, beta_filter=0.7):
    """
    Add normal noise to the trajectory at every `step_interval` step for specified environments.

    Args:
        trajectory (torch.Tensor): Original trajectory of shape (num_envs, n_look_ahead, action_dim).
        env_ids (torch.Tensor): Tensor containing indices of environments to apply the method.
        step_interval (int): Interval at which to add noise.
        noise_level (float): Magnitude of the noise.
        beta_filter (float): Temporal correlation factor for the noise.

    Returns:
        torch.Tensor: Noised nodes of shape (num_envs, n_noised_steps, action_dim).
    """
    num_envs, n_look_ahead, action_dim = trajectory.shape
    
    device = trajectory.device

    # Indices of the steps where noise is added
    noised_steps = torch.arange(0, n_look_ahead, step_interval, device=device)

    # Initialize the noised trajectory
    noised_nodes = trajectory[:, noised_steps, :].clone()

    # Initialize residual noise
    act_residual = torch.zeros((num_envs, action_dim), dtype=trajectory.dtype, device=device)

    # Add noise to specified env_ids only
    for i in range(len(noised_steps)):
        noise_sample = torch.normal(0, noise_level, (len(env_ids), action_dim), device=device)
        act_residual[env_ids] = beta_filter * noise_sample + (1 - beta_filter) * act_residual[env_ids]
        noised_nodes[env_ids, i, :] += act_residual[env_ids]

    noised_nodes[env_ids, :, 2] = torch.clamp(noised_nodes[env_ids, :, 2], 0.18, 0.4)

    return noised_nodes, noised_steps

def reconnect_with_cubic_spline(original_trajectory, noised_nodes, noised_steps, n_look_ahead, step_interval, env_ids):
    """
    Reconnect noised nodes into a smooth trajectory using cubic spline interpolation for specified environments.

    Args:
        original_trajectory (torch.Tensor): Original trajectory of shape (num_envs, n_look_ahead, action_dim).
        noised_nodes (torch.Tensor): Noised nodes of shape (num_envs, n_noised_steps, action_dim).
        noised_steps (torch.Tensor): Indices of noised nodes in the trajectory.
        n_look_ahead (int): Total number of steps in the trajectory.
        step_interval (int): Interval at which noise was added.
        env_ids (torch.Tensor): Tensor containing indices of environments to apply the method.

    Returns:
        torch.Tensor: Smoothed trajectory of shape (num_envs, n_look_ahead, action_dim).
    """
    num_envs, _, action_dim = original_trajectory.shape
    device = original_trajectory.device

    # Compute cubic spline second derivatives for the noised nodes (only for env_ids)
    M = cubic_spline_nd_torch_batched(noised_nodes[env_ids])

    # Create query timesteps
    query_steps = torch.arange(0, n_look_ahead, device=device).unsqueeze(0).repeat(len(env_ids), 1)

    # Normalize query steps relative to the noised steps
    normalized_query_steps = query_steps.float() / step_interval

    # Evaluate the cubic spline at all query steps for env_ids
    smoothed_subset = eval_cubic_spline_nd_torch_batched(noised_nodes[env_ids], M, normalized_query_steps)

    # Initialize smoothed trajectory as original and replace env_ids with smoothed data
    smoothed_trajectory = original_trajectory.clone()
    smoothed_trajectory[env_ids] = smoothed_subset

    return smoothed_trajectory

def cubic_spline_nd_torch_batched(points: torch.Tensor) -> torch.Tensor:
    """Compute cubic splines

    Compute the second-derivatives (M) for a natural cubic spline in D dimensions,
    in a *batched* manner, without looping over batch or dimension axes.
    The only loop is over the knot index K for the Thomas algorithm.

    Parameters
    ----------
    points : torch.Tensor
        A (B, K, D) tensor of data points, where:
        B = batch size (number of spline problems),
        K = number of knots,
        D = dimension of each point.

    Returns
    -------
    M : torch.Tensor
        A (B, K, D) tensor of second derivatives for each batch and dimension.
        Natural boundary conditions (M[..., 0] = M[..., -1] = 0).
    """
    B, K, D = points.shape

    # If K <= 2, all second derivatives are zero.
    if K <= 2:
        return torch.zeros_like(points)

    # 1) Flatten from (B, K, D) => (B*D, K)
    #    Each row in the flattened array corresponds to one 1D spline problem.
    points_flat = points.permute(0, 2, 1)  # shape (B, D, K)
    points_flat = points_flat.reshape(-1, K)  # shape (B*D, K)

    # We'll solve for M_flat in shape (B*D, K).
    M_flat = torch.zeros_like(points_flat)  # same shape as points_flat

    # 2) Build alpha = 6*(y[i+1] - 2*y[i] + y[i-1]) for i in [1..K-2]
    #    We'll put alpha in a (B*D, K) array, with alpha[:, 0] and alpha[:, K-1] unused
    alpha = torch.zeros_like(points_flat)
    # Vectorized assignment for i=1..K-2
    alpha[:, 1 : K - 1] = 6.0 * (
        points_flat[:, 2:] - 2.0 * points_flat[:, 1:-1] + points_flat[:, :-2]
    )

    # 3) Prepare arrays l, mu, z of shape (B*D, K) for the Thomas algorithm
    l = torch.zeros_like(points_flat)  # noqa
    mu = torch.zeros_like(points_flat)
    z = torch.zeros_like(points_flat)

    # Boundary conditions: M[0] = 0 => l[0] = 1, z[0] = 0
    l[:, 0] = 1.0
    mu[:, 0] = 0.0
    z[:, 0] = 0.0

    # 4) Decomposition pass (loop over K dimension only)
    for i in range(1, K - 1):
        l[:, i] = 4.0 - mu[:, i - 1]
        mu[:, i] = 1.0 / l[:, i]
        z[:, i] = (alpha[:, i] - z[:, i - 1]) / l[:, i]

    # Boundary at the end
    l[:, K - 1] = 1.0
    z[:, K - 1] = 0.0

    # 5) Back-substitution pass
    for i in range(K - 2, 0, -1):
        M_flat[:, i] = z[:, i] - mu[:, i] * M_flat[:, i + 1]

    # M_flat[:, 0] and M_flat[:, K-1] remain zero => natural boundary
    #   (which is already the case by default initialization).

    # 6) Reshape back to (B, K, D)
    M = M_flat.view(B, D, K).permute(0, 2, 1)  # => (B, K, D)
    return M


def eval_cubic_spline_nd_torch_batched(points: torch.Tensor, M: torch.Tensor, ts: torch.Tensor
    ) -> torch.Tensor:
    """Evaluate batched natural cubic splines

    Evaluate batched natural cubic splines (in D dimensions) at multiple parameters,
    without looping over batch or dimension.

    Parameters
    ----------
    points : (B, K, D) torch.Tensor
        The knot points for each of the B splines.
    M : (B, K, D) torch.Tensor
        The second derivatives for each spline (same shape as points).
    ts : (B, N) torch.Tensor
        Each row has N query parameters at which to evaluate the corresponding spline.

    Returns
    -------
    vals : (B, N, D) torch.Tensor
        Spline evaluations. For each batch b, we evaluate the b-th spline
        at all the ts[b, :], yielding shape (N, D). Stacked into shape (B, N, D).
    """
    B, K, D = points.shape
    # Edge case: if K == 1, everything collapses to the single point
    if K == 1:
        # shape => (B, 1, D) repeated N times => (B, N, D)
        return points[:, 0:1, :].expand(B, ts.shape[1], D)

    # 1) Find the interval indices: i = floor(ts)
    #    Then clamp them so 0 <= i <= K-2
    i = torch.floor(ts).long()  # shape (B, N)
    i_clamped = torch.clamp(i, min=0, max=K - 2)

    # 2) Compute mu = fractional part
    mu = ts - i_clamped.float()  # shape (B, N)

    # 3) Gather y_i, y_{i+1}, M_i, M_{i+1} using advanced indexing
    #    We build an index of shape (B, N, D) to select from dimension=1 of points.
    gather_idx = i_clamped.unsqueeze(-1).expand(-1, -1, D)  # (B, N, D)

    # shape (B, N, D)
    y_i = torch.gather(points, dim=1, index=gather_idx)
    y_ip1 = torch.gather(points, dim=1, index=gather_idx + 1)
    M_i = torch.gather(M, dim=1, index=gather_idx)
    M_ip1 = torch.gather(M, dim=1, index=gather_idx + 1)

    # 4) Broadcast mu from (B, N) to (B, N, 1) so arithmetic aligns along D
    mu_3d = mu.unsqueeze(-1)  # (B, N, 1)
    one_minus_mu_3d = 1.0 - mu_3d

    # 5) Apply the natural cubic spline formula (uniform spacing h=1):
    #    S(t) = ((1 - mu)^3 * M_i + mu^3 * M_ip1) / 6
    #           + (y_i - M_i / 6) * (1 - mu)
    #           + (y_ip1 - M_ip1 / 6) * mu
    #    All shapes => (B, N, D). No Python loop over B or D.

    mu_cubed = mu_3d**3
    one_minus_mu_cubed = one_minus_mu_3d**3

    term_1 = (one_minus_mu_cubed * M_i + mu_cubed * M_ip1) / 6.0
    term_2 = (y_i - M_i / 6.0) * one_minus_mu_3d
    term_3 = (y_ip1 - M_ip1 / 6.0) * mu_3d

    vals = term_1 + term_2 + term_3  # shape (B, N, D)
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
        return eval_cubic_spline_nd_torch_batched(points, M, t)

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