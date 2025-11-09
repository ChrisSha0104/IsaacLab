import torch
import numpy as np

def quat_geodesic_angle(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-8):
    """
    q1, q2: (..., 4) float tensors in [w, x, y, z] or [x, y, z, w]—either is fine
            as long as both use the same convention.
    Returns: (...,) radians in [0, pi]
    """
    # normalize
    q1 = q1 / (q1.norm(dim=-1, keepdim=True).clamp_min(eps))
    q2 = q2 / (q2.norm(dim=-1, keepdim=True).clamp_min(eps))

    # dot, handle sign ambiguity
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(-1 + eps, 1 - eps)

    return 2.0 * torch.arccos(dot)

class NearestNeighborBuffer:
    """
    Batched nearest-neighbor action retriever with horizon caching.
    Each call returns one action (N,8), reusing pre-fetched horizon actions.
    """

    def __init__(self, path: str, num_envs: int, horizon: int = 15):
        # Load .npz or .pt dataset (expects obs.* and action.* keys per episode)
        flat = np.load(path, allow_pickle=True)
        flat = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in flat.items()}

        # Group by episode prefix (e.g., "episode_0000/")
        episodes = sorted({k.split("/", 1)[0] for k in flat})
        data = {e: {s.split("/", 1)[1]: flat[s] for s in flat if s.startswith(e)} for e in episodes}

        # Compute max episode length and pad all to equal T
        lengths = torch.tensor([len(data[e]["obs.gripper"]) for e in episodes])
        T = int(lengths.max())

        self._max_episode_length = T
        self._total_episodes = len(episodes)
        self._num_envs = num_envs

        def pad(key, dim):
            out = torch.full((len(episodes), T, dim), float("nan"))
            for i, e in enumerate(episodes):
                x = data[e][key]
                out[i, :x.shape[0]] = x
            return out

        # Store obs and action tensors (E,T,dim)
        self._obs_pos, self._obs_quat, self._obs_grip = pad("obs.eef_pos",3), pad("obs.eef_quat",4), pad("obs.gripper",1)
        self._act_pos, self._act_quat, self._act_grip = pad("action.eef_pos",3), pad("action.eef_quat",4), pad("action.gripper",1)

        # Valid timestep mask (E,T)
        self._mask = torch.zeros((len(episodes), T), dtype=torch.bool)
        for i, L in enumerate(lengths): self._mask[i, :int(L)] = True

        # Shared horizon cache
        self._horizon = int(horizon)
        self._queued = None
        self._q_ptr = self._horizon  # force refill first time

        print(f"Loaded NearestNeighborBuffer with {self._total_episodes} episodes; max length {self._max_episode_length}.")

    def get_total_episodes(self) -> int:
        return self._total_episodes

    def get_max_episode_length(self) -> int:
        return int(self._max_episode_length)

    def _nn_indices(self, ep_idx, pos, quat=None, grip=None, verbose = False):
        """
        Find nearest neighbor timestep per env using position (+optional quat/grip).
        dist = pos_cm / 10 + ang_deg / 10 + 2 * grip_L1
        """
        dev = pos.device
        obs_p, obs_q, obs_g = self._obs_pos.to(dev)[ep_idx], self._obs_quat.to(dev)[ep_idx], self._obs_grip.to(dev)[ep_idx]
        mask = self._mask.to(dev)[ep_idx]

        # Base distance: position (m→cm)
        dist = 10 * torch.norm(obs_p - pos[:, None, :], dim=-1)  # (N,T)
        if verbose:
            print("pos_cm / 10 mean:", dist.nanmean().item())

        # Optional quaternion term (deg / 10)
        if quat is not None:
            ang = torch.rad2deg(quat_geodesic_angle(obs_q, quat[:, None, :])) / 10
            if verbose:
                print("ang_deg / 10 mean:", ang.nanmean().item())
            dist += ang

        # Optional gripper L1 term (normalized)
        if grip is not None:
            d_grip = 2 * (obs_g.squeeze(-1) - grip.view(-1,1)).abs()
            if verbose:
                print("grip_L1 * 2 mean:", d_grip.nanmean().item())
            dist += d_grip

        # Mask invalid timesteps
        dist = dist.masked_fill(~mask, float('inf'))
        dist = torch.nan_to_num(dist, nan=float('inf'))

        # Argmin index and valid length
        t0 = dist.argmin(dim=1)
        L  = mask.long().sum(dim=1)

        return t0, L

    @torch.no_grad()
    def get_actions(self, ep_idx: torch.Tensor, fingertip_pos: torch.Tensor,
                    fingertip_quat: torch.Tensor | None = None,
                    gripper: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns one step of actions (N,8). Recomputes NN if cache is empty.
        The horizon cache prefetches t..t+h-1 actions in batch.
        """
        N, dev = fingertip_pos.shape[0], fingertip_pos.device

        # --- Refill cache if empty ---
        if self._queued is None or self._q_ptr >= self._horizon:
            t0, L = self._nn_indices(ep_idx, fingertip_pos, fingertip_quat, gripper)
            ar = torch.arange(self._horizon, device=dev)
            idx = torch.minimum(t0[:,None] + ar, (L-1).clamp(min=0)[:,None])  # clamp past end

            # Gather all actions in parallel (N,H,8)
            act_p, act_q, act_g = self._act_pos.to(dev)[ep_idx], self._act_quat.to(dev)[ep_idx], self._act_grip.to(dev)[ep_idx]
            g3,g4,g1 = idx.unsqueeze(-1).expand(-1,-1,3), idx.unsqueeze(-1).expand(-1,-1,4), idx.unsqueeze(-1).expand(-1,-1,1)
            a_p, a_q, a_g = torch.gather(act_p,1,g3), torch.gather(act_q,1,g4), torch.gather(act_g,1,g1)
            self._queued = torch.cat([a_p,a_q,a_g], dim=-1)  # (N,H,8)
            self._q_ptr = 0

        # --- Pop next action ---
        out = self._queued[:, self._q_ptr, :]
        self._q_ptr += 1
        return out
