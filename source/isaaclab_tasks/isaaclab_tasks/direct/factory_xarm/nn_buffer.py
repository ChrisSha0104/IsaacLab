import torch, numpy as np

def quat_geodesic_angle(q1, q2, eps=1e-8):
    """Returns angular distance (radians) between two quaternions."""
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(eps)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(eps)
    dot = (q1 * q2).sum(-1).abs().clamp(-1 + eps, 1 - eps)
    return 2 * torch.arccos(dot)

class NearestNeighborBuffer:
    """Nearest-neighbor action retriever with batched horizon caching."""

    def __init__(self, path: str, num_envs: int, horizon: int = 15):
        flat = np.load(path, allow_pickle=True)
        flat = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in flat.items()}
        episodes = sorted({k.split("/", 1)[0] for k in flat})
        data = {e: {s.split("/", 1)[1]: flat[s] for s in flat if s.startswith(e)} for e in episodes}

        lengths = torch.tensor([len(data[e]["obs.gripper"]) for e in episodes])
        T = int(lengths.max())

        def pad_last(key, dim):
            """Pad each episode to T by repeating its last value."""
            out = torch.empty((len(episodes), T, dim))
            for i, e in enumerate(episodes):
                x = data[e][key]
                if len(x) < T:
                    pad = x[-1:].repeat(T - len(x), 1)
                    x = torch.cat([x, pad], dim=0)
                out[i] = x
            return out

        # tensors (E,T,dim)
        self._obs_pos, self._obs_quat, self._obs_grip = (
            pad_last("obs.eef_pos", 3),
            pad_last("obs.eef_quat", 4),
            pad_last("obs.gripper", 1),
        )
        self._act_pos, self._act_quat, self._act_grip = (
            pad_last("action.eef_pos", 3),
            pad_last("action.eef_quat", 4),
            pad_last("action.gripper", 1),
        )

        self._mask = torch.arange(T).expand(len(episodes), T) < lengths[:, None]
        self._horizon = horizon
        self._queued = None
        self._q_ptr = horizon
        self._num_envs = num_envs
        self._total_episodes = len(episodes)
        self._max_episode_length = T
        print(f"Loaded {len(episodes)} episodes; max length {T}.")

    def _nn_indices(self, eidx, pos, quat=None, grip=None, verbose=False):
        """Find nearest neighbor timestep per env using position (+quat/+grip)."""
        dev = pos.device
        obs_p = self._obs_pos.to(dev)[eidx]
        obs_q = self._obs_quat.to(dev)[eidx]
        obs_g = self._obs_grip.to(dev)[eidx]
        mask  = self._mask.to(dev)[eidx]

        # --- component terms ---
        pos_term = 10 * torch.norm(obs_p - pos[:, None, :], dim=-1)  # pos_cm / 10
        ang_term = torch.zeros_like(pos_term)
        grip_term = torch.zeros_like(pos_term)

        if quat is not None:
            ang_term = torch.rad2deg(quat_geodesic_angle(obs_q, quat[:, None, :])) / 10
        if grip is not None:
            grip_term = 2 * (obs_g.squeeze(-1) - grip.view(-1, 1)).abs()

        dist = pos_term + ang_term + grip_term
        dist = dist.masked_fill(~mask, float("inf"))
        t0 = dist.argmin(dim=1)
        L = mask.long().sum(dim=1)

        # --- verbose print ---
        if verbose:
            def masked_mean(x): return (x.masked_fill(~mask, torch.nan)).nanmean().item()
            print(
                f"[NN contrib] pos_cm/10: {masked_mean(pos_term):.3f}, "
                f"ang_deg/10: {masked_mean(ang_term):.3f}, "
                f"grip_L1*2: {masked_mean(grip_term):.3f}"
            )

        return t0, L

    @torch.no_grad()
    def get_actions(self, eidx, pos, quat=None, grip=None, verbose=False):
        """Return (N,8) actions; refill horizon batch if empty."""
        N, dev = pos.shape[0], pos.device
        if self._queued is None or self._q_ptr >= self._horizon:
            t0, L = self._nn_indices(eidx, pos, quat, grip, verbose)
            ar = torch.arange(self._horizon, device=dev)
            idx = torch.minimum(t0[:, None] + ar, (L - 1).clamp(min=0)[:, None])

            ap, aq, ag = (
                self._act_pos.to(dev)[eidx],
                self._act_quat.to(dev)[eidx],
                self._act_grip.to(dev)[eidx],
            )
            gi3, gi4, gi1 = (
                idx[..., None].expand(-1, -1, 3),
                idx[..., None].expand(-1, -1, 4),
                idx[..., None].expand(-1, -1, 1),
            )
            a = torch.cat(
                [torch.gather(ap, 1, gi3), torch.gather(aq, 1, gi4), torch.gather(ag, 1, gi1)],
                dim=-1,
            )
            self._queued, self._q_ptr = a, 0

        out = self._queued[:, self._q_ptr, :]
        self._q_ptr += 1
        return out
