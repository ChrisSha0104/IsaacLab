import torch, numpy as np

def quat_geodesic_angle(q1, q2, eps=1e-8):
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(eps)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(eps)
    dot = (q1 * q2).sum(-1).abs().clamp(-1 + eps, 1 - eps)
    return 2 * torch.arccos(dot)

class NearestNeighborBuffer:
    """Nearest-neighbor action retriever with per-env horizon queues (async)."""

    def __init__(self, path: str, num_envs: int, horizon: int = 15):
        flat = np.load(path, allow_pickle=True)
        flat = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in flat.items()}
        eps = sorted({k.split("/", 1)[0] for k in flat})
        data = {e: {s.split("/", 1)[1]: flat[s] for s in flat if s.startswith(e)} for e in eps}

        lengths = torch.tensor([len(data[e]["obs.gripper"]) for e in eps])
        T = int(lengths.max())

        def pad_last(key, d):
            out = torch.empty((len(eps), T, d))
            for i, e in enumerate(eps):
                x = data[e][key]
                if len(x) < T:
                    x = torch.cat([x, x[-1:].repeat(T - len(x), 1)], dim=0)
                out[i] = x
            return out

        self._obs_pos, self._obs_quat, self._obs_grip = pad_last("obs.eef_pos",3), pad_last("obs.eef_quat",4), pad_last("obs.gripper",1)
        self._act_pos, self._act_quat, self._act_grip = pad_last("action.eef_pos",3), pad_last("action.eef_quat",4), pad_last("action.gripper",1)
        self._mask = torch.arange(T).expand(len(eps), T) < lengths[:, None]

        self._num_envs, self._horizon = num_envs, int(horizon)
        self._queued = None                       # (N,H,8) allocated on first call's device
        self._q_ptr = torch.zeros(num_envs, dtype=torch.long)  # per-env read index
        self._q_len = torch.zeros(num_envs, dtype=torch.long)  # per-env available length (0 => empty)

        self._total_episodes = len(eps)
        self._max_episode_length = T
        print(f"Loaded {len(eps)} episodes; max length {T}.")

    # --- public helpers ---
    def get_total_episodes(self): return self._total_episodes
    def get_max_episode_length(self): return self._max_episode_length
    def clear(self, env_ids: torch.Tensor | np.ndarray | list):
        """Clear queues for specific envs (forces refill on next call)."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long)
        self._q_ptr[env_ids] = 0
        self._q_len[env_ids] = 0

    # --- core NN ---
    def _nn_indices(self, eidx, pos, quat=None, grip=None, verbose=False):
        dev = pos.device
        obs_p = self._obs_pos.to(dev)[eidx]
        obs_q = self._obs_quat.to(dev)[eidx]
        obs_g = self._obs_grip.to(dev)[eidx]
        mask  = self._mask.to(dev)[eidx]

        pos_term  = 10 * torch.norm(obs_p - pos[:, None, :], dim=-1)      # cm/10
        ang_term  = torch.zeros_like(pos_term)
        grip_term = torch.zeros_like(pos_term)
        if quat is not None: ang_term  = torch.rad2deg(quat_geodesic_angle(obs_q, quat[:, None, :])) / 10
        if grip is not None: grip_term = 2 * (obs_g.squeeze(-1) - grip.view(-1,1)).abs()

        dist = (pos_term + ang_term + grip_term).masked_fill(~mask, float("inf"))
        t0   = dist.argmin(dim=1)                 # (M,)
        L    = mask.long().sum(dim=1)             # (M,)

        if verbose:
            mmean = lambda x: (x.masked_fill(~mask, torch.nan)).nanmean().item()
            print(f"[NN contrib] pos_cm/10: {mmean(pos_term):.3f}, ang_deg/10: {mmean(ang_term):.3f}, grip_L1*2: {mmean(grip_term):.3f}")
        return t0, L

    @torch.no_grad()
    def get_actions(self, eidx: torch.Tensor, pos: torch.Tensor,
                    quat: torch.Tensor | None = None,
                    grip: torch.Tensor | None = None,
                    verbose: bool = False) -> torch.Tensor:
        """
        Asynchronous per-env queues:
          - Refill only envs whose queue is empty (ptr >= len).
          - Return one action per env and advance its ptr.
        """
        N, dev = pos.shape[0], pos.device
        if self._queued is None or self._queued.device != dev:
            self._queued = torch.empty((self._num_envs, self._horizon, 8), device=dev)

        # Which envs need refill?
        need = (self._q_ptr >= self._q_len)
        if need.any():
            ids = need.nonzero(as_tuple=False).squeeze(-1)   # (M,)
            # subset inputs
            t0, L = self._nn_indices(eidx[ids], pos[ids], None if quat is None else quat[ids],
                                     None if grip is None else grip[ids], verbose)
            ar  = torch.arange(self._horizon, device=dev)    # (H,)
            idx = torch.minimum(t0[:, None] + ar, (L - 1).clamp(min=0)[:, None])  # (M,H)

            ap, aq, ag = self._act_pos.to(dev)[eidx[ids]], self._act_quat.to(dev)[eidx[ids]], self._act_grip.to(dev)[eidx[ids]]
            gi3, gi4, gi1 = idx[..., None].expand(-1,-1,3), idx[..., None].expand(-1,-1,4), idx[..., None].expand(-1,-1,1)
            a = torch.cat([torch.gather(ap,1,gi3), torch.gather(aq,1,gi4), torch.gather(ag,1,gi1)], dim=-1)  # (M,H,8)

            self._queued[ids] = a
            self._q_ptr[ids] = 0
            self._q_len[ids] = self._horizon

        # Pop per-env next action
        env_ids = torch.arange(N, device=dev)
        step_idx = torch.minimum(self._q_ptr.to(dev), (self._q_len - 1).clamp(min=0).to(dev))
        out = self._queued[env_ids, step_idx, :]             # (N,8)

        # Advance ptr only where we have data
        has_data = (self._q_ptr < self._q_len)
        self._q_ptr[has_data] += 1

        return out
