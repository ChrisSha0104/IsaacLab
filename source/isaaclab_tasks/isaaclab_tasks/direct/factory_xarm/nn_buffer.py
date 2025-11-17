import torch, numpy as np

def quat_geodesic_angle(q1, q2, eps=1e-8):
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(eps)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(eps)
    dot = (q1 * q2).sum(-1).abs().clamp(-1 + eps, 1 - eps)
    return 2 * torch.arccos(dot)


class NearestNeighborBuffer:
    """Nearest-neighbor action retriever with per-env horizon queues."""

    def __init__(self, path: str, num_envs: int, horizon: int = 15,
                 device: str | torch.device = "cpu"):
        self._device = torch.device(device)

        flat = np.load(path, allow_pickle=True)
        flat = {k: torch.as_tensor(v, dtype=torch.float32, device=self._device)
                for k, v in flat.items()}

        eps = sorted({k.split("/", 1)[0] for k in flat})
        data = {e: {s.split("/", 1)[1]: flat[s] for s in flat if s.startswith(e)}
                for e in eps}

        lengths = torch.tensor([len(data[e]["obs.gripper"]) for e in eps],
                               device=self._device)
        T = int(lengths.max())

        def pad_last(key, d):
            out = torch.empty((len(eps), T, d), device=self._device)
            for i, e in enumerate(eps):
                x = data[e][key]
                if len(x) < T:
                    x = torch.cat([x, x[-1:].repeat(T - len(x), 1)], dim=0)
                out[i] = x
            return out

        self._obs_pos  = pad_last("obs.eef_pos", 3)
        self._obs_quat = pad_last("obs.eef_quat", 4)
        self._obs_grip = pad_last("obs.gripper", 1)
        self._act_pos  = pad_last("action.eef_pos", 3)
        self._act_quat = pad_last("action.eef_quat", 4)
        self._act_grip = pad_last("action.gripper", 1)
        self._mask     = (torch.arange(T, device=self._device)
                          .expand(len(eps), T) < lengths[:, None])

        self._num_envs = num_envs
        self._max_horizon = int(horizon)
        self._horizon_env = torch.full((num_envs,),
                                       self._max_horizon,
                                       dtype=torch.long,
                                       device=self._device)

        self._queued = None          # (N, H_max, 8), on self._device
        self._queued_idx = None      # (N, H_max)
        self._q_ptr = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        self._q_len = torch.zeros(num_envs, dtype=torch.long, device=self._device)

        self._total_episodes = len(eps)
        self._max_episode_length = T
        print(f"Loaded {len(eps)} episodes; max length {T} on {self._device}.")

    # --- public helpers ---

    def get_total_episodes(self):
        return self._total_episodes

    def get_max_episode_length(self):
        return self._max_episode_length

    def clear(self,
              env_ids: torch.Tensor | np.ndarray | list,
              horizon: int | torch.Tensor | np.ndarray | list | None = None):
        """
        Clear queues for the given env ids.
        - If horizon is int: apply same horizon to all env_ids.
        - If horizon is Tensor / ndarray / list: must have same length as env_ids.
        - If horizon is None: do not change horizons.
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)

        self._q_ptr[env_ids] = 0
        self._q_len[env_ids] = 0

        if horizon is None:
            return

        if isinstance(horizon, int):
            h = torch.full((env_ids.numel(),),
                           horizon,
                           dtype=torch.long,
                           device=self._device)
        else:
            h = torch.as_tensor(horizon, dtype=torch.long, device=self._device)
            if h.numel() != env_ids.numel():
                raise ValueError(
                    f"Horizon length mismatch: got {h.numel()} for {env_ids.numel()} env_ids."
                )

        self._horizon_env[env_ids] = h

    # --- core NN ---

    def _nn_indices(self, eidx, pos, quat=None, grip=None, verbose=False):
        # All tensors already on self._device
        obs_p = self._obs_pos[eidx]
        obs_q = self._obs_quat[eidx]
        obs_g = self._obs_grip[eidx]
        mask  = self._mask[eidx]

        pos_term  = 10 * torch.norm(obs_p - pos[:, None, :], dim=-1)
        ang_term  = torch.zeros_like(pos_term)
        grip_term = torch.zeros_like(pos_term)

        if quat is not None:
            ang_term = torch.rad2deg(
                quat_geodesic_angle(obs_q, quat[:, None, :])
            ) / 10
        if grip is not None:
            grip_term = 2 * (obs_g.squeeze(-1) - grip.view(-1, 1)).abs()

        dist = (pos_term + ang_term + grip_term).masked_fill(~mask, float("inf"))
        t0   = dist.argmin(dim=1)
        L    = mask.long().sum(dim=1)

        if verbose:
            mmean = lambda x: x.masked_fill(~mask, torch.nan).nanmean().item()
            print(f"[NN contrib] pos_cm/10: {mmean(pos_term):.3f}, "
                  f"ang_deg/10: {mmean(ang_term):.3f}, "
                  f"grip_L1*2: {mmean(grip_term):.3f}")
        return t0, L

    @torch.no_grad()
    def get_actions(self,
                    eidx: torch.Tensor,
                    pos: torch.Tensor,
                    quat: torch.Tensor | None = None,
                    grip: torch.Tensor | None = None,
                    verbose: bool = False) -> torch.Tensor:
        """
        Asynchronous per-env queues with per-env horizon:
          - Refill only envs whose queue is empty (ptr >= len).
          - Per env i, we use horizon self._horizon_env[i] (<= max_horizon).
          - Return one action per env and advance its ptr.
        """
        # Enforce device consistency
        if pos.device != self._device:
            raise ValueError(f"pos.device={pos.device} but buffer.device={self._device}")
        if quat is not None and quat.device != self._device:
            raise ValueError("quat must be on the same device as the buffer")
        if grip is not None and grip.device != self._device:
            raise ValueError("grip must be on the same device as the buffer")

        N = pos.shape[0]

        if self._queued is None:
            self._queued = torch.empty(
                (self._num_envs, self._max_horizon, 8), device=self._device
            )
            self._queued_idx = torch.empty(
                (self._num_envs, self._max_horizon), dtype=torch.long, device=self._device
            )

        refill = (self._q_ptr >= self._q_len)   # (num_envs,)
        if refill.any():
            ids = refill.nonzero(as_tuple=False).squeeze(-1)  # (M,)

            t0, L = self._nn_indices(eidx[ids],
                                     pos[ids],
                                     None if quat is None else quat[ids],
                                     None if grip is None else grip[ids],
                                     verbose)

            ar = torch.arange(self._max_horizon, device=self._device)   # (H_max,)
            idx = t0[:, None] + ar[None, :]                             # (M, H_max)
            idx = torch.minimum(idx, (L - 1).clamp(min=0)[:, None])

            ap = self._act_pos[eidx[ids]]   # (M, T, 3)
            aq = self._act_quat[eidx[ids]]  # (M, T, 4)
            ag = self._act_grip[eidx[ids]]  # (M, T, 1)

            gi3 = idx[..., None].expand(-1, -1, 3)
            gi4 = idx[..., None].expand(-1, -1, 4)
            gi1 = idx[..., None].expand(-1, -1, 1)

            a = torch.cat([
                torch.gather(ap, 1, gi3),
                torch.gather(aq, 1, gi4),
                torch.gather(ag, 1, gi1),
            ], dim=-1)  # (M, H_max, 8)

            self._queued[ids] = a
            self._queued_idx[ids] = idx

            H_env = self._horizon_env[ids]
            self._q_ptr[ids] = 0
            self._q_len[ids] = H_env

        env_ids = torch.arange(N, device=self._device)
        step_idx = torch.minimum(self._q_ptr, (self._q_len - 1).clamp(min=0))
        out = self._queued[env_ids, step_idx, :]  # (N, 8)

        has_data = (self._q_ptr < self._q_len)
        self._q_ptr[has_data] += 1

        return out
