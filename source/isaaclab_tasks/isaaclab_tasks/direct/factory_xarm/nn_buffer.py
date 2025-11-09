import torch
import numpy as np

class NearestNeighborBuffer:
    def __init__(self, action_data_path: str, num_envs: int):
        """
        Expects keys per episode:
          obs.eef_pos (T,3), obs.eef_quat (T,4), obs.gripper (T,1)
          action.eef_pos (T,3), action.eef_quat (T,4), action.gripper (T,1)
        """
        flat = np.load(action_data_path, allow_pickle=True)
        flat = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in flat.items()}

        # Group by episode
        episodes = sorted({k.split("/", 1)[0] for k in flat})
        data = {ep: {s.split("/", 1)[1]: flat[s] for s in flat if s.startswith(ep)} for ep in episodes}

        lens = torch.tensor([len(data[ep]["obs.gripper"]) for ep in episodes], dtype=torch.long)
        T_max = int(lens.max().item())

        def pad_stack(key, dim):
            out = torch.full((len(episodes), T_max, dim), float("nan"))
            for i, ep in enumerate(episodes):
                x = data[ep][key]
                out[i, : x.shape[0]] = x
            return out

        self._episodes = episodes
        self._episode_lengths = lens
        self._max_episode_length = T_max
        self._num_envs = num_envs
        self._total_episodes = len(episodes)

        # Observations (used for NN)
        self._obs_pos   = pad_stack("obs.eef_pos",   3)  # (E,T,3)
        self._obs_quat  = pad_stack("obs.eef_quat",  4)  # (E,T,4)
        self._obs_grip  = pad_stack("obs.gripper",   1)  # (E,T,1)

        # Actions (returned at matched timestep)
        self._act_pos   = pad_stack("action.eef_pos",  3)  # (E,T,3)
        self._act_quat  = pad_stack("action.eef_quat", 4)  # (E,T,4)
        self._act_grip  = pad_stack("action.gripper", 1)   # (E,T,1)

        self._valid_mask = torch.zeros((len(episodes), T_max), dtype=torch.bool)
        for i, L in enumerate(lens):
            self._valid_mask[i, : int(L)] = True

        print(f"Loaded NearestNeighborBuffer with {self._total_episodes} episodes; max length {self._max_episode_length}.")

    def get_total_episodes(self) -> int:
        return self._total_episodes

    def get_max_episode_length(self) -> int:
        return int(self._max_episode_length)

    @torch.no_grad()
    def get_actions(self, episode_idxs: torch.Tensor, fingertip_pos: torch.Tensor) -> torch.Tensor:
        """
        Nearest neighbor in obs.eef_pos within the selected episode,
        return the paired actions at the same timestep.
        Returns (N,8): [action_pos(3), action_quat(4), action_gripper(1)]
        """
        assert episode_idxs.ndim == 1
        assert fingertip_pos.ndim == 2 and fingertip_pos.shape[-1] == 3
        N, device = fingertip_pos.shape[0], fingertip_pos.device

        # Move needed tensors to device
        obs_pos = self._obs_pos.to(device)            # (E,T,3)
        mask    = self._valid_mask.to(device)         # (E,T)
        act_pos = self._act_pos.to(device)            # (E,T,3)
        act_qua = self._act_quat.to(device)           # (E,T,4)
        act_gr  = self._act_grip.to(device)           # (E,T,1)

        # Select per-env episodes
        obs_e = obs_pos[episode_idxs]                 # (N,T,3) # TODO: add noise?
        m_e   = mask[episode_idxs]                    # (N,T)
        ap_e  = act_pos[episode_idxs]
        aq_e  = act_qua[episode_idxs]
        ag_e  = act_gr[episode_idxs]

        # NN over obs.eef_pos (mask padded to +inf)
        dist2 = ((obs_e - fingertip_pos[:, None, :])**2).sum(-1)      # (N,T)
        dist2 = dist2.masked_fill(~m_e, float('inf'))
        idx   = dist2.argmin(dim=1)                                   # (N,)

        # Gather paired actions at that timestep
        t = idx.view(N, 1, 1)
        a_pos = ap_e.gather(1, t.expand(N, 1, 3)).squeeze(1)          # (N,3)
        a_qua = aq_e.gather(1, t.expand(N, 1, 4)).squeeze(1)          # (N,4)
        a_grp = ag_e.gather(1, t.expand(N, 1, 1)).squeeze(1)          # (N,1)

        # check nan in output
        if torch.isnan(a_pos).any() or torch.isnan(a_qua).any() or torch.isnan(a_grp).any():
            print("idx:", idx)
            import pdb; pdb.set_trace()

        return torch.cat([a_pos, a_qua, a_grp], dim=-1)               # (N,8)
