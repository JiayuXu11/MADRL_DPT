"""On-policy buffer for actor."""

import torch
import numpy as np
from harl.utils.trans_tools import _flatten, _sa_cast
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space


class OnPolicyActorBuffer:
    """On-policy buffer for actor data storage."""

    def __init__(self, args, obs_space, act_space):
        """Initialize on-policy actor buffer.
        Args:
            args: (dict) arguments
            obs_space: (gym.Space or list) observation space
            act_space: (gym.Space) action space
        """
        self.args = args
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = args["recurrent_n"]

        obs_shape = get_shape_from_obs_space(obs_space)

        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]

        # Buffer for observations of this actor.
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )

        # Buffer for rnn states of this actor.
        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # Buffer for available actions of this actor.
        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, act_space.n),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        # Buffer for actions of this actor.
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )

        # Buffer for action log probs of this actor.
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )

        # Buffer for masks of this actor. Masks denotes at which point should the rnn states be reset.
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        # Buffer for lt_today and lt_window
        self.lt_today = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.lt_window = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        # Buffer for active masks of this actor. Active masks denotes whether the agent is alive.
        self.active_masks = np.ones_like(self.masks)

        self.factor = None

        self.step = 0

    def update_factor(self, factor):
        """Save factor for this actor."""
        self.factor = factor.copy()

    def insert(
        self,
        obs,
        rnn_states,
        actions,
        action_log_probs,
        masks,
        active_masks=None,
        available_actions=None,
        lt_today=None,
        lt_window=None,
    ):
        """Insert data into actor buffer."""
        mec_type = self.args['mechanism_agent_type']
        if mec_type == 'pricing':
            self.obs[self.step] = obs.copy()
        else:
            self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()
        if lt_today is not None:
            self.lt_today[self.step] = lt_today.copy()
        if lt_window is not None:
            self.lt_window[self.step] = lt_window.copy()

        self.step = (self.step + 1) % self.episode_length

    def insert_final_obs(self, obs, available_actions=None):
        """Insert final observation into actor buffer for pricing mechanism type.

        For pricing type, obs is stored at self.step position during insert(),
        so we need to explicitly store the final next_obs at the last position.
        """
        self.obs[-1] = obs.copy()
        if available_actions is not None and self.available_actions is not None:
            self.available_actions[-1] = available_actions.copy()

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    # get sampler for recurrent_generator_actor
    def get_sampler_for_recurrent_generator_actor(self, actor_num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // actor_num_mini_batch

        assert episode_length % data_chunk_length == 0, (
            f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        )
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]
        return sampler

    def recurrent_generator_actor(self, advantages, actor_num_mini_batch, data_chunk_length, order_advantages, transship_advantages, sampler=None):
        """Training data generator for actor that uses RNN network.
        This generator splits the trajectories into chunks of length data_chunk_length,
        and therefore maybe more efficient than the naive_recurrent_generator_actor in training.
        """

        # get episode_length, n_rollout_threads, and mini_batch_size
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // actor_num_mini_batch

        assert episode_length % data_chunk_length == 0, (
            f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        )
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices
        if sampler is None:
            rand = torch.randperm(data_chunks).numpy()
            sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
                for i in range(actor_num_mini_batch)
            ]

        # The following data operations first transpose the first two dimensions of the data (episode_length, n_rollout_threads)
        # to (n_rollout_threads, episode_length), then reshape the data to (n_rollout_threads * episode_length, *dim).
        # Take obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *obs_shape) --> (episode_length, n_rollout_threads, *obs_shape)
        # --> (n_rollout_threads, episode_length, *obs_shape) --> (n_rollout_threads * episode_length, *obs_shape)
        if len(self.obs.shape) > 3:
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            obs = _sa_cast(self.obs[:-1])
        actions = _sa_cast(self.actions)
        action_log_probs = _sa_cast(self.action_log_probs)
        advantages = _sa_cast(advantages)
        order_advantages = _sa_cast(order_advantages)
        transship_advantages = _sa_cast(transship_advantages)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(self.active_masks[:-1])
        lt_today = _sa_cast(self.lt_today[:-1])
        lt_window = _sa_cast(self.lt_window[:-1])
        if self.factor is not None:
            factor = _sa_cast(self.factor)
        rnn_states = (
            self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        )
        if self.available_actions is not None:
            available_actions = _sa_cast(self.available_actions[:-1])

        # generate mini-batches
        for indices in sampler:
            obs_batch = []
            rnn_states_batch = []
            actions_batch = []
            available_actions_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            lt_today_batch = []
            lt_window_batch = []
            adv_targ = []
            order_adv_targ = []
            transship_adv_targ = []
            factor_batch = []

            for index in indices:
                ind = index * data_chunk_length
                obs_batch.append(obs[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind : ind + data_chunk_length])
                lt_today_batch.append(lt_today[ind : ind + data_chunk_length])
                lt_window_batch.append(lt_window[ind : ind + data_chunk_length])
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                order_adv_targ.append(order_advantages[ind : ind + data_chunk_length])
                transship_adv_targ.append(transship_advantages[ind : ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind : ind + data_chunk_length])  # only the beginning rnn states are needed
                if self.factor is not None:
                    factor_batch.append(factor[ind : ind + data_chunk_length])

            L, N = data_chunk_length, mini_batch_size
            # These are all ndarrays of size (data_chunk_length, mini_batch_size, *dim)
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            lt_today_batch = np.stack(lt_today_batch, axis=1)
            lt_window_batch = np.stack(lt_window_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)
            order_adv_targ = np.stack(order_adv_targ, axis=1)
            transship_adv_targ = np.stack(transship_adv_targ, axis=1)
            rnn_states_batch = np.stack(rnn_states_batch, axis=1)

            # flatten the (data_chunk_length, mini_batch_size, *dim) ndarrays to (data_chunk_length * mini_batch_size, *dim)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            lt_today_batch = _flatten(L, N, lt_today_batch)
            lt_window_batch = _flatten(L, N, lt_window_batch)
            adv_targ = _flatten(L, N, adv_targ)
            order_adv_targ = _flatten(L, N, order_adv_targ)
            transship_adv_targ = _flatten(L, N, transship_adv_targ)
            rnn_states_batch = _flatten(L, N, rnn_states_batch)
            if self.factor is not None:
                yield obs_batch, rnn_states_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, lt_today_batch, lt_window_batch, order_adv_targ, transship_adv_targ
            else:
                yield obs_batch, rnn_states_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, lt_today_batch, lt_window_batch, order_adv_targ, transship_adv_targ
