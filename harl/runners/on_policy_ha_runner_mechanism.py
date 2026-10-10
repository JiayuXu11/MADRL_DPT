"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
import torch.nn.functional as F
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner_mechanism import MixPolicyBaseRunner
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer


class OnMixPolicyHaRunner(MixPolicyBaseRunner):
    """Runner for on-policy HA algorithms with hierarchical agents"""

    def train(self):
        """Train the model."""
        actor_train_infos = [None] * self.num_agents
        critic_train_infos = [None] * len(self.critic_buffer)
        # critic_train_before_tf = False        # 标识critic被train过一次没有，避免重复train

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        advantages_collector = []
        for critic_buffer_id in range(len(self.critic_buffer)):
            if self.value_normalizer is not None:
                # shape: (episode_length, n_rollout_threads, dim)
                advantage = self.critic_buffer[critic_buffer_id].returns[
                    :-1
                ] - self.value_normalizer[critic_buffer_id].denormalize(
                    self.critic_buffer[critic_buffer_id].value_preds[:-1]
                )
            else:
                # shape: (episode_length, n_rollout_threads, dim)
                advantage = (
                    self.critic_buffer[0].returns[:-1] - self.critic_buffer[0].value_preds[:-1]
                )
            advantages_collector.append(advantage)
        # shape: (num_critic_buffer, episode_length, n_rollout_threads, dim)
        advantages = np.array(advantages_collector)

        # Compute split advantages for order and transship actions if enabled
        # We'll compute adjusted advantages per-agent in the training loop below
        order_advantages_dict = {}
        transship_advantages_dict = {}
        order_valid_mask_dict = {}


        gamma = self.algo_args["algo"]["gamma"]

        # For each agent, compute split advantages based on lt_today and lt_window
        for agent_id in range(self.num_agents):
            critic_id = min(agent_id, len(self.critic_buffer) - 1)
            agent_adv = advantages[critic_id]  # shape: (episode_length, n_rollout_threads, 1)

            # Get lt_today and lt_window from actor_buffer
            lt_today = self.actor_buffer[agent_id].lt_today[:-1]  # shape: (episode_length, n_rollout_threads, 1)
            lt_window = self.actor_buffer[agent_id].lt_window[:-1]  # shape: (episode_length, n_rollout_threads, 1)

            episode_length, n_threads = agent_adv.shape[0], agent_adv.shape[1]

            # Create time index grids: (episode_length, n_threads)
            t_indices = np.arange(episode_length)[:, np.newaxis]  # (episode_length, 1)
            t_indices = np.broadcast_to(t_indices, (episode_length, n_threads))  # (episode_length, n_threads)

            # Vectorized computation for order advantages
            # Order action: use adv[t + lt_today + 1]
            lt_today_int = lt_today.squeeze(-1).astype(np.int32)  # (episode_length, n_threads)
            future_t_order_raw = t_indices + lt_today_int  # (episode_length, n_threads)
            # 创建mask标记超出范围的位置
            order_valid_mask = ((future_t_order_raw >= 0) & (future_t_order_raw < episode_length))[:, :, np.newaxis]  # (episode_length, n_threads, 1)
            # 对索引进行截断以确保安全索引
            future_t_order = np.clip(future_t_order_raw, 0, episode_length - 1)  # (episode_length, n_threads)
            # 正确地采样order_adv，使得输出shape为(episode_length, n_threads, 1)
            # 使用高级索引: order_adv[i,j] = agent_adv[future_t_order[i,j], j, 0]
            j_indices = np.arange(n_threads)  # (n_threads,)
            order_adv = agent_adv[future_t_order, j_indices, 0]  # (episode_length, n_threads)
            order_adv = order_adv[:, :, np.newaxis]  # (episode_length, n_threads, 1)

            # Vectorized computation for transship advantages
            # Transship action: use adv[t] - gamma**lt_window * adv[t + lt_window]
            lt_window_int = lt_window.squeeze(-1).astype(np.int32)  # (episode_length, n_threads)
            future_t_transship = np.clip(t_indices + lt_window_int, 0, episode_length - 1)  # (episode_length, n_threads)

            # Gather future advantages
            # 使用高级索引: future_adv_transship[i,j] = agent_adv[future_t_transship[i,j], j, 0]
            future_adv_transship = agent_adv[future_t_transship, j_indices, 0]  # (episode_length, n_threads)
            future_adv_transship = future_adv_transship[:, :, np.newaxis]  # (episode_length, n_threads, 1)

            # Compute discount factors: gamma^lt_window
            discount = gamma ** lt_window  # (episode_length, n_threads, 1)

            # Compute transship advantages
            transship_adv = agent_adv
            # transship_adv = agent_adv - discount * future_adv_transship  # (episode_length, n_threads, 1)

            order_advantages_dict[agent_id] = order_adv
            transship_advantages_dict[agent_id] = transship_adv
            order_valid_mask_dict[agent_id] = order_valid_mask

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
        # 如果只有一个actor，那就共用所有actor_buffer的数据，混一起训练
        if self.share_param:
            mixed_actor_buffer = self._prepare_share_param_actor_buffer()
            # Normalize advantages, order_advantages_dict, transship_advantages_dict (with mask for order_adv)
            def _normalize(arr, mask=None):
                arr = arr.copy()
                if mask is not None:
                    arr[mask == 0.0] = np.nan
                    mean = np.nanmean(arr)
                    std = np.nanstd(arr)
                    normed = (arr - mean) / (std + 1e-5)
                    return normed * mask
                else:
                    mean = np.mean(arr)
                    std = np.std(arr)
                    return (arr - mean) / (std + 1e-5)

            for agent_id in order_advantages_dict:
                order_advantages_dict[agent_id] = _normalize(
                    order_advantages_dict[agent_id]
                )
                transship_advantages_dict[agent_id] = _normalize(
                    transship_advantages_dict[agent_id]
                )
                if not self.multi_critic_buffer_tf:
                    if agent_id == 0:
                        advantages[0] = _normalize(advantages[0])
                else:
                    advantages[agent_id] = _normalize(advantages[agent_id])

            mixed_advantages = self._mix_share_param_advantages(advantages)
            mixed_order_advantages = self._mix_share_param_timeseries(order_advantages_dict)
            mixed_transship_advantages = self._mix_share_param_timeseries(transship_advantages_dict)
            mixed_order_valid_mask = self._mix_share_param_timeseries(order_valid_mask_dict)
            actor_train_info = self.actor[0].train(
                mixed_actor_buffer,
                mixed_advantages,
                "NO_EP", # 反正不是EP就行
                mixed_order_advantages,
                mixed_transship_advantages,
                mixed_order_valid_mask,
            )
            actor_train_infos = [actor_train_info for _ in range(self.num_agents)]
        # 如果有多个actor，那就分别训练，各个actor只用自己的actor_buffer数据
        else:
            for agent_id in agent_order:
                self.actor_buffer[agent_id].update_factor(
                    factor
                )  # current actor save factor

                # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
                available_actions = (
                    None
                    if self.actor_buffer[agent_id].available_actions is None
                    else self.actor_buffer[agent_id]
                    .available_actions[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
                )

                # compute action log probs for the actor before update.
                if self.use_factor:
                    old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                        self.actor_buffer[agent_id]
                        .obs[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                        self.actor_buffer[agent_id]
                        .rnn_states[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                        self.actor_buffer[agent_id].actions.reshape(
                            -1, *self.actor_buffer[agent_id].actions.shape[2:]
                        ),
                        self.actor_buffer[agent_id]
                        .masks[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                        available_actions,
                        self.actor_buffer[agent_id]
                        .active_masks[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                    )

                # update actor
                actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        advantages[min(agent_id, len(self.critic_buffer) - 1)].copy(),
                        "EP",
                        order_advantages_dict[agent_id],
                        transship_advantages_dict[agent_id],
                        order_valid_mask_dict[agent_id]
                    )
                # compute action log probs for updated agent\
                if self.use_factor:
                    new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                        self.actor_buffer[agent_id]
                        .obs[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                        self.actor_buffer[agent_id]
                        .rnn_states[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                        self.actor_buffer[agent_id].actions.reshape(
                            -1, *self.actor_buffer[agent_id].actions.shape[2:]
                        ),
                        self.actor_buffer[agent_id]
                        .masks[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                        available_actions,
                        self.actor_buffer[agent_id]
                        .active_masks[:-1]
                        .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                    )

                    # update factor for next agent
                    factor = factor * _t2n(
                        getattr(torch, self.action_aggregation)(
                            torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                        ).reshape(
                            self.algo_args["train"]["episode_length"],
                            self.algo_args["train"]["n_rollout_threads"],
                            1,
                        )
                    )
                actor_train_infos[agent_id] = actor_train_info

        # update critic
        # 单个critic_buffer，单个critic
        if (not self.multi_critic_buffer_tf) and (not self.multi_critic_tf):
            critic_train_info = self.critic[0].train(self.critic_buffer[0], self.value_normalizer[0])
            critic_train_infos[0] = critic_train_info
        # 多个critic_buffer，多个critic
        elif self.multi_critic_buffer_tf and self.multi_critic_tf:
            for agent_id in agent_order:
                critic_train_info = self.critic[agent_id].train(self.critic_buffer[agent_id], self.value_normalizer[agent_id])
                critic_train_infos[agent_id] = critic_train_info
        # 多个critic_buffer，单个critic
        elif self.multi_critic_buffer_tf and (not self.multi_critic_tf):
            critic_train_info = self.critic[0].share_param_train(self.critic_buffer, self.value_normalizer)
            for agent_id in agent_order:
                critic_train_infos[agent_id] = critic_train_info
        else:
            raise NotImplementedError

        return actor_train_infos, critic_train_infos

    def _ensure_share_param_actor_buffer(self):
        if not hasattr(self, "_share_param_actor_buffer"):
            mix_args = {**self.algo_args["train"], **self.algo_args["model"]}
            mix_args["n_rollout_threads"] = (
                mix_args["n_rollout_threads"] * self.num_agents
            )
            self._share_param_actor_buffer = OnPolicyActorBuffer(
                mix_args, self.envs.observation_space[0], self.envs.action_space[0]
            )
        return self._share_param_actor_buffer

    def _prepare_share_param_actor_buffer(self):
        buffer = self._ensure_share_param_actor_buffer()
        threads_per_agent = self.algo_args["train"]["n_rollout_threads"]
        for agent_id, actor_buffer in enumerate(self.actor_buffer):
            start = agent_id * threads_per_agent
            end = start + threads_per_agent
            buffer.obs[:, start:end] = actor_buffer.obs
            buffer.rnn_states[:, start:end] = actor_buffer.rnn_states
            buffer.masks[:, start:end] = actor_buffer.masks
            buffer.active_masks[:, start:end] = actor_buffer.active_masks
            buffer.lt_today[:, start:end] = actor_buffer.lt_today
            buffer.lt_window[:, start:end] = actor_buffer.lt_window
            buffer.actions[:, start:end] = actor_buffer.actions
            buffer.action_log_probs[:, start:end] = actor_buffer.action_log_probs
            if (
                buffer.available_actions is not None
                and actor_buffer.available_actions is not None
            ):
                buffer.available_actions[:, start:end] = actor_buffer.available_actions
        buffer.factor = None
        return buffer

    def _mix_share_param_advantages(self, advantages):
        episode_length = advantages.shape[1]
        threads_per_agent = self.algo_args["train"]["n_rollout_threads"]
        tail_shape = advantages.shape[3:]
        mixed_shape = (episode_length, threads_per_agent * self.num_agents, *tail_shape)
        mixed = np.zeros(mixed_shape, dtype=advantages.dtype)
        for agent_id in range(self.num_agents):
            src_id = min(agent_id, advantages.shape[0] - 1)
            start = agent_id * threads_per_agent
            end = start + threads_per_agent
            mixed[:, start:end] = advantages[src_id]
        return mixed

    def _mix_share_param_timeseries(self, tensor_dict):
        if not tensor_dict:
            return None
        sample = tensor_dict[0]
        episode_length = sample.shape[0]
        threads_per_agent = self.algo_args["train"]["n_rollout_threads"]
        tail_shape = sample.shape[2:]
        mixed_shape = (episode_length, threads_per_agent * self.num_agents, *tail_shape)
        mixed = np.zeros(mixed_shape, dtype=sample.dtype)
        for agent_id in range(self.num_agents):
            start = agent_id * threads_per_agent
            end = start + threads_per_agent
            mixed[:, start:end] = tensor_dict[agent_id]
        return mixed

    def off_train(self):
        """Train the model"""
        self.total_it += 1
        data = self.off_buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.off_critic.turn_on_grad()
        if self.args["algo_mechanism"] == "hasac":
            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.off_num_agents):
                next_action, next_logp_action = self.off_actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                )
                next_actions.append(next_action)
                next_logp_actions.append(next_logp_action)
            self.off_critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.off_value_normalizer,
            )
        else:
            next_actions = []
            for agent_id in range(self.off_num_agents):
                next_actions.append(
                    self.off_actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
            self.off_critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.off_critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            # train actors
            if self.args["algo_mechanism"] == "hasac":
                actions = [None] * self.off_num_agents
                logp_actions = [None] * self.off_num_agents
                # actions shape: (n_agents, batch_size, dim)
                # logp_actions shape: (n_agents, batch_size, 1)
                if self.fixed_order:
                    agent_order = list(range(self.off_num_agents))
                else:
                    agent_order = list(np.random.permutation(self.off_num_agents))
                for agent_id in agent_order:
                    self.off_actor[agent_id].turn_on_grad()
                    # train this agent
                    actions[agent_id], logp_actions[agent_id] = self.off_actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )

                    logp_action = logp_actions[agent_id]
                    actions_t = torch.cat(actions, dim=-1)

                    value_pred = self.off_critic.get_values(sp_share_obs, actions_t)

                    # 创建 mask：logp_action 在 +-1e-6 之间时不参与训练
                    nonzero_mask = (logp_action.abs() > 1e-6).float()

                    if self.algo_mechanism_args["algo"]["use_policy_active_masks"]:
                        combined_mask = sp_valid_transition[agent_id] * nonzero_mask
                        actor_loss = (
                            -torch.sum(
                                (value_pred - self.alpha[agent_id] * logp_action)
                                * combined_mask
                            )
                            / combined_mask.sum().clamp(min=1)
                        )
                    else:
                        actor_loss = (
                            -torch.sum(
                                (value_pred - self.alpha[agent_id] * logp_action)
                                * nonzero_mask
                            )
                            / nonzero_mask.sum().clamp(min=1)
                        )
                    self.off_actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.off_actor[agent_id].actor_optimizer.step()
                    self.off_actor[agent_id].turn_off_grad()
                    # train this agent's alpha
                    if self.algo_mechanism_args["algo"]["auto_alpha"]:
                        log_prob = (
                            logp_actions[agent_id].detach()
                            + self.target_entropy[agent_id]
                        )
                        alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                        self.alpha_optimizer[agent_id].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[agent_id].step()
                        self.alpha[agent_id] = torch.exp(
                            self.log_alpha[agent_id].detach()
                        )
                    actions[agent_id], _ = self.off_actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                # train critic's alpha
                if self.algo_mechanism_args["algo"]["auto_alpha"]:
                    self.off_critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            else:
                if self.args["algo_mechanism"] == "had3qn":
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.off_num_agents):
                            actions.append(
                                self.off_actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, 1)
                    update_actions, get_values = self.off_critic.train_values(
                        sp_share_obs, actions
                    )
                    if self.fixed_order:
                        agent_order = list(range(self.off_num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.off_num_agents))
                    for agent_id in agent_order:
                        self.off_actor[agent_id].turn_on_grad()
                        # actor preds
                        actor_values = self.off_actor[agent_id].train_values(
                            sp_obs[agent_id], actions[agent_id]
                        )
                        # critic preds
                        critic_values = get_values()
                        # update
                        actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                        self.off_actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.off_actor[agent_id].actor_optimizer.step()
                        self.off_actor[agent_id].turn_off_grad()
                        update_actions(agent_id)
                else:
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.off_num_agents):
                            actions.append(
                                self.off_actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, dim)
                    if self.fixed_order:
                        agent_order = list(range(self.off_num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.off_num_agents))
                    for agent_id in agent_order:
                        self.off_actor[agent_id].turn_on_grad()
                        # train this agent
                        actions[agent_id] = self.off_actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                        actions_t = torch.cat(actions, dim=-1)
                        value_pred = self.off_critic.get_values(sp_share_obs, actions_t)
                        actor_loss = -torch.mean(value_pred)
                        self.off_actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.off_actor[agent_id].actor_optimizer.step()
                        self.off_actor[agent_id].turn_off_grad()
                        # actions[agent_id] = self.off_actor[agent_id].get_actions(
                        #     sp_obs[agent_id], False
                        # )
                # soft update
                for agent_id in range(self.off_num_agents):
                    self.off_actor[agent_id].soft_update()
            self.off_critic.soft_update()
