"""HAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.basic_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class HAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize HAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.use_advantage_split = args.get("use_advantage_split", False)
        self.gamma = args.get("gamma", 0.99)

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 12:
            (
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                lt_today_batch,
                lt_window_batch,
                adv_targ_order,
                adv_targ_transship,
            ) = sample
            factor_batch = np.ones(adv_targ.shape[0])
        else:
            (
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                factor_batch,
                lt_today_batch,
                lt_window_batch,
                adv_targ_order,
                adv_targ_transship,
            ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        adv_targ_order = check(adv_targ_order).to(**self.tpdv)
        adv_targ_transship = check(adv_targ_transship).to(**self.tpdv)
        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update with advantage splitting for multi-discrete actions
        # For transshipment env: action[:, 0] is order, action[:, 1] is transship
        # action_log_probs shape: (batch_size, num_actions) for MultiDiscrete
        # old_action_log_probs_batch shape: (batch_size, num_actions)

        if self.use_advantage_split:
            # Use split advantages for per-action policy loss
            # action_log_probs: (batch_size, 2) for order and transship
            # We compute separate importance weights for each action

            # Use the split advantages from the sample batch
            # adv_targ_order and adv_targ_transship are already flattened from the buffer
            order_adv_batch = adv_targ_order
            transship_adv_batch = adv_targ_transship

            # Extract log probs for each action
            # action_log_probs[:, 0] is for order, action_log_probs[:, 1] is for transship
            order_log_prob = action_log_probs[:, 0:1]  # (batch_size, 1)
            transship_log_prob = action_log_probs[:, 1:2]  # (batch_size, 1)

            order_old_log_prob = old_action_log_probs_batch[:, 0:1]  # (batch_size, 1)
            transship_old_log_prob = old_action_log_probs_batch[:, 1:2]  # (batch_size, 1)

            # Compute importance weights per action
            order_imp_weights = torch.exp(order_log_prob - order_old_log_prob)
            transship_imp_weights = torch.exp(transship_log_prob - transship_old_log_prob)

            # Compute surrogate losses for each action
            order_surr1 = order_imp_weights * order_adv_batch
            order_surr2 = torch.clamp(order_imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * order_adv_batch

            transship_surr1 = transship_imp_weights * transship_adv_batch
            transship_surr2 = torch.clamp(transship_imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * transship_adv_batch

            # Combine losses
            order_loss = torch.min(order_surr1, order_surr2)
            transship_loss = torch.min(transship_surr1, transship_surr2)
            combined_loss = order_loss + transship_loss  # (batch_size, 1)

            if self.use_policy_active_masks:
                policy_action_loss = (
                    -torch.sum(factor_batch * combined_loss, dim=-1, keepdim=True) * active_masks_batch
                ).sum() / active_masks_batch.sum()
            else:
                policy_action_loss = -torch.sum(factor_batch * combined_loss, dim=-1, keepdim=True).mean()

            # Use combined importance weights for logging
            imp_weights = (order_imp_weights + transship_imp_weights) / 2.0

        else:
            # Standard HAPPO: aggregate importance weights across actions
            imp_weights = getattr(torch, self.action_aggregation)(
                torch.exp(action_log_probs - old_action_log_probs_batch),
                dim=-1,
                keepdim=True,
            )
            surr1 = imp_weights * adv_targ
            surr2 = (
                torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * adv_targ
            )

            if self.use_policy_active_masks:
                policy_action_loss = (
                    -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                    * active_masks_batch
                ).sum() / active_masks_batch.sum()
            else:
                policy_action_loss = -torch.sum(
                    factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
                ).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()  # add entropy term

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type, order_advantages=None, transship_advantages=None, order_valid_mask=None):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) standard advantages for all actions.
            state_type: (str) type of state.
            order_advantages: (np.ndarray, optional) advantages for order action (action[:, 0]).
            transship_advantages: (np.ndarray, optional) advantages for transship action (action[:, 1])
            order_valid_mask: (np.ndarray, optional) mask for order action (action[:, 0]).
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        # Normalize advantages
        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

            # Normalize split advantages if provided
            if order_advantages is not None:
                order_advantages_copy = order_advantages.copy()
                order_advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
                mean_order = np.nanmean(order_advantages_copy)
                std_order = np.nanstd(order_advantages_copy)
                order_advantages = (order_advantages - mean_order) / (std_order + 1e-5)
                order_advantages = order_advantages * order_valid_mask

            if transship_advantages is not None:
                transship_advantages_copy = transship_advantages.copy()
                transship_advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
                mean_transship = np.nanmean(transship_advantages_copy)
                std_transship = np.nanstd(transship_advantages_copy)
                transship_advantages = (transship_advantages - mean_transship) / (std_transship + 1e-5)


        for _ in range(self.ppo_epoch):
            data_generator = actor_buffer.recurrent_generator_actor(
                advantages, self.actor_num_mini_batch, self.data_chunk_length, order_advantages, transship_advantages
            )
            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (list[OnPolicyActorBuffer]) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            num_agents: (int) number of agents.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        # 为保证训练次数一致/不爆内存，这里对actor_mini_batch和ppo_epoch进行调整
        actor_num_mini_batch = self.actor_num_mini_batch * num_agents
        ppo_epoch = max(self.ppo_epoch // num_agents, 5)
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                agent_id_fixed = min(agent_id, advantages.shape[0] - 1)
                advantages_ori = advantages[agent_id_fixed].copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages[agent_id_fixed].copy()
                advantages_copy[actor_buffer[agent_id_fixed].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            # 对每个agent的advantages进行标准化
            advantages_list = []
            for agent_id in range(num_agents):
                mean_advantages = np.nanmean(advantages_copy_list[agent_id])
                std_advantages = np.nanstd(advantages_copy_list[agent_id])
                normalized_adv = (advantages_ori_list[agent_id] - mean_advantages) / (std_advantages + 1e-5)
                advantages_list.append(normalized_adv)
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])

        for _ in range(ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    sampler = actor_buffer[agent_id].get_sampler_for_recurrent_generator_actor(
                        actor_num_mini_batch, self.data_chunk_length
                    )
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        actor_num_mini_batch,
                        self.data_chunk_length,
                        sampler=sampler
                    )
                elif self.use_naive_recurrent_policy:
                    sampler = actor_buffer[agent_id].get_sampler_for_naive_recurrent_generator_actor(
                        actor_num_mini_batch
                    )
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], actor_num_mini_batch, sampler=sampler
                    )
                else:
                    sampler = actor_buffer[agent_id].get_sampler_for_feed_forward_generator_actor(
                        actor_num_mini_batch
                    )
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], actor_num_mini_batch, sampler=sampler
                    )
                data_generators.append(data_generator)

            for _ in range(actor_num_mini_batch):
                batches = [[] for _ in range(8)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(8):
                        batches[i].append(sample[i])
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[7][0] is None:
                    batches[7] = None
                else:
                    batches[7] = np.concatenate(batches[7], axis=0)
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.share_param_update(
                    tuple(batches), num_agents
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = ppo_epoch * actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_update(self, sample, num_agents):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD.
        Args:
            sample: (Tuple) contains data batch containing all agents.
            num_agents: (int) number of agents.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 8:
            (
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            ) = sample
        else:
            (
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                _,
            ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        # 概率连加loss部分
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )
        # active_masks_batch的shape为[num_agents * batch_size, 1], 所以需要先将其拆乘[num_agents, batch_size, 1]
        active_masks_batch_reshaped = active_masks_batch.view(num_agents, -1, active_masks_batch.shape[-1])
        active_masks_batch_reshaped_mean = torch.mean(active_masks_batch_reshaped, dim=0)
        active_masks_batch_reshaped_mean[active_masks_batch_reshaped_mean < 1] = 0

        # 此处imp_weights和adv_targ的shape为[num_agents * batch_size, dim], 所以需要先将其拆乘[num_agents, batch_size, dim]
        imp_weights_reshaped = imp_weights.view(num_agents, -1, imp_weights.shape[-1])
        adv_targ_reshaped = adv_targ.view(num_agents, -1, adv_targ.shape[-1])
        # 动作概率连乘
        imp_weights_prod = torch.prod(imp_weights_reshaped, dim=0)
        # 动作优势连加求平均
        adv_targ_mean = torch.mean(adv_targ_reshaped, dim=0)
        # 概率连乘loss部分
        surr3 = imp_weights_prod * adv_targ_mean
        surr4 = torch.clamp(imp_weights_prod, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_mean

        if self.use_policy_active_masks:
            policy_action_sum_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum() / num_agents
            policy_action_prod_loss = (
                -torch.sum(torch.min(surr3, surr4), dim=-1, keepdim=True)
                * active_masks_batch_reshaped_mean
            ).sum() / active_masks_batch_reshaped_mean.sum()
        else:
            policy_action_sum_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean() / num_agents
            policy_action_prod_loss = -torch.sum(
                torch.min(surr3, surr4), dim=-1, keepdim=True
            ).mean()

        policy_loss = (1 - self.prod_prob_weight) * policy_action_sum_loss + self.prod_prob_weight * policy_action_prod_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef / num_agents).backward()  # add entropy term

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
