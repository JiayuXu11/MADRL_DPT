import numpy as np
import math
from harl.envs.multi_lt_transship.generator_vec import *





S_I = 10
S_O = 10

class VecBaseline:
    def __init__(self, args):
        # heuristic parameters
        self.k = args.get('k', 0)
        self.stationary_tf = args['stationary_tf']
        self.args = args
        self.num_envs = int(args.get('num_envs', 1))
        self.num_agents = int(args.get('num_agents', 1))
        self.pre_ema = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        self.pre_ema_d_sqr = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        self.unit_shape = (self.num_envs, self.num_agents, 1)

    @staticmethod
    def calculate_mu_and_d_sqr(t, ema, ema_d_sqr, pre_ema, pre_ema_d_sqr, lt_sqr):
        mu = ema * t
        d_sqr = ema_d_sqr * t
        d_sqr += lt_sqr * np.square(ema)
        return mu, d_sqr

    @staticmethod
    def get_hist_demand(env):
        # 返回 shape: (E, A, step_num)


        return env.demand[:, :, :env.step_index]

    def get_leadtime_est(self, env):
        """
        Estimate the mean and variance of lead time for each agent in every environment.

        Set:
            self.unit_shape: (E, A, 1)
            lead_time_mean: (E, A)
            lead_time_sqr: (E, A)
            lt_spec: lead time generator specification, list or single spec
        """
        E, A = env.num_envs, env.num_agents
        self.unit_shape = (E, A, 1)
        lead_time_mean = np.zeros((E, A), dtype=np.float32)
        lead_time_sqr = np.zeros((E, A), dtype=np.float32)
        lt_spec = env.args.get("LT_generators")
        if isinstance(lt_spec, list):
            for stage_spec in lt_spec:
                stage_gen = create_generator(self.unit_shape, stage_spec)
                gen_mean, gen_std = stage_gen.get_theoretical_mean_std()
                lead_time_mean += gen_mean
                lead_time_sqr += gen_std ** 2
        else:
            lt_gen = create_generator(self.unit_shape, lt_spec)
            lead_time_mean, lead_time_sqr = lt_gen.get_theoretical_mean_std()

        return lead_time_mean, lead_time_sqr

    def get_demand_est(self, env):
        """
        Estimate the mean and variance of demand for each agent in every environment.

        For stationary demand, uses generator's theoretical mean and variance.
        For non-stationary demand, uses exponential moving average (EMA) over observed data (with alpha configurable).
        Updates and returns the estimated mean and variance for the current timestep.

        Set:
            step_index: (E,)
            hist_demand: (E, A, max_t)
            pre_demand: (E, A)
            demand_spec: demand generator specification, list or single spec
            self.pre_ema: (E, A)
            self.pre_ema_d_sqr: (E, A)

        Returns:
            demand_mean_est: (E, A) estimated mean demand
            demand_sqr_est:  (E, A) estimated variance of demand
        """
        if self.num_envs != env.num_envs:
            self.num_envs = env.num_envs
            self.unit_shape = (env.num_envs, env.num_agents, 1)
            self.pre_ema = np.zeros((env.num_envs, env.num_agents), dtype=np.float32)
            self.pre_ema_d_sqr = np.zeros_like(self.pre_ema)
        method = self.args.get(
            "baseline_forecast_method", self.args.get("estimate_demand_method", "ema")
        )
        if not self.stationary_tf and method in {"arima", "xgboost"}:
            from harl.envs.multi_lt_transship.demand_forecasting import baseline_forecasts
            if getattr(self, "_forecast_source", None) is not env.demand:
                self._forecast_source = env.demand
                self._baseline_mean, self._baseline_std = baseline_forecasts(
                    env.demand, {**env.args, **self.args}, method
                )
            indices = np.arange(env.num_envs)
            steps = np.minimum(env.step_index.reshape(-1), env.episode_length)
            mean = self._baseline_mean[indices, :, steps]
            variance = np.square(self._baseline_std[indices, :, steps])
            self.pre_ema, self.pre_ema_d_sqr = mean, variance
            return mean, variance
        if not self.stationary_tf and method != "ema":
            raise ValueError(f"Unsupported baseline forecast method: {method}")

        # The EMA branch below is unchanged from the original baseline.
        E, A = env.num_envs, env.num_agents
        step_index = env.step_index.reshape(-1)
        max_t = int(step_index.max())
        hist_demand = np.zeros((E, A, max_t), dtype=env.demand.dtype)
        pre_demand = np.zeros((E, A), dtype=env.demand.dtype)
        for e in range(E):
            t_idx = int(step_index[e])
            if t_idx > 0:
                hist_demand[e, :, :t_idx] = env.demand[e, :, :t_idx]
                pre_demand[e, :] = env.demand[e, :, t_idx - 1]

        if self.stationary_tf:
            demand_spec = self.args.get("demand_generators")
            demand_gen = create_generator(self.unit_shape, demand_spec)
            demand_mean_est, demand_sqr_est = demand_gen.get_theoretical_mean_std()
        else:
            alpha = float(env.args.get("alpha", 0.3))
            if len(hist_demand) == 0:
                demand_mean_est = pre_demand
                demand_sqr_est = np.zeros_like(pre_demand, dtype=np.float32)
            else:
                demand_mean_est = alpha * pre_demand + (1 - alpha) * self.pre_ema
                demand_sqr_est = alpha * np.square(pre_demand - demand_mean_est) + (1 - alpha) * self.pre_ema_d_sqr
            self.pre_ema = demand_mean_est
            self.pre_ema_d_sqr = demand_sqr_est

        return demand_mean_est, demand_sqr_est


    def get_order_action(self, env, obs):
        """
        Calculate heuristic replenishment order quantity for each agent in every environment subject to moq.

        Returns:
            order_heu: np.ndarray of shape (E, A)
        """
        inv = env.inventory
        orders_num = np.sum(env.intransit_hist, axis=2)  # shape: (E, A)
        inv_pos = inv + orders_num

        lead_time_mean, lead_time_sqr = self.get_leadtime_est(env)
        demand_mean_est, demand_sqr_est = self.get_demand_est(env)
        rep_converage = lead_time_mean + 1

        if self.stationary_tf:
            mu_t_plus_l = demand_mean_est * rep_converage
            sig_t_plus_l = np.sqrt(demand_sqr_est * rep_converage + lead_time_sqr * np.square(demand_mean_est))
            s = mu_t_plus_l + self.k * sig_t_plus_l
        else:
            mu_t_plus_l, d_sqr_t_plus_l = self.calculate_mu_and_d_sqr(
                rep_converage, demand_mean_est, demand_sqr_est, self.pre_ema, self.pre_ema_d_sqr, lead_time_sqr)
            sig_t_plus_l = np.sqrt(d_sqr_t_plus_l)
            s = mu_t_plus_l + self.k * sig_t_plus_l

        moq = env.MOQ
        order_heu = np.where(inv_pos < s, np.maximum(moq, np.round(s - inv_pos)), 0)

        return order_heu.astype(np.int32)

    def get_action(self, env, obs):
        """
        Returns:
            actions: np.ndarray of shape (E, A, 2)
        """
        order_amounts = self.get_order_action(env, obs)
        transship_amounts = np.zeros_like(order_amounts)
        actions = np.stack([order_amounts, transship_amounts], axis=-1)
        return actions

class VecEppen(VecBaseline):
    def __init__(self, args):
        super().__init__(args)
        self.use_demand_correlation = args.get('eppen_use_demand_correlation', False)
        if not isinstance(self.use_demand_correlation, (bool, np.bool_)):
            raise ValueError('eppen_use_demand_correlation must be a boolean')
        if self.use_demand_correlation:
            from harl.envs.multi_lt_transship.eppen_correlation import load_training_correlation
            self.training_demand_correlation, self.correlation_train_length = load_training_correlation(args, self.num_agents)

    def get_order_action_eppen(self, env, obs):
        """
        Implements the VecEppen multi-echelon basestock policy for the multi-agent, multi-lead-time transshipment environment.

        Returns:
            order_eppen: np.ndarray of shape (E, A)
        """
        E, A = env.num_envs, env.num_agents
        lead_time_mean, lead_time_sqr = self.get_leadtime_est(env)

        demand_mean_est, demand_sqr_est = self.get_demand_est(env)
        lead_time_sig = np.sqrt(lead_time_sqr)
        demand_sig_est = np.sqrt(demand_sqr_est)

        lead_time_mean, lead_time_sqr = self.get_leadtime_est(env)
        unified_avg_lt_mean = np.mean(lead_time_mean, axis=1, keepdims=True)

        inv = env.inventory
        orders_num = np.sum(env.intransit_hist, axis=2)
        inv_pos = inv + orders_num

        h, b = env.H, env.B
        from scipy import stats
        k = stats.norm.ppf(b/(b+h))
        replenishment_cycle = np.ones((E, 1), dtype=np.float32)

        unified_total_coverage = unified_avg_lt_mean + replenishment_cycle
        unified_cum_mu_within_total_coverage = np.sum(demand_mean_est, axis=1, keepdims=True) * unified_total_coverage

        sig_within_total_coverage = np.sqrt(demand_sig_est ** 2 * (unified_avg_lt_mean + replenishment_cycle) + (lead_time_sig * demand_mean_est) ** 2)
        sum_sig_within_total_coverage = np.sum(sig_within_total_coverage, axis=1, keepdims=True)
        sig_weight_within_total_coverage = sig_within_total_coverage / np.clip(sum_sig_within_total_coverage, 1e-8, None)

        demand_sig_sqr_sum = np.sum(demand_sig_est ** 2, axis=1, keepdims=True)
        if self.use_demand_correlation:
            from harl.envs.multi_lt_transship.eppen_correlation import pooled_demand_variance
            demand_sig_sqr_sum = pooled_demand_variance(demand_sig_est, self.training_demand_correlation)
        demand_mu_sum = np.sum(demand_mean_est, axis=1, keepdims=True)
        # unified_lt_sig = np.sqrt(np.sum(lead_time_sig ** 2, axis=1, keepdims=True)) / lead_time_sig.shape[1]
        unified_lt_sig = np.mean(lead_time_sig, axis=1, keepdims=True)

        re_sig_within_total_coverage = np.sqrt(
            unified_total_coverage * demand_sig_sqr_sum
            + (demand_mu_sum ** 2) * unified_lt_sig ** 2
        )

        unified_avg_k = np.mean(k, keepdims=True)
        # step 1: calculate the basestock level for total system
        cum_sig_lt_of_total = unified_avg_k * re_sig_within_total_coverage
        basestock_of_system = unified_cum_mu_within_total_coverage + cum_sig_lt_of_total

        # step 2:  calculate the basestock level separately
        # cum_mu_within_total_coverage = unified_total_coverage * demand_mean_est
        # unified_cum_mu_within_total_coverage = unified_total_coverage * demand_mu_sum
        # cum_sig_within_total_coverage = sig_weight_within_total_coverage * (basestock_of_system - unified_cum_mu_within_total_coverage)
        cum_mu_within_sep_coverage = (lead_time_mean + replenishment_cycle) * demand_mean_est
        sum_cum_mu_within_sep_coverage = np.sum(cum_mu_within_sep_coverage, axis=1, keepdims=True)
        cum_sig_within_sep_coverage = sig_weight_within_total_coverage * (basestock_of_system - sum_cum_mu_within_sep_coverage)

        sep_basestock = cum_mu_within_sep_coverage + cum_sig_within_sep_coverage
        order_gap = sep_basestock - inv_pos

        # step 3: make replenishment
        moq = env.MOQ
        order_eppen = np.where(order_gap > 0, np.maximum(moq, np.round(order_gap)), 0.0)

        return order_eppen

    def get_action(self, env, obs):
        order_amounts = self.get_order_action_eppen(env, obs)
        transship_amounts = np.zeros_like(order_amounts)
        actions = np.stack([order_amounts, transship_amounts], axis=-1)  # (E, A, 2)
        return actions
