import numpy as np
from gym import spaces
import copy
from harl.envs.multi_lt_transship import ENV_ARGS_DEFAULT, DISTANCE_LIST, TEST_DIR, EVAL_DIR
from harl.envs.multi_lt_transship.generator_vec import create_generator

class VecTransshipEnv:
    """
    Fully vectorized multi-agent transshipment environment (no inheritance, no per-env/agent loops).

    Shape Convention:
      - E: num_envs
      - A: num_agents
      - T: episode_length
      - M: max_intransit_items (max lead time for all stages combined)
      - B: max_inventory_batches

    State Tensors:
      - demand: (E, A, T) - demand per env/agent/timestep
      - LT: (E, A, T) - total lead time per env/agent/timestep (simplified to single value)
      - shipping_loss: (E, A, T) - shipping loss probability
      - inventory: (E, A) - current inventory level
      - delivery_schedule: (E, A, T) - scheduled arrivals per timestep
      - order_today: (E, A) - current order quantities
      - transship_intend: (E, A) - transshipment intentions (+demand, -supply)
      - transship_matrix: (E, A, A) - actual transshipment flows [from_i, to_j]
      - reward: (E, A) - per-step rewards
      - reward_cum: (E, A) - cumulative rewards

    Internal state uses pure numpy arrays with explicit agent axis.
    """

    def __init__(self, args: dict):

        self.args = copy.deepcopy(ENV_ARGS_DEFAULT)
        self.args.update(args)
        # Core sizes
        self.num_envs = int(self.args.get("num_envs", 1))  # E
        self._configured_num_envs = self.num_envs
        self.eval_dataset_mode = self.args.get("eval_dataset_mode", "auto")
        if self.eval_dataset_mode == "auto":
            spec = self.args.get("demand_generators", {})
            self.eval_dataset_mode = "all" if isinstance(spec, dict) and spec.get("gen_func") == "shanshu" else "sample"
        if self.eval_dataset_mode not in {"all", "sample"}:
            raise ValueError("eval_dataset_mode must be auto, all or sample")
        self.episode_length = int(self.args["episode_length"])  # T
        self.num_agents = int(self.args["num_agents"])  # A
        self.n_agents = self.num_agents
        self.reward_type = self.args["reward_type"]

        # Advantage-splitting / logging helpers
        self.advantage_split_mode = self.args.get("advantage_split_mode", "lt_today")  # lt_today, lt_avg, fixed, no
        self.advantage_fixed_lt = int(self.args.get("advantage_fixed_lt", 0))
        self.price_agent_feature_mode = self.args.get("price_agent_feature_mode", None)
        self.include_price_signal_in_obs = bool(
            self.args.get("include_price_signal_in_obs", False)
        )

        # Read transshipment-related parameters first (needed for mechanism validation)
        self.transship_type = self.args['transship_type']  # 转运类型 'pure_autonomy': 自己决策transship的量,
                                                           # 'half_autonomy': 只决策是否参与transship, 'no': 不参与transship, 'force_to_transship': 强制转运
        allocate_config = self.args.get('product_allocation_method', {'allocate_func': 'baseline'})
        if isinstance(allocate_config, dict):
            self.product_allocation_method = allocate_config.get('allocate_func', 'baseline')
            self.allocate_args = allocate_config.get('allocate_args', {})
        else:
            self.product_allocation_method = str(allocate_config)
            self.allocate_args = {}
        self.revenue_allocation_method = self.args['revenue_allocation_method']  # 'constant', 'ratio' or 'agent_specific'
        self.constant_transshipment_price = self.args['constant_transshipment_price']  # only used when revenue_allocation_method is 'constant'

        self.mechanism_agent_type = str(
            self.args.get("mechanism_agent_type", "none")
        ).lower()
        valid_mech_types = {
            "none",
            "pricing",
            "cen_allocation",
            "dec_allocation",
        }
        if self.mechanism_agent_type not in valid_mech_types:
            raise ValueError(
                f"Unsupported mechanism_agent_type: {self.mechanism_agent_type}"
            )

        # mechanism_reward_mode: 'avg_reward' (default) or 'pooled_cost_saving'
        self.mechanism_reward_mode = str(
            self.args.get("mechanism_reward_mode", "avg_reward")
        ).lower()
        if self.mechanism_reward_mode not in {"avg_reward", "pooled_cost_saving"}:
            raise ValueError(
                f"Unsupported mechanism_reward_mode: {self.mechanism_reward_mode}"
            )
        if self.mechanism_agent_type in ("dec_allocation"):
            if self.product_allocation_method != "mechanism_agent":
                raise ValueError(
                    "mechanism_agent_type requires product_allocation_method 'mechanism_agent'."
                )
        self.actor_obs_type_mechanism = self.args.get(
            "actor_obs_type_mechanism", "intentions_only"
        ) # 'intentions_only' | 'full' | 'compressed'
        self.critic_obs_type_mechanism = self.args.get(
            "critic_obs_type_mechanism", "full"
        ) #  'full' | 'compressed'
        # 新参数：即使 mechanism_agent_type == "pricing" 也不在obs中包含价格信号
        self.pricing_without_telling = bool(
            self.args.get("pricing_without_telling", False)
        )
        self._include_price_feature = (
            self.include_price_signal_in_obs
            or (self.mechanism_agent_type == "pricing" and not self.pricing_without_telling)
        )

        # Action/obs configuration
        self.action_type = self.args["action_type"]  # 'discrete' | 'multi_discrete' | 'continuous' | 'heu_discrete'
        self.actor_obs_type = self.args["actor_obs_type"]
        self.critic_obs_type = self.args["critic_obs_type"]

        # Cost and operational parameters (per-agent constants)
        self.H = self._broadcast_param(self.args["H"])  # holding
        self.B = self._broadcast_param(self.args["B"])  # penalty/backlog
        self.P = self._broadcast_param(self.args["P"])  # selling price
        self.C = self._broadcast_param(self.args["C"])  # ordering cost
        self.K = self._broadcast_param(self.args["K"])  # fixed order cost
        self.MOQ = self._broadcast_param(self.args["MOQ"])  # minimum order quantity baseline

        # Bounds/steps
        self.order_ub = self._broadcast_param(self.args["order_ub"])  # scalar per agent
        self.order_step_size = float(self.args["order_step_size"])  # scalar global
        self.transship_lb = self._broadcast_param(self.args["transship_lb"])  # scalar per agent
        self.transship_ub = self._broadcast_param(self.args["transship_ub"])  # scalar per agent
        self.transship_step_size = float(self.args["transship_step_size"])  # scalar global

        # Shipping/distance
        distance_matrix = DISTANCE_LIST[self.args['distance_matrix_index']][:self.num_agents, :self.num_agents]
        # ensure (A,A)
        assert distance_matrix.shape[0] >= self.num_agents and distance_matrix.shape[1] >= self.num_agents
        self.distance_matrix = distance_matrix.astype(np.float32)
        self.shipping_cost_per_distance = float(self.args["shipping_cost_per_distance"])  # scalar
        self.shipping_cost_matrix = self.shipping_cost_per_distance * self.distance_matrix

        # Misc
        self.backlog_tf = bool(self.args["backlog_tf"])  # backlog or lost sales
        self.gamma = float(self.args["gamma"])  # discount factor
        self.state_normalize = bool(self.args["state_normalize"])  # normalize obs
        self.state_clip_list = list(self.args["state_clip_list"])  # [lb, ub]
        self.reward_scale_tf = bool(self.args["reward_scale_tf"])
        self.reactive_tf = bool(self.args["reactive_tf"])  # reactive vs proactive fulfillment order
        self.self_interest_rate = float(self.args.get("self_interest_rate", 1.0))  # 自私程度，1.0表示完全自私，0.0表示完全利他

        # price_agent_signal initialization
        self.price_agent_signal = np.ones((self.num_envs, ), dtype=np.float32) * self.constant_transshipment_price
        self.distance_matrix_index = self.args['distance_matrix_index']
        # max lt
        self.max_lt = self.args.get("LT_generators")[0]['gen_args']['max_num']
        # 保存最近max_lt天的尚未arrive的intransit历史
        self.intransit_hist = np.zeros((self.num_envs, self.num_agents, self.max_lt), dtype=np.float32)

        # Initialize mechanism (from transship_multi_lt.py logic)
        self._init_mechanism()

        # Derived: observation/action spaces (per-agent; E is runtime batch only)
        self.observation_space = self._build_actor_obs_spaces()
        self.share_observation_space = self._build_critic_obs_spaces()
        self.action_space = self._build_action_spaces()
        mech_share_dim = self._calc_mechanism_critic_obs_dim(self.critic_obs_type_mechanism)
        mech_actor_dim = self._calc_mechanism_obs_dim(self.actor_obs_type_mechanism)
        self.mechanism_share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(mech_share_dim,),
                dtype=np.float32,
            )
        ]
        self.mechanism_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(mech_actor_dim,), dtype=np.float32)]
        self.mechanism_action_space = self._build_mechanism_action_space()
        self._pending_mechanism_data = None



    def _init_mechanism(self):
        """
        Initialize transshipment allocation mechanism based on product_allocation_method.
        Replicates logic from transship_multi_lt.py lines 62-82.
        """
        allocate_config = self.args.get('product_allocation_method', 'baseline')

        if isinstance(allocate_config, dict):
            method = allocate_config.get('allocate_func', 'baseline')
            self.allocate_args = allocate_config.get('allocate_args', {})
        else:
            method = str(allocate_config)
            self.allocate_args = {}

        self.product_allocation_method = method

        # Import mechanism class
        if method == 'homo_distance':
            from harl.envs.multi_lt_transship.dist_mechanism.homo_dist import HomoDistribution as mechanism
        elif method == 'least_distance':
            from harl.envs.multi_lt_transship.dist_mechanism.least_dist import LeastDistanceDistribution as mechanism
        elif method == 'no':
            from harl.envs.multi_lt_transship.dist_mechanism.no_mechanism import NoMechanism as mechanism
        elif method == 'anup':
            from harl.envs.multi_lt_transship.dist_mechanism.anupindi_mechanism import AnupindiDistribution as mechanism
        elif method == 'baseline':
            from harl.envs.multi_lt_transship.dist_mechanism.baseline_mechanism import BaselineDistribution as mechanism
        elif method == 'best':
            from harl.envs.multi_lt_transship.dist_mechanism.best_mechanism import BestDistribution as mechanism
        elif method == 'mechanism_agent':
            from harl.envs.multi_lt_transship.dist_mechanism.agent_mechanism import AgentMechanism as mechanism
        else:
            raise Exception(f'wrong product_allocation_method: {method}')

        # Instantiate mechanism (will need vectorized version)
        self.dist_mechanism = mechanism(self.distance_matrix, self.allocate_args, self)

    # ---------------------------- Public API ----------------------------
    def reset(self, usage: str = "train"):
        """
        Reset all environments. if self.mechanism_agent_type == "pricing": 返回的是mechanism agent的obs，否则返回常规agent的obs

        Args:
            usage: 'train' | 'eval' | 'test'

        Returns:
            actor_obs: (E, A, actor_obs_dim) or (E, 1, actor_obs_dim) if for_mechanism=True
            critic_obs: (E, A, critic_obs_dim) or (E, 1, critic_obs_dim) if for_mechanism=True
            avail_actions: None
        """
        self.usage = usage
        self.train_tf = usage == "train"
        count = self._configured_num_envs
        if usage in {"eval", "test"} and self.eval_dataset_mode == "all":
            from pathlib import Path
            directory = (TEST_DIR if usage == "test" else EVAL_DIR).format(self.args.get("dataset_name", "default"))
            count = np.loadtxt(Path(directory) / "agent_0" / "demand.csv", delimiter=",", ndmin=2).shape[1]
        if count != self.num_envs:
            self.num_envs = count
            self.price_agent_signal = np.full(count, self.constant_transshipment_price, dtype=np.float32)
            self._init_mechanism()
        # Runtime state tensors (allocated in reset)
        self._allocate_runtime_state()

        # Precompute stochastic sequences and estimates for all envs and timesteps (no per-step recompute)
        self._precompute_sequences()
        self._reset_runtime_state()

        if self.mechanism_agent_type == "pricing":
            actor_obs = self._compute_mechanism_actor_obs()
            critic_obs = self._compute_mechanism_critic_obs()
        else:
            actor_obs = self._compute_actor_obs(self.actor_obs_type)
            critic_obs = self._compute_critic_obs(self.critic_obs_type)
        avail_actions = self._get_avail_actions()
        return actor_obs, critic_obs, avail_actions

    def step(self, actions: np.ndarray):
        f"""
        Vectorized step for all environments, 此处actions指的是常规agent的actions
        if self.mechanism_agent_type == "none", 则返回常规agent的obs和critic_obs，反之返回mechanism agent所需的obs和critic_obs
        if self.mechanism_agent_type in {"none", "pricing"}, 则state_update,并且更新step_index
        if self.mechanism_agent_type in {"cen_allocation", "dec_allocation"}, 则不执行state_update和step_index

        Args:
            actions: shape (E, A, action_dim)

        Returns:
            For mechanism_agent_type is not 'none':
                actor_obs: (E, A, actor_obs_dim) or (E, 1, actor_obs_dim) if self.mechanism_agent_type == "none"
                critic_obs: (E, A, critic_obs_dim) or (E, 1, critic_obs_dim) if self.mechanism_agent_type == "none"
                rewards: (E, A) 若没有执行state_update则返回None
                done_agents: (E, A) 若没有执行state_update则返回None
                infos: (E, A) list of list of dicts 若没有执行state_update则返回None
                avail_actions: 直接扔None
                lt_today: (E, A) replenishment的policy loss从t + lt_today到后面的所有 若没有执行state_update则返回None
                lt_window: (E, A) transship的policy loss的覆盖horizon的长度，如果没有设置则返回None，表示horizon为后面的所有 若没有执行state_update则返回None
            Otherwise: standard env step tuple as in previous implementation.
        """
        E, A, T = self.num_envs, self.num_agents, self.episode_length
        t = np.minimum(self.step_index, T - 1)
        mech_type = self.mechanism_agent_type

        # 获取常规agent的action并转换为order和transship_intend
        order_today, transship_intend = self._action_map(actions, t)
        self.transship_intend = transship_intend

        # 对于 cen_allocation 和 dec_allocation: 只保存数据，等待 step_mechanism
        if mech_type in {"cen_allocation", "dec_allocation"}:
            if self._pending_mechanism_data is not None:
                raise RuntimeError("step_mechanism must be called before next step.")
            self.order_today = order_today
            self._pending_mechanism_data = {
                "order_today": order_today,
                "transship_intend": transship_intend,
                "t_index": t,
            }
            # 返回mechanism agent的obs
            actor_obs = self._compute_mechanism_actor_obs()
            critic_obs = self._compute_mechanism_critic_obs()
            return [actor_obs, critic_obs, None, None, None, None, None, None]

        # 对于 none 和 pricing: 执行 state_update
        self.order_today = order_today

        # Allocation via mechanism -> transship_matrix (E,A,A), payment (E,A)
        transship_matrix, payment = self._alloc_transshipment(transship_intend)
        self.transship_matrix = transship_matrix
        self.transship_actual = transship_matrix.sum(axis=2)

        # State update
        rewards, lt_today, lt_window, demand_t = self._state_update(
            order_today, transship_matrix, payment, t
        )

        self.step_index = self.step_index + 1

        # Build done and infos
        done_env = self.step_index >= T
        done_agents = np.repeat(done_env[:, None], A, axis=1)
        if self.train_tf:
            infos = [
                [{} for _ in range(A)]
                for e in range(E)
            ]
        else:
            infos = self._build_log_infos(lt_today, lt_window, demand_t)

        # Add mechanism_reward to infos (based on mechanism_reward_mode)
        if self.mechanism_reward_mode == "pooled_cost_saving":
            mechanism_reward = self.pooled_cost_saving_avg
        else:  # default: avg_reward
            mechanism_reward = rewards.mean(axis=1)  # (E,)
        for e in range(E):
            for a in range(A):
                infos[e][a]["mechanism_reward"] = float(mechanism_reward[e])

        avail_actions = self._get_avail_actions()

        # 对于 pricing: 返回mechanism agent的obs（但已经完成了state_update）
        if mech_type == "pricing":
            actor_obs = self._compute_mechanism_actor_obs()
            critic_obs = self._compute_mechanism_critic_obs()
            return [actor_obs, critic_obs, rewards, done_agents, infos, avail_actions, lt_today, lt_window]

        # 对于 none: 返回常规agent的obs
        actor_obs = self._compute_actor_obs(self.actor_obs_type)
        critic_obs = self._compute_critic_obs(self.critic_obs_type)
        return [actor_obs, critic_obs, rewards, done_agents, infos, avail_actions, lt_today, lt_window]

    def step_mechanism(self, mechanism_actions):
        """
        input只包含mechanism agent的action,记得保存mechanism对应的action，保存到self._pending_mechanism_data中，输出返回常规agent的obs和critic_obs
        if self.mechanism_agent_type in {"cen_allocation", "dec_allocation"}, 则执行state_update,并且更新step_index
        if self.mechanism_agent_type in {"none", "pricing"}, 则不执行state_update和step_index
        Args:
            mechanism_actions: shape (E, A)
        Returns:
            actor_obs: (E, A, actor_obs_dim)
            critic_obs: (E, A, critic_obs_dim)
            rewards: (E, A) 若没有执行state_update则返回None
            done_agents: (E, A) 若没有执行state_update则返回None
            infos: (E, A) list of list of dicts 若没有执行state_update则返回None
            avail_actions: 直接扔None
            lt_today: (E, A) replenishment的policy loss从t + lt_today到后面的所有 若没有执行state_update则返回None
            lt_window: (E, A) transship的policy loss的覆盖horizon的长度，如果没有设置则返回None，表示horizon为后面的所有 若没有执行state_update则返回None
        """
        E, A, T = self.num_envs, self.num_agents, self.episode_length
        mech_type = self.mechanism_agent_type

        # 对于 pricing: 只更新price signal，不执行state_update（已经在step中完成）
        if mech_type == "pricing":
            mech_effects = self.action_map_mechanism(mechanism_actions)
            price_action = mech_effects.get("price")
            if price_action is not None:
                price_arr = np.asarray(price_action, dtype=np.float32).reshape(self.num_envs)
                self.price_agent_signal = price_arr
            # 返回常规agent的obs（在下一次step的时候会用到更新后的price_agent_signal）
            actor_obs = self._compute_actor_obs(self.actor_obs_type)
            critic_obs = self._compute_critic_obs(self.critic_obs_type)
            avail_actions = self._get_avail_actions()
            # pricing模式下，step_mechanism不返回rewards等，因为已经在step中返回过了
            # 这里返回的是下一个时刻的obs（用于mechanism agent的训练）
            return [actor_obs, critic_obs, None, None, None, None, None, None]

        # 对于 cen_allocation 和 dec_allocation: 执行state_update
        if mech_type not in ("cen_allocation", "dec_allocation"):
            raise RuntimeError("step_mechanism only applies to allocation or pricing mechanisms.")

        if self._pending_mechanism_data is None:
            raise RuntimeError("Call step() before step_mechanism().")

        # 获取mechanism action的效果
        mech_effects = self.action_map_mechanism(mechanism_actions)
        order_today = self._pending_mechanism_data["order_today"]
        transship_intend = self._pending_mechanism_data["transship_intend"]
        t_index = self._pending_mechanism_data["t_index"]
        self._pending_mechanism_data = None

        # 如果mechanism修改了transship_intentions，使用修改后的值
        if "transship_intentions" in mech_effects:
            transship_intend = mech_effects["transship_intentions"]

        # 执行state_update
        self.order_today = order_today

        # Allocation via mechanism -> transship_matrix (E,A,A), payment (E,A)
        transship_matrix, payment = self._alloc_transshipment(
            transship_intend,
            mechanism_ratio=mech_effects.get("transship_ratio"),
            mechanism_matrix=mech_effects.get("transship_matrix"),
        )
        self.transship_matrix = transship_matrix
        self.transship_intend = transship_intend
        self.transship_actual = transship_matrix.sum(axis=2)

        # State update
        rewards, lt_today, lt_window, demand_t = self._state_update(
            order_today, transship_matrix, payment, t_index
        )

        self.step_index = self.step_index + 1

        # Build done and infos
        done_env = self.step_index >= T
        done_agents = np.repeat(done_env[:, None], A, axis=1)
        if self.train_tf:
            infos = [
                [{"bad_transition": True} if self.step_index[e] > T else {} for _ in range(A)]
                for e in range(E)
            ]
        else:
            infos = self._build_log_infos(lt_today, lt_window, demand_t)

        # Add mechanism_reward to infos (based on mechanism_reward_mode)
        if self.mechanism_reward_mode == "pooled_cost_saving":
            mechanism_reward =  self.pooled_cost_saving_avg  # (E,)
        else:  # default: avg_reward
            mechanism_reward = rewards.mean(axis=1)  # (E,)
        for e in range(E):
            for a in range(A):
                infos[e][a]["mechanism_reward"] = float(mechanism_reward[e])

        rewards_out = rewards[..., None]

        # Apply price signal if provided by mechanism
        price_action_next = mech_effects.get("price")
        if price_action_next is not None:
            price_arr = np.asarray(price_action_next, dtype=np.float32).reshape(E)
            self.price_agent_signal = price_arr

        # 返回常规agent的obs
        actor_obs = self._compute_actor_obs(self.actor_obs_type)
        critic_obs = self._compute_critic_obs(self.critic_obs_type)
        avail_actions = self._get_avail_actions()
        return [actor_obs, critic_obs, rewards_out, done_agents, infos, avail_actions, lt_today, lt_window]

    # ---------------------------- Internal: setup ----------------------------

    def _broadcast_param(self, value) -> np.ndarray:
        """
        Convert scalar/dict/list value into per-agent float32 array.

        Args:
            value: scalar, dict, list, or array

        Returns:
            np.ndarray: shape (A,) dtype float32
        """
        A = self.num_agents
        if isinstance(value, dict):
            # ordered by agent index
            arr = [value.get(f"agent{i}") for i in range(A)]
            return np.asarray(arr, dtype=np.float32)
        if isinstance(value, (list, tuple, np.ndarray)):
            val_arr = np.asarray(value, dtype=np.float32)
            if val_arr.shape == (A,):
                return val_arr.astype(np.float32)
            if val_arr.size == 1:
                return np.full((A,), float(val_arr.reshape(())), dtype=np.float32)
            # fallback: broadcast first element
            return np.full((A,), float(val_arr.flat[0]), dtype=np.float32)
        # scalar
        return np.full((A,), float(value), dtype=np.float32)

    def _build_actor_obs_spaces(self):
        A = self.num_agents
        dim_per_agent = self._calc_actor_obs_dim(
            self.actor_obs_type, self.mechanism_agent_type
        )
        return [spaces.Box(low=-np.inf, high=np.inf, shape=(dim_per_agent,), dtype=np.float32) for _ in range(A)]

    def _build_critic_obs_spaces(self):
        A = self.num_agents
        dim_per_agent = self._calc_critic_obs_dim(
            self.critic_obs_type, self.mechanism_agent_type
        )
        return [spaces.Box(low=-np.inf, high=np.inf, shape=(dim_per_agent,), dtype=np.float32) for _ in range(A)]

    def _build_action_spaces(self):
        # Replicate discrete space formulas with per-agent bounds, independent of batch
        spaces_list = []
        agents_control_transship = self.mechanism_agent_type != "cen_allocation"
        for _ in range(self.num_agents):
            if self.action_type == "multi_discrete":
                order_dim = (self.order_ub[_] / self.order_step_size + (2 if (self.MOQ[_] > 0) else 1))
                if self.transship_type == "pure_autonomy":
                    transship_dim = (self.transship_ub[_] - self.transship_lb[_]) / self.transship_step_size
                elif self.transship_type == "half_autonomy":
                    transship_dim = 2
                elif self.transship_type in ("no", "force_to_transship"):
                    transship_dim = 1
                else:
                    raise Exception("wrong transship_type")
                if not agents_control_transship:
                    transship_dim = 1
                spaces_list.append(spaces.MultiDiscrete([int(order_dim), int(transship_dim)]))
            elif self.action_type == "discrete":
                if self.MOQ[_] > 0:
                    order_dim = self.order_ub[_] / self.order_step_size + 2
                else:
                    order_dim = self.order_ub[_] / self.order_step_size + 1
                if self.transship_type == "pure_autonomy":
                    transship_dim = (self.transship_ub[_] - self.transship_lb[_]) / self.transship_step_size
                elif self.transship_type == "half_autonomy":
                    transship_dim = 2
                elif self.transship_type in ("no", "force_to_transship"):
                    transship_dim = 1
                else:
                    raise Exception("wrong transship_type")
                if not agents_control_transship:
                    transship_dim = 1
                spaces_list.append(spaces.Discrete(int(order_dim * transship_dim)))

            elif self.action_type in ("continuous", "heu_discrete"):
                spaces_list.append(
                    spaces.Box(
                        low=np.array([0.0, self.transship_lb[_]], dtype=np.float32),
                        high=np.array([self.order_ub[_], self.transship_ub[_]], dtype=np.float32),
                        shape=(2,),
                        dtype=np.float32,
                    )
                )
            else:
                raise Exception("wrong action_type")

        return spaces_list

    def _build_mechanism_action_space(self):
        if self.mechanism_agent_type == "pricing":
            return [
                spaces.Box(
                    low=np.array([-1.0], dtype=np.float32),
                    high=np.array([1.0], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32,
                )
            ]
        if self.mechanism_agent_type == "cen_allocation":
            # Net transshipment actual (positive receive, negative ship out)
            ub = np.maximum(self.transship_ub, 1.0)
            lb = -np.maximum(-self.transship_lb, 1.0)
            if self.num_agents == 2:
                return [
                    spaces.Box(
                        low=np.array([-1.0], dtype=np.float32),
                        high=np.array([1.0], dtype=np.float32),
                        shape=(1,),
                        dtype=np.float32,
                    )
                ]
            else:
                return [
                    spaces.Box(
                        low=lb.astype(np.float32),
                        high=ub.astype(np.float32),
                        shape=(self.num_agents,),
                        dtype=np.float32,
                    )
                ]
        if self.mechanism_agent_type == "dec_allocation":
            return [
                spaces.Box(
                    low=np.zeros((self.num_agents,), dtype=np.float32),
                    high=np.ones((self.num_agents,), dtype=np.float32),
                    shape=(self.num_agents,),
                    dtype=np.float32,
                )
            ]
        return None

    def _calc_actor_obs_dim(self, obs_type: str, mechanism_agent_type: str) -> int:
        """
        Calculate actor observation dimension per agent.

        Base features: inventory, demand_t, demand_est_t, demand_std_est_t, intransit_per_day (max_lt dims)
        Additional: lt_mean, lt_std, moq (3 dims)
        Optional: price (1) if include_price_signal_in_obs or pricing mechanism

        Returns:
            int: observation dimension
        """
        include_price = self.include_price_signal_in_obs or (
            mechanism_agent_type == "pricing" and not self.pricing_without_telling
        )
        # 4 base scalars + max_lt intransit + 3 (lt_mean, lt_std, moq) + optional price
        per_agent_full = 4 + self.max_lt + 3 + (1 if include_price else 0)

        if obs_type == "full":
            return per_agent_full * self.num_agents
        if obs_type == "self":
            return per_agent_full
        raise Exception("wrong actor_obs_type")

    def _calc_critic_obs_dim(self, obs_type: str, mechanism_agent_type: str) -> int:
        """
        Calculate critic observation dimension per agent.

        Critic obs = actor obs + demand statistics + inventory estimates

        Returns:
            int: critic observation dimension
        """
        # Critic augments actor obs with demand statistics and inventory estimations
        inv_hor = int(self.args["inventory_est_horizon"])  # horizon
        demand_info = self.args["demand_info_for_critic"]
        add_long_quant = 5 if ("long_quantile" in demand_info) else 0
        add_long_std = 1 if ("long_std" in demand_info) else 0
        add_long_mean = 1 if ("long_mean" in demand_info) else 0
        add_short_all = inv_hor if ("short_all" in demand_info) else 0
        add_short_mean = 1 if ("short_mean" in demand_info) else 0
        inv_for_critic = bool(self.args["inventory_est_for_critic_tf"])  # inventory estimation for self
        inv_others_for_critic = bool(self.args["inventory_est_others_for_critic_tf"])  # others
        add_inv_self = inv_hor if inv_for_critic else 0
        add_inv_others = (2 * inv_hor) if (inv_for_critic and inv_others_for_critic) else 0
        critic_full_per_agent = add_long_quant + add_long_std + add_long_mean + add_short_all + add_short_mean + add_inv_self + add_inv_others

        # Actor obs dimension - must use actual actor_obs_dim since it depends on actor_obs_type
        actor_obs_dim = self._calc_actor_obs_dim(
            self.actor_obs_type, mechanism_agent_type
        )

        if obs_type == "full":
            # All agents see global state: flatten all critic_cat + all actor_obs
            # critic_cat: (E, A, critic_full_per_agent) -> reshape to (E, A*critic_full_per_agent)
            # actor_obs: (E, A, actor_obs_dim) -> reshape to (E, A*actor_obs_dim)
            # Total: A * critic_full_per_agent + A * actor_obs_dim = A * (critic_full_per_agent + actor_obs_dim)
            return (critic_full_per_agent + actor_obs_dim) * self.num_agents
        if obs_type == "self":
            # Each agent sees: critic_full_per_agent + actual actor_obs
            return critic_full_per_agent + actor_obs_dim
        raise Exception("wrong critic_obs_type")

    def _calc_mechanism_obs_dim(self, obs_type: str) -> int:
        if obs_type == "intentions_only":
            return 2 * self.num_agents # normalized intentions + intentions
        if obs_type == "full":
            intentions_dim = 2 * self.num_agents if self.mechanism_agent_type == "dec_allocation" else 0
            actor_full_dim = self._calc_actor_obs_dim("full", self.mechanism_agent_type)
            return intentions_dim + actor_full_dim
        if obs_type == "compressed":
            # Compressed: mean + std of per-agent features (unnormalized)
            # Features per agent: actor_obs ("self") + intention (1 if dec_allocation else 0)
            actor_dim_per_agent = self._calc_actor_obs_dim("self", self.mechanism_agent_type)
            intentions_dim_per_agent = 1 if self.mechanism_agent_type == "dec_allocation" else 0
            total_per_agent = actor_dim_per_agent + intentions_dim_per_agent
            return 2 * total_per_agent + intentions_dim_per_agent * self.num_agents
        raise Exception("wrong mechanism obs type")

    def _calc_mechanism_critic_obs_dim(self, obs_type: str) -> int:
        # For mechanism critic, we use the same logic as _calc_mechanism_obs_dim
        # but for the critic observation type


        if obs_type == "full":
            intentions_dim = self.num_agents if self.mechanism_agent_type == "dec_allocation" else 0
            critic_full_dim = self._calc_critic_obs_dim("full", self.mechanism_agent_type)
            return intentions_dim + critic_full_dim
        if obs_type == "compressed":
            critic_dim_per_agent = self._calc_critic_obs_dim("self", self.mechanism_agent_type)
            intentions_dim_per_agent = 1 if self.mechanism_agent_type == "dec_allocation" else 0
            total_per_agent = critic_dim_per_agent + intentions_dim_per_agent
            return 2 * total_per_agent
        raise Exception("wrong mechanism critic obs type")

    def _compute_mechanism_actor_obs(self) -> np.ndarray:
        """
        Compute mechanism actor observations.

        Returns:
            np.ndarray: shape (E, 1, obs_dim)
        """
        E = self.num_envs
        obs_type = self.actor_obs_type_mechanism
        if obs_type == "intentions_only":
            # Only return transshipment intentions: (E, num_agents)
            intentions = self.transship_intend  # (E, num_agents)
            intentions_norm = intentions / np.maximum(self.inventory, 1)
            obs = np.concatenate([intentions_norm, intentions], axis=1)
            return obs[:, None, :]  # (E, 1, num_agents)

        if obs_type == "full":
            # Combine intentions (if applicable) + actor full obs
            parts = []


            # Actor full obs
            actor_full_obs = self._compute_actor_obs("full", for_mechanism=True)  # (E, actor_full_dim)
            parts.append(actor_full_obs)

            # Intentions part: only for dec_allocation mechanism
            if self.mechanism_agent_type == "dec_allocation":
                intentions = self.transship_intend  # (E, num_agents)
                intentions_norm = intentions / np.maximum(self.inventory, 1)
                parts.append(intentions_norm)
                parts.append(intentions)

            # Concatenate and reshape
            if parts:
                obs = np.concatenate(parts, axis=1)  # (E, total_dim)
            else:
                obs = actor_full_obs  # (E, actor_full_dim)
            return obs[:, None, :]  # (E, 1, total_dim)

        if obs_type == "compressed":
            # Unnormalized per-agent features
            # actor_obs: (E, A, dim)
            actor_obs = self._compute_actor_obs("self", normalize_env=True)

            # Combine with intentions if needed
            if self.mechanism_agent_type == "dec_allocation":
                intentions = self.transship_intend  # (E, A)
                intentions_norm = intentions / np.maximum(self.inventory, 1)
                # Concatenate along feature axis
                per_agent_features = np.concatenate([actor_obs, intentions_norm[..., None]], axis=2)
            else:
                per_agent_features = actor_obs

            # Compute mean and std across agents (axis 1)
            # (E, A, D) -> (E, D)
            mean_obs = per_agent_features.mean(axis=1)
            std_obs = np.log(1 + per_agent_features.std(axis=1))

            # Concatenate mean and std
            if self.mechanism_agent_type == "dec_allocation":
                obs = np.concatenate([mean_obs, std_obs, intentions], axis=1)  # (E, 2*D)
            else:
                obs = np.concatenate([mean_obs, std_obs], axis=1)  # (E, 2*D)
            return obs[:, None, :]  # (E, 1, 2*D)

        raise Exception("wrong mechanism obs type")

    def _compute_mechanism_critic_obs(self) -> np.ndarray:
        """
        Compute mechanism critic observations.

        Returns:
            np.ndarray: shape (E, 1, obs_dim)
        """
        E = self.num_envs
        obs_type = self.critic_obs_type_mechanism

        if obs_type == "full":
            parts = []

            # Decision critic obs
            decision_critic_obs = self._compute_critic_obs("full", for_mechanism=True)  # (E, decision_critic_obs_dim)
            parts.append(decision_critic_obs)

            # Intentions critic obs: only for dec_allocation mechanism
            if self.mechanism_agent_type == "dec_allocation":
                intentions = self.transship_intend  # (E, num_agents)
                intentions_norm = intentions / np.maximum(self.inventory, 1)
                parts.append(intentions_norm)

            # Concatenate and reshape
            if len(parts) > 1:
                obs = np.concatenate(parts, axis=1)  # (E, total_dim)
            else:
                obs = decision_critic_obs  # (E, decision_critic_obs_dim)
            return obs[:, None, :]  # (E, 1, total_dim)

        if obs_type == "compressed":
            # Unnormalized per-agent features
            # critic_obs: (E, A, dim)
            critic_obs = self._compute_critic_obs("self", normalize_env=True)

            # Combine with intentions if needed
            if self.mechanism_agent_type == "dec_allocation":
                intentions = self.transship_intend  # (E, A)
                intentions_norm = intentions / np.maximum(self.inventory, 1)
                per_agent_features = np.concatenate([critic_obs, intentions_norm[..., None]], axis=2)
            else:
                per_agent_features = critic_obs

            # Compute mean and std across agents
            mean_obs = per_agent_features.mean(axis=1)
            std_obs = np.log(1 + per_agent_features.std(axis=1))

            obs = np.concatenate([mean_obs, std_obs], axis=1)
            return obs[:, None, :]

        raise Exception("wrong mechanism critic obs type")
    # ---------------------------- Internal: runtime state ----------------------------
    def _allocate_runtime_state(self):
        """
        Allocate all runtime state tensors.

        Shapes:
            step_index: (E,)
            inventory: (E, A)
            order_today: (E, A)
            transship_intend: (E, A)
            transship_actual: (E, A)
            transship_matrix: (E, A, A)
            delivery_schedule: (E, A, T)
            intransit_hist: (E, A, L)
            reward: (E, A)
            reward_cum: (E, A)
            demand: (E, A, T)
            LT: (E, A, T) - total lead time (simplified, no stages)
            shipping_loss: (E, A, T)
            demand_est: (E, A, T)
            demand_std_est: (E, A, T)
        """
        E, A = self.num_envs, self.num_agents
        L = self.max_lt
        T = self.episode_length

        self.step_index = np.zeros((E,), dtype=np.int32)
        self.done_mask = np.zeros((E,), dtype=bool)

        # State tensors
        self.inventory = np.zeros((E, A), dtype=np.float32)
        self.order_today = np.zeros((E, A), dtype=np.float32)
        self.transship_intend = np.zeros((E, A), dtype=np.float32)
        self.transship_actual = np.zeros((E, A), dtype=np.float32)
        self.transship_matrix = np.zeros((E, A, A), dtype=np.float32)

        # Delivery schedule for orders: arrival quantities per time index
        self.delivery_schedule = np.zeros((E, A, T), dtype=np.float32)

        # In-transit history: track orders for the past L days
        self.intransit_hist = np.zeros((E, A, L), dtype=np.float32)

        # Accumulators
        self.reward = np.zeros((E, A), dtype=np.float32)
        self.reward_cum = np.zeros((E, A), dtype=np.float32)
        self.reward_selfish = np.zeros((E, A), dtype=np.float32)
        self.reward_selfish_cum = np.zeros((E, A), dtype=np.float32)
        self.inventory_before_arrive = np.zeros((E, A), dtype=np.float32)
        self.shortage = np.zeros((E, A), dtype=np.float32)
        self.demand_fulfilled_all = np.zeros((E, A), dtype=np.float32)
        self.demand_fulfilled_intime = np.zeros((E, A), dtype=np.float32)
        self.demand_fulfilled_late = np.zeros((E, A), dtype=np.float32)
        self.ordering_times = np.zeros((E, A), dtype=np.float32)
        self.transship_instantly_used = np.zeros((E, A), dtype=np.float32)
        self.shipping_cost_pure = np.zeros((E, A), dtype=np.float32)
        self.shipping_cost_all = np.zeros((E, A), dtype=np.float32)
        self.ordering_cost = np.zeros((E, A), dtype=np.float32)
        self.penalty_cost = np.zeros((E, A), dtype=np.float32)
        self.holding_cost = np.zeros((E, A), dtype=np.float32)
        self.selling_revenue = np.zeros((E, A), dtype=np.float32)

        # Precomputed sequences (filled in _precompute_sequences)
        self.demand = np.zeros((E, A, T), dtype=np.float32)
        self.LT = np.zeros((E, A, T), dtype=np.float32)  # Simplified: single total LT
        self.shipping_loss = np.zeros((E, A, T), dtype=np.float32)

        self.demand_est = np.zeros((E, A, T + 1))
        self.demand_std_est = np.ones((E, A, T + 1))

        # Pooled cost saving for mechanism_reward
        self.pooled_cost_saving_avg = np.zeros((E,), dtype=np.float32)

    def _reset_runtime_state(self):
        E, A, T = self.num_envs, self.num_agents, self.episode_length
        self.step_index[...] = 0
        self.done_mask[...] = False

        # Reset inventory to demand_mean_const
        self.inventory[...] = self.demand_mean_const.reshape(1, A)
        self.order_today.fill(0.0)
        self.transship_intend.fill(0.0)
        self.transship_actual.fill(0.0)
        self.transship_matrix.fill(0.0)
        # Set intransit_hist to demand_mean_const for all lead time days
        self.intransit_hist[...] = np.broadcast_to(self.demand_mean_const.reshape(1, A, 1), (E, A, self.max_lt))
        self.intransit_hist[:, :, :self.max_lt // 2] = 0

        # Set delivery_schedule based on intransit_hist and LT
        # intransit_hist[:, :, i] represents orders placed at t=-(i+1)
        # These orders will arrive at t = -(i+1) + LT, which is LT-i-1 days from now
        self.delivery_schedule.fill(0.0)
        # Use LT at t=0 as reference for past orders
        LT_ref = self.LT[:, :, 0]  # (E, A)
        for i in range(self.max_lt // 2, self.max_lt):
            t_before = i - self.max_lt
            arrival_day = np.clip(LT_ref.astype(np.int32) + t_before, 0, T - 1)  # (E, A)
            # Vectorized update using advanced indexing
            env_indices = np.arange(E).reshape(E, 1)  # (E, 1)
            agent_indices = np.arange(A).reshape(1, A)  # (1, A)
            self.delivery_schedule[env_indices, agent_indices, arrival_day] += self.intransit_hist[:, :, i]

        # Zero accumulators
        self.reward.fill(0.0)
        self.reward_cum.fill(0.0)
        self.reward_selfish.fill(0.0)
        self.reward_selfish_cum.fill(0.0)
        self.inventory_before_arrive.fill(0.0)
        self.shortage.fill(0.0)
        self.demand_fulfilled_all.fill(0.0)
        self.demand_fulfilled_intime.fill(0.0)
        self.demand_fulfilled_late.fill(0.0)
        self.ordering_times.fill(0.0)
        self.transship_instantly_used.fill(0.0)
        self.shipping_cost_pure.fill(0.0)
        self.shipping_cost_all.fill(0.0)
        self.ordering_cost.fill(0.0)
        self.penalty_cost.fill(0.0)
        self.holding_cost.fill(0.0)
        self.selling_revenue.fill(0.0)
        self.pooled_cost_saving_avg.fill(0.0)
        self.price_agent_signal = None
        self._pending_mechanism_data = None

    # ---------------------------- Internal: observations ----------------------------
    def _compute_actor_obs(self, actor_obs_type: str, for_mechanism: bool = False, normalize_env: bool = None) -> np.ndarray:
        """
        Compute actor observations for all envs and agents.

        Args:
            for_mechanism: If True, returns (E, actor_obs_dim) instead of (E, A, actor_obs_dim)
            normalize_env: If provided, overrides self.state_normalize

        Returns:
            np.ndarray: shape (E, A, actor_obs_dim) or (E, actor_obs_dim) if for_mechanism=True
        """
        E, A, L = self.num_envs, self.num_agents, self.max_lt
        # Basic features per agent (inventory, demand_t, demand_est_t, demand_std_est_t, intransit_per_day, lt_mean, lt_std, moq)
        t = self.step_index.clip(min=0, max=self.episode_length - 1)  # (E,)
        t_idx = t.reshape(E, 1, 1)

        demand_est_t = np.take_along_axis(self.demand_est, t_idx, axis=2).reshape(E, A)  # (E, A)
        demand_std_est_t = np.take_along_axis(self.demand_std_est, t_idx, axis=2).reshape(E, A)  # (E, A)
        demand_t = np.take_along_axis(self.demand, t_idx - 1, axis=2).reshape(E, A) if t_idx[0] > 0 else demand_est_t

        # Use intransit_hist: detailed intransit for each day in the past L days
        intransit_per_day = self.intransit_hist  # (E, A, L)

        # Use precomputed lt_mean_const and lt_std_const (shape (A,)), broadcast to (E, A)
        lt_mean = np.broadcast_to(self.lt_mean_const.reshape(1, A), (E, A)).astype(np.float32)  # (E, A)
        lt_std = np.broadcast_to(self.lt_std_const.reshape(1, A), (E, A)).astype(np.float32)  # (E, A)
        moq = np.broadcast_to(self.MOQ.reshape(1, A), (E, A)).astype(np.float32)  # (E, A)

        normalize = self.state_normalize if normalize_env is None else normalize_env
        if normalize:
            # Normalize using stored constants from demand generator
            mean_const = np.maximum(self.demand_mean_const, 1e-6)
            std_const = np.maximum(self.demand_std_const, 1e-6)

            inventory_norm = self.inventory / mean_const - 1.0
            demand_norm = demand_t / mean_const - 1.0
            demand_est_norm = demand_est_t / mean_const - 1.0
            demand_std_est_norm = np.log(demand_std_est_t / std_const + 1e-6)
            # Normalize intransit_per_day: (E, A, L)
            intransit_per_day_norm = intransit_per_day / np.broadcast_to(mean_const.reshape(1, A, 1), (E, A, L)) - 1.0

            # Normalize lt_mean and lt_std using max_lt (std uses sqrt(max_lt))
            lt_mean_norm = lt_mean / self.max_lt - 1.0
            lt_std_norm = np.log(lt_std / np.sqrt(self.max_lt) + 1e-6)
            # Normalize moq using demand mean_const
            moq_norm = moq / mean_const - 1.0

            lb, ub = self.state_clip_list
            # Stack scalar features and then concatenate with intransit time series
            norm_scalars = np.stack([inventory_norm, demand_norm, demand_est_norm, demand_std_est_norm], axis=2)  # (E, A, 4)
            intransit_per_day_norm = np.clip(intransit_per_day_norm, lb, ub)
            norm_scalars = np.clip(norm_scalars, lb, ub)
            # Add lt_mean, lt_std, moq features
            lt_moq_features = np.stack([lt_mean_norm, lt_std_norm, moq_norm], axis=2)  # (E, A, 3)
            lt_moq_features = np.clip(lt_moq_features, lb, ub)
            full_arr = np.concatenate([norm_scalars, intransit_per_day_norm, lt_moq_features], axis=2)  # (E, A, 4+L+3)
        else:
            per_agent_full = [
                self.inventory,  # (E, A)
                demand_t,  # (E, A)
                demand_est_t,  # (E, A)
                demand_std_est_t,  # (E, A)
                intransit_per_day,  # (E, A, L)
                lt_mean,  # (E, A)
                lt_std,  # (E, A)
                moq,  # (E, A)
            ]

            out = []
            for x in per_agent_full:
                if x.ndim == 2:
                    out.append(x[..., None])
                else:
                    out.append(x)
            full_arr = np.concatenate(out, axis=2)  # (E,A,features)


        if self._include_price_feature:
            if self.price_agent_signal is None:
                price_feature = np.zeros((E, A, 1), dtype=np.float32)
            else:
                price_feature = np.broadcast_to((self.price_agent_signal - 1).reshape(E, 1, 1), (E, A, 1)) / (self.B + self.P).reshape(1, A, 1)
                # state_clip
                price_feature = np.clip(price_feature, self.state_clip_list[0], self.state_clip_list[1])
            full_arr = np.concatenate([full_arr, price_feature], axis=2)

        if actor_obs_type == "full":
            if for_mechanism:
                # For mechanism: flatten all agents' observations into a single global observation
                # Shape: (E, 1, A * per_agent_features)
                global_obs = full_arr.reshape(E, -1)  # (E, A * per_agent_features)
                return global_obs
            else:
                # Put self first per agent, then others
                per_agent_concat = []
                for a in range(self.num_agents):
                    self_first = full_arr[:, a : a + 1, :]
                    others = np.concatenate([full_arr[:, :a, :], full_arr[:, a + 1 :, :]], axis=1)
                    per_agent_concat.append(np.concatenate([self_first, others], axis=1).reshape(E, -1))
                return np.stack(per_agent_concat, axis=1)

        if actor_obs_type == "self":
            return full_arr
        raise Exception("wrong actor_obs_type")

    def _compute_critic_obs(self, critic_obs_type: str, for_mechanism: bool = False, normalize_env: bool = None) -> np.ndarray:
        """
        Compute critic observations for all envs and agents.

        Args:
            for_mechanism: If True, returns (E, critic_obs_dim) instead of (E, A, critic_obs_dim)
            normalize_env: If provided, overrides self.state_normalize

        Returns:
            np.ndarray: shape (E, A, critic_obs_dim) or (E, critic_obs_dim) if for_mechanism=True
        """
        E, A, T = self.num_envs, self.num_agents, self.episode_length
        inv_h = int(self.args["inventory_est_horizon"])  # horizon

        # Actor obs for critic concatenation basis (per-agent)
        actor_obs = self._compute_actor_obs(self.actor_obs_type, for_mechanism=False, normalize_env=normalize_env)

        # Time indices
        t = np.minimum(self.step_index, T - 1)
        t_idx = t.reshape(E, 1)

        # Use precomputed demand statistics (much faster than computing on-the-fly)
        # Index precomputed arrays at current timestep using advanced indexing
        demand_info = self.args["demand_info_for_critic"]

        # Efficient gather using fancy indexing: precomputed[env_idx, :, t[env_idx]]
        # Create index arrays for gathering
        env_indices = np.arange(E)  # [0, 1, 2, ..., E-1]

        # Gather statistics for current timestep from precomputed arrays
        long_mean_arr = []
        long_std_arr = []
        long_quantile_arr = []
        short_mean_arr = []
        short_all_arr = []

        if "long_mean" in demand_info and self.precomputed_long_mean is not None:
            # precomputed_long_mean is (E, A, T), gather at t[e] for each env e
            # Result: (E, A)
            long_mean_t = self.precomputed_long_mean[env_indices, :, t]
            long_mean_arr = [long_mean_t]

        if "long_std" in demand_info and self.precomputed_long_std is not None:
            # precomputed_long_std is (E, A, T), gather at t[e] for each env e
            # Result: (E, A)
            long_std_t = self.precomputed_long_std[env_indices, :, t]
            long_std_arr = [long_std_t]

        if "long_quantile" in demand_info and self.precomputed_long_quantile is not None:
            # precomputed_long_quantile is (E, A, T, 5), gather at t[e] for each env e
            # Result: (E, A, 5)
            long_quantile_t = self.precomputed_long_quantile[env_indices, :, t, :]
            # Split into 5 separate arrays for compatibility
            for q_idx in range(5):
                long_quantile_arr.append(long_quantile_t[:, :, q_idx])

        if "short_mean" in demand_info and self.precomputed_short_mean is not None:
            # Result: (E, A)
            short_mean_t = self.precomputed_short_mean[env_indices, :, t]
            short_mean_arr = [short_mean_t]

        if "short_all" in demand_info and self.precomputed_short_all is not None:
            # precomputed_short_all is (E, A, T, inv_h), gather at t[e] for each env e
            # Result: (E, A, inv_h)
            short_all_t = self.precomputed_short_all[env_indices, :, t, :]
            short_all_arr = [short_all_t]

        # Inventory estimation for next inv_h days using delivery_schedule and demand (loop over horizon only)
        inv_est = np.zeros((E, A, inv_h), dtype=np.float32)
        inv_prev = np.copy(self.inventory)
        for h in range(inv_h):
            t_h = np.minimum(t + h, T - 1).reshape(E, 1, 1)
            arr_h = np.take_along_axis(self.delivery_schedule, t_h, axis=2).reshape(E, A)
            dem_h = np.take_along_axis(self.demand, np.minimum(t + h, T - 1).reshape(E, 1, 1), axis=2).reshape(E, A)
            inv_est[:, :, h] = inv_prev - dem_h
            inv_base = inv_est[:, :, h] if self.backlog_tf else np.maximum(inv_est[:, :, h], 0.0)
            inv_prev = inv_base + arr_h
        est_inv_arr = inv_est

        # Others' shortage/excess averages
        if self.args["inventory_est_for_critic_tf"] and self.args["inventory_est_others_for_critic_tf"]:
            shortage = np.maximum(-est_inv_arr, 0.0)
            excess = np.maximum(est_inv_arr, 0.0)
            tot_short = shortage.sum(axis=1, keepdims=True)
            tot_excess = excess.sum(axis=1, keepdims=True)
            # average of others for each agent: (sum_all - self) / (A-1)
            est_shortage_others = (tot_short - shortage) / max(A - 1, 1)
            est_excess_others = (tot_excess - excess) / max(A - 1, 1)
        else:
            est_shortage_others = np.zeros_like(est_inv_arr)
            est_excess_others = np.zeros_like(est_inv_arr)

        # Normalization
        normalize = self.state_normalize if normalize_env is None else normalize_env
        if normalize:
            # Use stored constants for normalization
            mean_const = np.maximum(self.demand_mean_const, 1e-6)
            # Reshape mean_const for proper broadcasting: (A,) -> (1,A,1)
            mean_const_broadcast = mean_const.reshape(1, -1, 1)

            # Normalize long/short and inv predictions
            def _norm_clip(arr):
                out = arr / mean_const_broadcast - 1.0
                lb, ub = self.state_clip_list
                return np.clip(out, lb, ub)

            # mean_const is (A,), need (1,A) for 2D arrays
            mean_const_2d = mean_const.reshape(1, -1)
            std_const_2d = np.maximum(self.demand_std_const, 1e-6).reshape(1, -1)
            long_mean_arr = [np.clip(long_mean_arr[0] / mean_const_2d - 1.0, self.state_clip_list[0], self.state_clip_list[1])] if long_mean_arr else []
            if long_std_arr:
                long_std_arr = [np.log(long_std_arr[0] / std_const_2d + 1e-6)]
            if long_quantile_arr:
                long_quantile_arr = [np.clip(x / mean_const_2d - 1.0, self.state_clip_list[0], self.state_clip_list[1]) for x in long_quantile_arr]
            short_mean_arr = [np.clip(short_mean_arr[0] / mean_const_2d - 1.0, self.state_clip_list[0], self.state_clip_list[1])] if short_mean_arr else []
            short_all_arr = [ _norm_clip(short_all_arr[0]) ] if short_all_arr else []
            est_inv_arr = _norm_clip(est_inv_arr)
            # For others' estimates, scale by assumed rates similar to legacy
            fillrate_assumed = 0.9
            est_shortage_others = np.clip(est_shortage_others / ((1 - fillrate_assumed) * mean_const_broadcast) - 1.0,
                                          self.state_clip_list[0], self.state_clip_list[1])
            est_excess_others = np.clip(est_excess_others / mean_const_broadcast - 1.0,
                                        self.state_clip_list[0], self.state_clip_list[1])

        # Assemble per-agent critic features
        long_stack = long_mean_arr + long_std_arr + long_quantile_arr
        short_stack = short_mean_arr + (short_all_arr if short_all_arr else [])
        critic_full = []
        if long_stack:
            critic_full.append(np.stack(long_stack, axis=2))  # (E,A,k)
        if short_stack:
            if len(short_stack) == 1:
                critic_full.append(short_stack[0])  # (E,A) or (E,A,H)
            else:
                # short_mean + short_all -> concat along feature axis
                short_cat = [x if x.ndim == 3 else x[:, :, None] for x in short_stack]
                critic_full.append(np.concatenate(short_cat, axis=2))
        if self.args["inventory_est_for_critic_tf"]:
            critic_full.append(est_inv_arr)
        # others
        if self.args["inventory_est_others_for_critic_tf"]:
            critic_full.append(est_shortage_others)
            critic_full.append(est_excess_others)

        # Ensure all parts are 3D (E,A,feat)
        critic_feats = []
        for comp in critic_full:
            if comp.ndim == 2:
                critic_feats.append(comp[:, :, None])
            else:
                critic_feats.append(comp)
        critic_cat = np.concatenate(critic_feats, axis=2) if critic_feats else np.zeros((E, A, 0), dtype=np.float32)

        demand_info = self.args["demand_info_for_critic"]
        # Combine with actor obs per critic_obs_type
        if critic_obs_type == "full":
            # All agents see the same global state: all critic_cat + all actor_obs
            # Shape: (E, A, total_features) where total_features = A * (critic_features + actor_features)
            global_obs = np.concatenate([critic_cat.reshape(E, -1), actor_obs.reshape(E, -1)], axis=1)  # (E, total_features)
            if for_mechanism:
                return global_obs
            # Broadcast to all agents: each agent gets the same global observation
            return np.broadcast_to(global_obs[:, None, :], (E, A, global_obs.shape[1]))
        if critic_obs_type == "self":
            return np.concatenate([critic_cat, actor_obs], axis=2)
        raise Exception("wrong critic_obs_type")

    def _get_avail_actions(self):
        return None

        #     n = self.action_space[0].n





        # raise Exception("wrong action_type")

    # ---------------------------- Internal: action -> order/transship_intend ----------------------------
    def _action_map(self, actions: np.ndarray, t_index: np.ndarray):
        E, A = self.num_envs, self.num_agents
        # demand estimate at t
        t_idx = t_index.reshape(E, 1, 1)
        demand_est_t = np.take_along_axis(self.demand_est, t_idx, axis=2).reshape(E, A)

        if self.action_type == "discrete":
            act = actions.astype(np.int64)
            moq_flag = ((self.MOQ > 0)).astype(np.int64)
            order_dim = (self.order_ub / self.order_step_size + 2 * moq_flag + (1 - moq_flag)).astype(np.int64)
            order_index = np.mod(act, order_dim.reshape(1, A))
            transship_index = np.floor_divide(act, order_dim.reshape(1, A))
            order_ratio = np.where(order_index > 0,
                                   (order_index - 1) * self.order_step_size + 1.0,
                                   0.0)
            # transship ratio per agent
            transship_ratio = transship_index * self.transship_step_size + self.transship_lb.reshape(1, A)
        elif self.action_type == "multi_discrete":
            order_index = actions[..., 0].astype(np.int64)
            transship_index = actions[..., 1].astype(np.int64)
            moq_flag = ((self.MOQ > 0)).astype(np.int64)
            order_ratio = np.where(order_index > 0,
                                   (order_index - 1) * self.order_step_size + 1.0,
                                   0.0) if moq_flag.any() else (order_index * self.order_step_size)
            # broadcast lb by agent
            transship_ratio = transship_index * self.transship_step_size + self.transship_lb.reshape(1, A)
        elif self.action_type in ("continuous", "heu_discrete"):
            order_ratio = actions[..., 0].astype(np.float32)
            transship_ratio = actions[..., 1].astype(np.float32)
        else:
            raise Exception("wrong action_type")

        # Order mapping
        if self.action_type not in ("continuous", "heu_discrete"):
            order_amounts = np.round(np.maximum(order_ratio - 1.0, 0.0) * demand_est_t
                                    + np.where(order_ratio > 0.0, self.MOQ, 0.0)).astype(np.float32)
        else:
            order_amounts = actions[..., 0]

        # Transship intention mapping by type
        if self.transship_type == "pure_autonomy":
            # positive ratio scaled by demand_est, negative ratio scaled by available inventory
            pos = np.maximum(transship_ratio, 0.0)
            neg = np.minimum(transship_ratio, 0.0)
            intentions = np.round(pos * demand_est_t + neg * np.maximum(self.inventory - demand_est_t, 0.0)).astype(np.float32)
        elif self.transship_type == "half_autonomy":
            intentions = (transship_index > 0).astype(np.float32)
        elif self.transship_type == "no":
            intentions = np.zeros((E, A), dtype=np.float32)
        elif self.transship_type == "force_to_transship":
            intentions = np.ones((E, A), dtype=np.float32)
        else:
            raise Exception("wrong transship_type")

        return order_amounts, intentions

    def action_map_mechanism(self, mechanism_actions: np.ndarray):
        """
        Map mechanism agent actions into normalized dictionaries.
        Returns dict with optional keys: 'price', 'transship_intentions',
        'transship_ratio', 'transship_matrix'.
        """
        E, A = self.num_envs, self.num_agents
        mech_type = self.mechanism_agent_type
        # 先squeeze
        mechanism_actions = mechanism_actions.squeeze(1) if len(mechanism_actions.shape) == 3 else mechanism_actions
        if mechanism_actions is None:
            return {}
        if isinstance(mechanism_actions, dict):
            return mechanism_actions
        if mech_type == "none":
            return {}

        if mech_type == "pricing": # (E,)
            price = (mechanism_actions + 1.0) / 2.0 * np.average(self.B + self.H)
            return {"price": price}

        if mech_type == "cen_allocation":
            intentions = mechanism_actions.astype(np.float32)
            t_idx = self.step_index.reshape(E, 1, 1)
            demand_est_t = np.take_along_axis(self.demand_est, t_idx, axis=2).reshape(E, A)
            if self.num_agents == 2:
                # 如果intentions为正表示agent1向agent0转运，为负表示agent0向agent1转运. base为被转运者的inv-demand_est
                base = demand_est_t  # (E, A)
                pos = np.maximum(intentions, 0.0) # (E, 1)
                neg = np.minimum(intentions, 0.0) # (E, 1)
                intentions_1 = np.round(pos * base[:,[0]] + neg * base[:,[1]]).astype(np.float32) # (E, A)
                intentions_2 = np.round(-pos * base[:,[0]] - neg * base[:,[1]]).astype(np.float32) # (E, A)
                intentions = np.concatenate([intentions_1, intentions_2], axis=1) # (E, A)
                return {
                    "transship_intentions": intentions,
                }
            else:
                # Positive value: demand estimation, negative: available inventory
                pos = np.maximum(intentions, 0.0)
                neg = np.minimum(intentions, 0.0)
                intentions = np.round(pos * demand_est_t + neg * np.maximum(self.inventory - demand_est_t, 0.0)).astype(np.float32)

                return {
                    "transship_intentions": intentions,
                }

        if mech_type == "dec_allocation":
            if mechanism_actions.shape[1] != self.num_agents:
                raise ValueError(
                    f"mechanism_action dim mismatch: expected {self.num_agents}, got {mechanism_actions.shape[1]}"
                )
            return {"transship_ratio": mechanism_actions.astype(np.float32)}

        raise ValueError(f"Unsupported mechanism agent type {mech_type}")

    # ---------------------------- Internal: allocation ----------------------------
    def _get_price_array_for_payment(self):
        """
        专门给dist_mechanism让他们调用的，不能删 用于计算payment
        Get the price array for payment calculation.
        Returns None if revenue_allocation_method is not 'constant',
        otherwise returns the price_agent_signal.
        """
        if self.revenue_allocation_method != "constant":
            return None
        # price_agent_signal shape: (E,), expand to (E, A) for broadcasting
        if self.price_agent_signal is None:
            # Fallback to constant price if price_agent_signal is None
            return np.ones((self.num_envs, self.num_agents), dtype=np.float32) * self.constant_transshipment_price
        return np.broadcast_to(self.price_agent_signal.reshape(self.num_envs, 1), (self.num_envs, self.num_agents))

    def _alloc_transshipment(self, intentions: np.ndarray, mechanism_ratio=None, mechanism_matrix=None):
        """
        Allocate transshipment flows using the configured distribution mechanism.
        """
        E, A = self.num_envs, self.num_agents
        if mechanism_matrix is not None:
            flows = mechanism_matrix.astype(np.float32)
            payment = self.dist_mechanism.get_vectorized_payment(flows)
            return flows, payment

        flows = np.zeros((E, A, A), dtype=np.float32)
        payment = np.zeros((E, A), dtype=np.float32)
        if self.product_allocation_method == "mechanism_agent":  # only dec_allocation
            price_arr = self._get_price_array_for_payment()
            return self._alloc_with_agent_mechanism(
                intentions, mechanism_ratio, price_arr
            )

        # Vectorized implementation for least_distance
        if self.product_allocation_method == "least_distance":
            flows, payment = self.dist_mechanism.get_vectorized_mechanism_result(intentions)
            return flows, payment

        # Fallback: loop over envs for other mechanisms
        for env_idx in range(E):
            intents = intentions[env_idx].astype(np.float32).tolist()
            kwargs = {}
            price_arr = self._get_price_array_for_payment()[env_idx]
            flows[env_idx], payment[env_idx] = self.dist_mechanism.get_mechanism_result(
                intents, price_arr=price_arr, env_idx=env_idx
            )

        return flows, payment

    def _alloc_with_agent_mechanism(self, intentions, mechanism_ratio, price_arr):
        E, A = self.num_envs, self.num_agents
        flows = np.zeros((E, A, A), dtype=np.float32)
        payment = np.zeros((E, A), dtype=np.float32)
        if mechanism_ratio is not None:
            ratio = np.clip(mechanism_ratio.astype(np.float32), 0.0, None)
            ratio_sum = ratio.sum(axis=1, keepdims=True)
            invalid_mask = ratio_sum <= 0
            ratio_sum = np.where(invalid_mask, 1.0, ratio_sum)
            ratio = ratio / ratio_sum
            if np.any(invalid_mask):
                ratio[invalid_mask[:, 0]] = 1.0 / max(A, 1)
        else:
            ratio = np.full((E, A), 1.0 / max(A, 1), dtype=np.float32)
        for env_idx in range(E):
            intents = intentions[env_idx].astype(np.float32).tolist()
            kwargs = {}
            if ratio is not None:
                kwargs["transship_fulfill_ratio"] = ratio[env_idx]
            if price_arr is not None:
                kwargs["price_arr"] = price_arr[env_idx]
            mech_matrix, mech_payment = self.dist_mechanism.get_mechanism_result(
                intents, **kwargs
            )
            flows[env_idx] = np.asarray(mech_matrix, dtype=np.float32)
            payment[env_idx] = np.asarray(mech_payment, dtype=np.float32)
        return flows, payment

    def _build_log_infos(
        self, lt_today: np.ndarray, lt_window: np.ndarray, demand_t: np.ndarray
    ):
        infos = []
        for env_idx in range(self.num_envs):
            env_info = []
            for agent_idx in range(self.num_agents):
                info = {
                    "inventory": float(self.inventory[env_idx, agent_idx]),
                    "inventory_before_arrive": float(
                        self.inventory_before_arrive[env_idx, agent_idx]
                    ),
                    "demand": float(demand_t[env_idx, agent_idx]),
                    "order": float(self.order_today[env_idx, agent_idx]),
                    "transship": float(self.transship_actual[env_idx, agent_idx]),
                    "transship_intend": float(self.transship_intend[env_idx, agent_idx]),
                    "shortage": float(self.shortage[env_idx, agent_idx]),
                    "reward": float(self.reward[env_idx, agent_idx]),
                    "reward_selfish": float(self.reward_selfish[env_idx, agent_idx]),
                    "tsp_price": float(self.price_agent_signal[env_idx] if self.price_agent_signal is not None else self.constant_transshipment_price),
                    "tsp_revenue_allocation": float(self.shipping_cost_all[env_idx, agent_idx])

                }
                env_info.append(info)
            infos.append(env_info)
        return infos

    def _compute_lt_window(self, lt_today: np.ndarray) -> np.ndarray:
        """用于决定transshipment的policy loss的覆盖horizon的长度，如果没有设置则返回None，表示horizon为后面的所有
        Args:
            lt_today: shape (E, A)
        Returns:
            lt_window: shape (E, A)
        """
        mode = self.advantage_split_mode
        if mode == "lt_today":
            lt_window = np.rint(lt_today).astype(np.int32)
        elif mode == "lt_avg":
            lt_window = np.broadcast_to(
                self.lt_avg_agents.reshape(1, self.num_agents), lt_today.shape
            ).astype(np.int32)
        elif mode == "fixed":
            lt_window = np.full_like(
                lt_today, int(self.advantage_fixed_lt), dtype=np.int32
            )
        else:
            return None
        return lt_window

    # ---------------------------- Internal: state update and rewards ----------------------------
    def _state_update(self, order_today: np.ndarray, flows: np.ndarray, payment: np.ndarray, t_index: np.ndarray):
        E, A = self.num_envs, self.num_agents
        t_idx = t_index.reshape(E, 1, 1)
        demand_t = np.take_along_axis(self.demand, t_idx, axis=2).reshape(E, A)
        ship_loss_t = np.take_along_axis(self.shipping_loss, t_idx, axis=2).reshape(E, A)



        #     intentions = np.maximum(intentions, -np.maximum(self.inventory, 0.0))

        #     intentions = np.maximum(intentions, -np.maximum(self.inventory - demand_t, 0.0))

        trans_actual = flows.sum(axis=2)  # net inflow per agent
        # Schedule arrivals from today's orders
        # LT today (simplified: single total LT)
        LT_today = np.take_along_axis(self.LT, t_idx, axis=2).squeeze(2)  # (E, A)
        lt_window = self._compute_lt_window(LT_today)
        arrival_day = np.take_along_axis(self.order_arrival_time, t_idx, axis=2).squeeze(2)  # (E, A)
        valid_arrival = (arrival_day < self.episode_length) & (order_today > 0)
        # shipping loss at arrival day, to integer
        arr_idx = arrival_day.clip(min=0, max=self.episode_length - 1).astype(np.int32).reshape(E, A, 1)
        ship_loss_arrival = np.take_along_axis(self.shipping_loss, arr_idx, axis=2).reshape(E, A)
        arrival_qty_sched = order_today * (1.0 - ship_loss_arrival)
        # scatter-add into schedule
        env_ids = np.broadcast_to(np.arange(E)[:, None], (E, A))[valid_arrival]
        agent_ids = np.broadcast_to(np.arange(A)[None, :], (E, A))[valid_arrival]
        arrival_days_valid = arrival_day[valid_arrival]
        qty_valid = arrival_qty_sched[valid_arrival]
        np.add.at(self.delivery_schedule, (env_ids, agent_ids, arrival_days_valid), qty_valid)
        # arrivals today
        arrival_num = np.take_along_axis(self.delivery_schedule, t_idx, axis=2).reshape(E, A)
        # consume today's schedule
        t_b = np.broadcast_to(t_index.reshape(E, 1), (E, A))

        # zero out consumed arrivals
        all_envs = np.broadcast_to(np.arange(E)[:, None], (E, A))
        all_agents = np.broadcast_to(np.arange(A)[None, :], (E, A))
        self.delivery_schedule[all_envs, all_agents, t_b] = 0.0

        # 把今天到货的intransit_hist置为0
        t_range = np.arange(t_index[0] - self.max_lt, t_index[0])
        arrival_idx = self.order_arrival_time[:, :, np.maximum(t_range, 0)] + np.minimum(t_range, 0)
        arrival_mask = (arrival_idx <= t_index[0])
        self.intransit_hist[arrival_mask] = 0.0
        self.intransit_hist = np.concatenate([self.intransit_hist[:, :, 1:], order_today.reshape(E, A, 1)], axis=2)

        # Reactive: apply transship then demand
        if self.reactive_tf:
            inv_after_transship = self.inventory + trans_actual
            demand_intime = np.maximum(np.minimum(inv_after_transship, demand_t), 0.0)
            shortage = np.maximum(demand_t - inv_after_transship, 0.0)
            inv_after_fulfill = inv_after_transship - (demand_t if self.backlog_tf else demand_intime)
            inv_after_arrival = inv_after_fulfill + arrival_num
            self.inventory_before_arrive = inv_after_fulfill
            self.transship_instantly_used = np.maximum(trans_actual, 0.0)
        else:
            inv_start = self.inventory
            inv_after_transship = inv_start + trans_actual
            demand_intime = np.maximum(np.minimum(inv_after_transship, demand_t), 0.0)
            shortage = np.maximum(demand_t - inv_after_transship, 0.0)
            inv_after_fulfill = inv_after_transship - (demand_t if self.backlog_tf else demand_intime)

            inv_after_arrival = inv_after_fulfill + arrival_num
            self.inventory_before_arrive = inv_after_fulfill
            self.transship_instantly_used.fill(0.0)

        # Update inventory
        self.inventory = inv_after_arrival
        self.shortage = shortage
        self.demand_fulfilled_intime = demand_intime
        fulfilled_all = demand_t - shortage
        self.demand_fulfilled_all = fulfilled_all
        self.demand_fulfilled_late = np.maximum(fulfilled_all - demand_intime, 0.0)
        self.ordering_times = self.ordering_times + (order_today > 0).astype(np.float32)

        # Costs
        holding_cost = self.H.reshape(1, A) * np.maximum(inv_after_fulfill, 0.0)
        penalty_cost = self.B.reshape(1, A) * shortage
        selling_items_qty = demand_intime + np.maximum(np.minimum(-inv_after_fulfill, arrival_num), 0.0)
        selling_revenue = self.P.reshape(1, A) * selling_items_qty
        self.holding_cost = holding_cost
        self.penalty_cost = penalty_cost
        self.selling_revenue = selling_revenue

        payment_type = self.args.get("payment_type", "selling_first")
        if payment_type == "pay_first":
            ordering_cost = self.C.reshape(1, A) * order_today + self.K.reshape(1, A) * (order_today > 0).astype(np.float32)
        elif payment_type == "product_first":
            # 如果是product_first就是按照arrival收钱
            ordering_cost = self.C.reshape(1, A) * arrival_num + self.K.reshape(1, A) * (arrival_num > 0).astype(np.float32)
        elif payment_type == "selling_first":
            # 如果是selling_first就是按照今天卖了多少收钱
            ordering_cost = self.C.reshape(1, A) * selling_items_qty
        else:
            raise Exception("wrong payment_type")
        self.ordering_cost = ordering_cost

        # Shipping costs charged on outflows
        shipping_cost_all = payment
        self.shipping_cost_all = shipping_cost_all
        self.shipping_cost_pure = shipping_cost_all

        # Residual value at end of episode (eval/test only)
        residual_val = np.zeros((E, A), dtype=np.float32)
        if not self.train_tf and payment_type != "selling_first":
            is_last = (t_index >= (self.episode_length - 1)).reshape(E, 1)
            residual_val = np.where(is_last, np.maximum(self.inventory, 0.0) * self.C.reshape(1, A), 0.0)

        # Reward
        if not self.reward_scale_tf or not self.train_tf:
            reward_selfish = -ordering_cost - shipping_cost_all - holding_cost - penalty_cost + selling_revenue + residual_val
        else:
            # Reward normalization using Z-Score for each quantity component
            # normalize(X) = (X - μ_d) / σ_d, then weight by cost coefficients
            # Get normalization constants (shape: (A,))
            mean_const = self.demand_mean_const.reshape(1, A)
            std_const = np.maximum(self.demand_std_const, 1e-6).reshape(1, A)

            # Z-Score normalize function
            def z_normalize(x):
                return (x - mean_const) / std_const

            # For ordering: normalize the quantity based on payment_type
            if payment_type == "pay_first":
                order_qty_norm = z_normalize(order_today)
                ordering_cost_norm = self.C.reshape(1, A) * order_qty_norm
            elif payment_type == "product_first":
                arrival_qty_norm = z_normalize(arrival_num)
                ordering_cost_norm = self.C.reshape(1, A) * arrival_qty_norm
            else:  # selling_first
                selling_qty_norm = z_normalize(selling_items_qty)
                ordering_cost_norm = self.C.reshape(1, A) * selling_qty_norm

            # For selling revenue
            selling_qty_norm = z_normalize(selling_items_qty)
            selling_revenue_norm = self.P.reshape(1, A) * selling_qty_norm

            # Compute normalized reward (same structure as original)
            reward_selfish = -ordering_cost_norm - shipping_cost_all / std_const - holding_cost / std_const - penalty_cost / std_const + selling_revenue_norm

        self.reward_selfish = reward_selfish
        self.reward_selfish_cum = self.reward_selfish_cum + reward_selfish

        # 计算平均reward（所有agent的平均）
        reward_avg = np.mean(reward_selfish, axis=1, keepdims=True)  # (E, 1)
        reward_avg = np.broadcast_to(reward_avg, (E, A))  # (E, A)

        # 根据self_interest_rate混合selfish reward和average reward
        reward = self.self_interest_rate * reward_selfish + (1 - self.self_interest_rate) * reward_avg

        self.reward = reward
        self.reward_cum = self.reward_cum + reward

        # Compute pooled cost saving for mechanism_reward (if mode is 'pooled_cost_saving')
        # Actual total cost across all agents
        actual_total_holding = holding_cost.sum(axis=1)  # (E,)
        actual_total_penalty = penalty_cost.sum(axis=1)  # (E,)
        actual_total_shipping_cost = shipping_cost_all.sum(axis=1)  # (E,)
        actual_total_cost = actual_total_holding + actual_total_penalty + actual_total_shipping_cost  # (E,)

        # Without pooling scenario: all inventory and demand not pooled together
        # inv_start is the inventory before transship, demand_t is the demand at t
        shortage_without_pooling = np.maximum(demand_t - inv_start, 0.0)  # (E, A)
        holding_without_pooling = np.maximum(inv_start - demand_t, 0.0)  # (E, A)
        cost_without_pooling = self.H.reshape(1, A) * holding_without_pooling + self.B.reshape(1, A) * shortage_without_pooling  # (E, A)
        self.pooled_cost_saving_avg = (cost_without_pooling.sum(axis=1) - actual_total_cost) / self.num_agents  # (E,)

        return (
            reward.astype(np.float32),
            LT_today.astype(np.float32),
            lt_window.astype(np.int32),
            demand_t.astype(np.float32),
        )

    # ---------------------------- Internal: precompute sequences ----------------------------
    def _load_sequences_from_dataset(self):
        """
        Load demand, LT, and shipping_loss sequences from CSV files.
        CSV format: rows=timesteps, columns=episodes
        For each environment, randomly select an episode column.

        Sets:
            self.demand: (E, A, T)
            self.LT: (E, A, T)
            self.shipping_loss: (E, A, T)
        """
        E, A, T = self.num_envs, self.num_agents, self.episode_length
        dataset_name = self.args.get('dataset_name', 'default')

        # Determine data directory
        if self.usage == 'test':
            data_dir = TEST_DIR.format(dataset_name)
        elif self.usage == 'eval':
            data_dir = EVAL_DIR.format(dataset_name)
        else:
            raise ValueError(f"Cannot load dataset for usage={self.usage}")

        # Check if directory exists
        import os
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {data_dir}\n"
                f"Restore the bundled data files or provide the configured dataset."
            )

        # Initialize arrays
        self.demand = np.zeros((E, A, T), dtype=np.float32)
        self.LT = np.zeros((E, A, T), dtype=np.float32)
        self.shipping_loss = np.zeros((E, A, T), dtype=np.float32)

        # Load data for each agent
        for agent_idx in range(A):
            agent_data_dir = os.path.join(data_dir, f"agent_{agent_idx}")

            # Load CSV files
            demand_file = os.path.join(agent_data_dir, "demand.csv")
            LT_file = os.path.join(agent_data_dir, "LT.csv")
            shippingloss_file = os.path.join(agent_data_dir, "shippingloss.csv")

            # Check files exist
            if not os.path.exists(demand_file):
                raise FileNotFoundError(f"Missing file: {demand_file}")
            if not os.path.exists(LT_file):
                raise FileNotFoundError(f"Missing file: {LT_file}")
            if not os.path.exists(shippingloss_file):
                raise FileNotFoundError(f"Missing file: {shippingloss_file}")

            # Load data: shape (T, num_episodes)
            demand_all_episodes = np.loadtxt(demand_file, delimiter=',', dtype=np.float32)
            LT_all_episodes = np.loadtxt(LT_file, delimiter=',', dtype=np.float32)
            shippingloss_all_episodes = np.loadtxt(shippingloss_file, delimiter=',', dtype=np.float32)

            # Handle 1D case (single episode)
            if demand_all_episodes.ndim == 1:
                demand_all_episodes = demand_all_episodes.reshape(-1, 1)
                LT_all_episodes = LT_all_episodes.reshape(-1, 1)
                shippingloss_all_episodes = shippingloss_all_episodes.reshape(-1, 1)

            num_available_episodes = demand_all_episodes.shape[1]

            for filename, values in ((demand_file, demand_all_episodes), (LT_file, LT_all_episodes), (shippingloss_file, shippingloss_all_episodes)):
                if values.shape[0] < T or values.shape[1] != num_available_episodes:
                    raise ValueError(f"Invalid episode shape in {filename}: {values.shape}")
            if self.eval_dataset_mode == "all":
                if num_available_episodes != E:
                    raise ValueError(f"Episode count mismatch in {demand_file}: expected {E}, got {num_available_episodes}")
                episode_indices = np.arange(E)
            else:
                np.random.seed(42)
                episode_indices = np.random.randint(0, num_available_episodes, size=E)
            self.evaluation_episode_indices = episode_indices.copy()
            self.demand[:, agent_idx, :] = demand_all_episodes[:T, episode_indices].T
            self.LT[:, agent_idx, :] = LT_all_episodes[:T, episode_indices].T
            self.shipping_loss[:, agent_idx, :] = shippingloss_all_episodes[:T, episode_indices].T

    def _precompute_sequences(self):
        """
        Precompute all stochastic sequences using vectorized generators.
        If usage is 'eval' or 'test', load from CSV files based on dataset_name.

        Generates:
            demand: (E, A, T)
            LT: (E, A, T) - total lead time (simplified)
            shipping_loss: (E, A, T)
            demand_est: (E, A, T)
            demand_std_est: (E, A, T)
            order_arrival_time: (E, A, T) - 如果在t时刻订货，将会在什么时候收到货

        Precomputed critic statistics (for efficiency):
            precomputed_long_mean: (E, A, T) - long window mean for each timestep
            precomputed_long_std: (E, A, T) - long window std for each timestep
            precomputed_long_quantile: (E, A, T, 5) - long window quantiles [0.05,0.25,0.5,0.75,0.95]
            precomputed_short_mean: (E, A, T) - short window mean for each timestep
            precomputed_short_all: (E, A, T, inv_h) - short window all values for each timestep
        """
        E, A, T = self.num_envs, self.num_agents, self.episode_length
        shape = (E, A, T)

        # Check if we should load from dataset
        if self.usage in ['eval', 'test']:
            # Load sequences from CSV files
            self._load_sequences_from_dataset()
            # Create a generator to get theoretical mean/std for normalization
            demand_spec = self.args.get("demand_generators")
            demand_gen = create_generator(shape, demand_spec)
        else:
            # Generate sequences using generators (train mode)
            # Demand generation
            demand_spec = self.args.get("demand_generators")
            demand_gen = create_generator(shape, demand_spec)
            self.demand = demand_gen.generate(round_int=True, clip_nonneg=True)

            # Shipping loss generation
            shipping_loss_spec = self.args.get("shipping_loss_generators")
            if shipping_loss_spec is not None:
                shipping_loss_gen = create_generator(shape, shipping_loss_spec)
                self.shipping_loss = shipping_loss_gen.generate(round_int=False, clip_nonneg=True)
            else:
                self.shipping_loss = np.zeros(shape, dtype=np.float32)

            # LT generation (simplified: single total LT instead of per-stage)
            lt_spec = self.args.get("LT_generators")
            if isinstance(lt_spec, list):
                # If list, sum all stages
                total_lt = np.zeros(shape, dtype=np.float32)
                for stage_spec in lt_spec:
                    stage_gen = create_generator(shape, stage_spec)
                    total_lt += stage_gen.generate(round_int=True, clip_nonneg=True)
                self.LT = total_lt
            else:
                lt_gen = create_generator(shape, lt_spec)
                self.LT = lt_gen.generate(round_int=True, clip_nonneg=True)

        self.LT = np.maximum(self.LT, 1)
        self.order_arrival_time = (self.LT + np.arange(T).reshape(1, 1, T)).astype(np.int32)
        self.lt_avg_agents = np.mean(self.LT, axis=(0, 2)) # shape (A,)

        # Compute LT mean/std constants for normalization (per agent, shape (A,))
        lt_spec = self.args.get("LT_generators")
        if isinstance(lt_spec, list):
            # If list, sum all stages' theoretical mean/std
            lt_mean_const = np.zeros(A, dtype=np.float32)
            lt_var_const = np.zeros(A, dtype=np.float32)
            for stage_spec in lt_spec:
                stage_gen = create_generator(shape, stage_spec)
                stage_mean, stage_std = stage_gen.get_theoretical_mean_std()
                lt_mean_const += stage_mean
                lt_var_const += stage_std ** 2  # sum of variances for independent stages
            lt_std_const = np.sqrt(lt_var_const)
        else:
            lt_gen = create_generator(shape, lt_spec)
            lt_mean_const, lt_std_const = lt_gen.get_theoretical_mean_std()
        self.lt_mean_const = lt_mean_const  # shape (A,)
        self.lt_std_const = lt_std_const    # shape (A,)

        # Estimate demand mean/std for all t
        mean_const, std_const = demand_gen.get_theoretical_mean_std()
        # Store constants for normalization
        self.demand_mean_const = mean_const
        self.demand_std_const = std_const
        est_method = self.args.get("estimate_demand_method", "rolling_mean")
        if est_method == "mean":
            # Use generator's theoretical mean/std
            mean_const, std_const = demand_gen.get_theoretical_mean_std()
            # mean_const and std_const are (A,), need to broadcast to (E, A, T)
            mean_arr = np.broadcast_to(mean_const.reshape(1, A, 1), (E, A, T + 1)).astype(np.float32)
            std_arr = np.broadcast_to(std_const.reshape(1, A, 1), (E, A, T + 1)).astype(np.float32)
            self.demand_est = mean_arr
            self.demand_std_est = std_arr
        elif est_method == "rolling_mean":
            window = int(self.args.get("rolling_window", 10))
            csum = np.cumsum(self.demand, axis=2)
            csum2 = np.cumsum(self.demand ** 2, axis=2)
            pad0 = np.zeros((E, A, 1), dtype=np.float32)
            csum_pad = np.concatenate([pad0, csum], axis=2)
            csum2_pad = np.concatenate([pad0, csum2], axis=2)
            # indices for start = t - w + 1 (clipped at 0)
            idx = np.arange(T)
            start = np.maximum(idx - window + 1, 0)
            length = (idx - start) + 1
            # broadcast to (E,A,T)
            start_b = start.reshape(1, 1, T)
            end_b = (idx + 1).reshape(1, 1, T)
            length_b = length.reshape(1, 1, T)
            sum_w = csum_pad[:, :, end_b[0, 0, :]] - csum_pad[:, :, start_b[0, 0, :]]
            sum2_w = csum2_pad[:, :, end_b[0, 0, :]] - csum2_pad[:, :, start_b[0, 0, :]]
            self.demand_est = sum_w / np.maximum(length_b, 1)
            var = sum2_w / np.maximum(length_b, 1) - self.demand_est ** 2
            var = np.maximum(var, 0.0)
            self.demand_std_est = np.sqrt(var)
            # Set the first window//2 elements of self.demand_std_est to std_const for all envs and agents
            init_n = window // 2
            if init_n > 0:
                # std_const is shape (A,)
                std_const_expand = np.broadcast_to(std_const.reshape(1, A, 1), (E, A, init_n)).astype(np.float32)
                self.demand_std_est[:, :, :init_n] = std_const_expand
            self.demand_est = np.concatenate([np.broadcast_to(self.demand_mean_const.reshape(1, A, 1), (E, A, 1)).astype(np.float32), self.demand_est], axis=2)
            self.demand_std_est = np.concatenate([np.broadcast_to(std_const.reshape(1, A, 1), (E, A, 1)).astype(np.float32), self.demand_std_est], axis=2)
        elif est_method == "ema":
            alpha = float(self.args.get("alpha", 0.3))
            EAT_shape = (E, A)
            est = np.zeros((E, A, T + 1), dtype=np.float32)
            std_est_sqr = np.zeros((E, A, T + 1), dtype=np.float32)
            # t=0 initialization
            est[:, :, 0] = np.broadcast_to(self.demand_mean_const.reshape(1, A), (E, A)).astype(np.float32)
            std_est_sqr[:, :, 0] = np.broadcast_to(std_const.reshape(1, A), (E, A)).astype(np.float32)
            # vectorized over E,A; single loop over time is acceptable
            one_minus = 1.0 - alpha
            for t in range(1, T + 1):
                cur = self.demand[:, :, t - 1]
                est[:, :, t] = alpha * cur + one_minus * est[:, :, t - 1]
                std_est_sqr[:, :, t] = alpha * (cur - est[:, :, t - 1]) ** 2 + one_minus * std_est_sqr[:, :, t - 1]
            self.demand_est = est
            self.demand_std_est = np.sqrt(std_est_sqr)
        else:
            raise Exception("wrong estimate_demand_method")

        # Precompute demand statistics for critic observation (vectorized for all timesteps)
        self._precompute_critic_demand_stats()

    def _precompute_critic_demand_stats(self):
        """
        Precompute demand statistics for critic observation to avoid redundant computation.
        This computes long/short window statistics for all timesteps at once.

        Precomputes:
            - precomputed_long_mean: (E, A, T)
            - precomputed_long_std: (E, A, T) if needed
            - precomputed_long_quantile: (E, A, T, 5) if needed
            - precomputed_short_mean: (E, A, T)
            - precomputed_short_all: (E, A, T, inv_h) if needed
        """
        E, A, T = self.num_envs, self.num_agents, self.episode_length
        inv_h = int(self.args["inventory_est_horizon"])
        gamma = float(self.args["gamma"]) if "gamma" in self.args else 0.99
        long_len = int(round(1.0 / (1.0 - gamma + 1e-10)))
        demand_info = self.args["demand_info_for_critic"]

        # Precompute for all timesteps at once using advanced indexing
        # For each timestep t, we need windows starting from t+inv_h (long) and t (short)
        # Use mirror reflection when indices exceed T: T+5 → T-5, 2T+3 → T-3

        # Create indices for all timesteps: shape (T, 1, 1)
        t_all = np.arange(T).reshape(T, 1, 1)

        # Long window indices: (T, 1, long_len)
        k_long = np.arange(long_len).reshape(1, 1, long_len)
        start_long = t_all + inv_h
        idx_long_all = start_long + k_long  # (T, 1, long_len)
        # Mirror reflection: if idx >= T, reflect back
        # Formula: idx_mod = idx % (2*T), if idx_mod >= T: result = 2*T - idx_mod - 1
        idx_long_mod = idx_long_all % (2 * T)
        idx_long_all = np.where(idx_long_mod >= T, 2 * T - idx_long_mod - 1, idx_long_mod)

        # Short window indices: (T, 1, inv_h)
        k_short = np.arange(inv_h).reshape(1, 1, inv_h)
        idx_short_all = t_all + k_short  # (T, 1, inv_h)
        # Apply same mirror reflection
        idx_short_mod = idx_short_all % (2 * T)
        idx_short_all = np.where(idx_short_mod >= T, 2 * T - idx_short_mod - 1, idx_short_mod)

        # Extract windows for all timesteps using memory-efficient approach
        # Since indices only depend on t (not on e or a), we don't need (E, A, T, window_len)
        # Instead, use (T, window_len) indices and gather on-the-fly for each (e, a)

        # Simplify indices: (T, 1, window_len) -> (T, window_len)
        idx_long_simple = idx_long_all[:, 0, :].astype(np.int32)  # (T, long_len)
        idx_short_simple = idx_short_all[:, 0, :].astype(np.int32)  # (T, inv_h)

        # Compute statistics directly without storing full windows
        # For each statistic, loop over T (unavoidable for memory efficiency)
        if "long_mean" in demand_info:
            long_means = np.zeros((E, A, T), dtype=np.float32)
            for t_val in range(T):
                # demand[:, :, idx_long_simple[t_val, :]] gives (E, A, long_len)
                window = self.demand[:, :, idx_long_simple[t_val, :]]  # (E, A, long_len)
                long_means[:, :, t_val] = window.mean(axis=2)
            self.precomputed_long_mean = long_means
        else:
            self.precomputed_long_mean = None

        if "long_std" in demand_info:
            long_stds = np.zeros((E, A, T), dtype=np.float32)
            for t_val in range(T):
                window = self.demand[:, :, idx_long_simple[t_val, :]]  # (E, A, long_len)
                long_stds[:, :, t_val] = window.std(axis=2)
            self.precomputed_long_std = long_stds
        else:
            self.precomputed_long_std = None

        if "long_quantile" in demand_info:
            qs = [0.05, 0.25, 0.5, 0.75, 0.95]
            long_quantiles = np.zeros((E, A, T, 5), dtype=np.float32)
            for t_val in range(T):
                window = self.demand[:, :, idx_long_simple[t_val, :]]  # (E, A, long_len)
                # Compute quantiles along axis=2 (window dimension)
                for q_idx, q in enumerate(qs):
                    long_quantiles[:, :, t_val, q_idx] = np.quantile(window, q, axis=2)
            self.precomputed_long_quantile = long_quantiles
        else:
            self.precomputed_long_quantile = None

        if "short_mean" in demand_info:
            short_means = np.zeros((E, A, T), dtype=np.float32)
            for t_val in range(T):
                window = self.demand[:, :, idx_short_simple[t_val, :]]  # (E, A, inv_h)
                short_means[:, :, t_val] = window.mean(axis=2)
            self.precomputed_short_mean = short_means
        else:
            self.precomputed_short_mean = None

        if "short_all" in demand_info:
            # This one needs to store the full window, but more memory efficient than before
            short_all_windows = np.zeros((E, A, T, inv_h), dtype=np.float32)
            for t_val in range(T):
                short_all_windows[:, :, t_val, :] = self.demand[:, :, idx_short_simple[t_val, :]]
            self.precomputed_short_all = short_all_windows
        else:
            self.precomputed_short_all = None

    def close(self):
        pass
