"""
Vectorized generators for demand, lead time, and shipping loss.
All generators support batch generation with shape (E, A, T) where:
    E: num_envs
    A: num_agents
    T: episode_length
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from harl.envs.multi_lt_transship import SHANSHU_GEN_SOURCE

INF = 1e9


class VectorizedGenerator:
    """
    Base vectorized generator.

    Shape convention:
        - Input shape: (E, A, T) where E=envs, A=agents, T=timesteps
        - Output: np.ndarray of shape (E, A, T)
        - Parameters can be scalars or arrays of shape (A) for per-agent values
    """

    def __init__(self, shape, max_num=INF):
        """
        Args:
            shape: tuple (E, A, T)
            max_num: maximum value to clip (scalar or array of shape (A))
        """
        self.shape = shape
        self.E, self.A, self.T = shape
        self.max_num = self._normalize_param(max_num)
        self.theo_mean = None
        self.theo_std = None

    def _normalize_param(self, param):
        """
        Normalize parameter to shape (A).

        Args:
            param: scalar or array-like of shape (A)

        Returns:
            np.ndarray of shape (A)
        """
        if np.isscalar(param):
            return np.full(self.A, param, dtype=np.float32)
        else:
            param = np.asarray(param, dtype=np.float32)
            if param.shape == ():
                return np.full(self.A, param.item(), dtype=np.float32)
            elif param.shape == (self.A,):
                return param
            else:
                raise ValueError(f"Parameter shape {param.shape} incompatible with A={self.A}")

    def _broadcast_to_EAT(self, param_A):
        """
        Broadcast parameter from shape (A) to (E, A, T).

        Args:
            param_A: array of shape (A)

        Returns:
            array of shape (E, A, T)
        """
        # Reshape to (1, A, 1) then broadcast to (E, A, T)
        return np.broadcast_to(param_A[np.newaxis, :, np.newaxis], self.shape)

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Generate data.

        Returns:
            np.ndarray: shape (E, A, T), dtype float32
        """
        raise NotImplementedError

    def get_theoretical_mean_std(self):
        """
        Returns:
            tuple: (mean: np.ndarray of shape (A), std: np.ndarray of shape (A))
        """
        raise NotImplementedError


class UniformGenerator(VectorizedGenerator):
    """
    Uniform distribution: U(min_num, max_num)

    Args:
        shape: (E, A, T)
        min_num: scalar or array of shape (A)
        max_num: scalar or array of shape (A)
    """

    def __init__(self, shape, min_num=0, max_num=20):
        super().__init__(shape, max_num)
        self.min_num = self._normalize_param(min_num)

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Returns:
            np.ndarray: shape (E, A, T)
        """
        # Broadcast parameters to (E, A, T)
        min_EAT = self._broadcast_to_EAT(self.min_num)
        max_EAT = self._broadcast_to_EAT(self.max_num)

        data = np.random.uniform(min_EAT, max_EAT, size=self.shape)
        if clip_nonneg:
            data = np.clip(data, min_EAT, max_EAT)
        if round_int:
            data = np.round(data)
        return data.astype(np.float32)

    def get_theoretical_mean_std(self):
        """
        Returns:
            (mean, std): tuple of arrays of shape (A)
        """
        if self.theo_mean is None:
            self.theo_mean = (self.min_num + self.max_num) / 2.0
            self.theo_std = ((self.max_num - self.min_num) ** 2 / 12.0) ** 0.5
        return self.theo_mean, self.theo_std


class NormalGenerator(VectorizedGenerator):
    """
    Normal distribution: N(mean, std^2)

    Args:
        shape: (E, A, T)
        mean: scalar or array of shape (A)
        std: scalar or array of shape (A)
        max_num: scalar or array of shape (A)
    """

    def __init__(self, shape, mean=10, std=2, max_num=INF):
        super().__init__(shape, max_num)
        self.mean = self._normalize_param(mean)
        self.std = self._normalize_param(std)

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Returns:
            np.ndarray: shape (E, A, T)
        """
        # Broadcast parameters to (E, A, T)
        mean_EAT = self._broadcast_to_EAT(self.mean)
        std_EAT = self._broadcast_to_EAT(self.std)
        max_EAT = self._broadcast_to_EAT(self.max_num)

        data = np.random.normal(mean_EAT, std_EAT, size=self.shape)
        if clip_nonneg:
            data = np.clip(data, 0, max_EAT)
        if round_int:
            data = np.round(data)
        return data.astype(np.float32)

    def get_theoretical_mean_std(self):
        """
        Truncated normal mean/std.

        Returns:
            (mean, std): tuple of arrays of shape (A)
        """
        if self.theo_mean is None:
            a = (0 - self.mean) / self.std
            b = (self.max_num - self.mean) / self.std
            mean = self.mean + self.std * (norm.pdf(a) - norm.pdf(b)) / (norm.cdf(b) - norm.cdf(a))
            var = (1 + (a * norm.pdf(a) - b * norm.pdf(b)) / (norm.cdf(b) - norm.cdf(a))
                   - ((norm.pdf(a) - norm.pdf(b)) / (norm.cdf(b) - norm.cdf(a))) ** 2) * self.std ** 2
            self.theo_mean = mean
            self.theo_std = var ** 0.5
        return self.theo_mean, self.theo_std


class PoissonGenerator(VectorizedGenerator):
    """
    Poisson distribution: Poisson(mean)

    Args:
        shape: (E, A, T)
        mean: scalar or array of shape (A)
        max_num: scalar or array of shape (A)
    """

    def __init__(self, shape, mean=10, max_num=INF):
        super().__init__(shape, max_num)
        self.mean = self._normalize_param(mean)

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Returns:
            np.ndarray: shape (E, A, T)
        """
        # Broadcast parameters to (E, A, T)
        mean_EAT = self._broadcast_to_EAT(self.mean)
        max_EAT = self._broadcast_to_EAT(self.max_num)

        data = np.random.poisson(mean_EAT, size=self.shape).astype(np.float32)
        if clip_nonneg:
            data = np.clip(data, 0, max_EAT)
        return data

    def get_theoretical_mean_std(self):
        """
        Returns:
            (mean, std): tuple of arrays of shape (A)
        """
        if self.theo_mean is None:
            self.theo_mean = self.mean
            self.theo_std = self.mean ** 0.5
        return self.theo_mean, self.theo_std


class ConstantGenerator(VectorizedGenerator):
    """
    Constant value.

    Args:
        shape: (E, A, T)
        value: scalar or array of shape (A)
    """

    def __init__(self, shape, value=10):
        super().__init__(shape, value)
        self.value = self._normalize_param(value)

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Returns:
            np.ndarray: shape (E, A, T)
        """
        # Broadcast value to (E, A, T)
        data = self._broadcast_to_EAT(self.value).copy()
        if round_int:
            data = np.round(data)
        return data.astype(np.float32)

    def get_theoretical_mean_std(self):
        """
        Returns:
            (mean, std): tuple of arrays of shape (A)
        """
        if self.theo_mean is None:
            self.theo_mean = self.value
            self.theo_std = np.zeros(self.A, dtype=np.float32)
        return self.theo_mean, self.theo_std


class KimGenerator(VectorizedGenerator):
    """
    Kim distribution: mean ~ U(meanMin, meanMax), std ~ U(stdMin, stdMax), X ~ N(mean, std^2)

    Args:
        shape: (E, A, T)
        meanMin: scalar or array of shape (A)
        meanMax: scalar or array of shape (A)
        stdMin: scalar or array of shape (A)
        stdMax: scalar or array of shape (A)
        max_num: scalar or array of shape (A)
    """

    def __init__(self, shape, meanMin=5, meanMax=30, stdMin=1, stdMax=5, max_num=INF):
        super().__init__(shape, max_num)
        self.meanMin = self._normalize_param(meanMin)
        self.meanMax = self._normalize_param(meanMax)
        self.stdMin = self._normalize_param(stdMin)
        self.stdMax = self._normalize_param(stdMax)

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Returns:
            np.ndarray: shape (E, A, T)
        """
        # Broadcast parameters to (E, A, T)
        meanMin_EAT = self._broadcast_to_EAT(self.meanMin)
        meanMax_EAT = self._broadcast_to_EAT(self.meanMax)
        stdMin_EAT = self._broadcast_to_EAT(self.stdMin)
        stdMax_EAT = self._broadcast_to_EAT(self.stdMax)
        max_EAT = self._broadcast_to_EAT(self.max_num)

        # Vectorized: sample mean and std for each (e, a, t)
        means = np.random.uniform(meanMin_EAT, meanMax_EAT, size=self.shape)
        stds = np.random.uniform(stdMin_EAT, stdMax_EAT, size=self.shape)
        data = np.random.normal(means, stds)
        if clip_nonneg:
            data = np.clip(data, 0, max_EAT)
        if round_int:
            data = np.round(data)
        return data.astype(np.float32)

    def get_theoretical_mean_std(self):
        """
        Returns:
            (mean, std): tuple of arrays of shape (A)
        """
        if self.theo_mean is None:
            self.theo_mean = (self.meanMin + self.meanMax) / 2.0
            self.theo_std = (self.stdMin + self.stdMax) / 2.0
        return self.theo_mean, self.theo_std


class MertonGenerator(VectorizedGenerator):
    """
    Merton jump diffusion (simplified vectorized version).

    Args:
        shape: (E, A, T)
        std: scalar or array of shape (A)
        max_num: scalar or array of shape (A)
    """

    def __init__(self, shape, std=4, max_num=INF):
        super().__init__(shape, max_num)
        self.std = self._normalize_param(std)

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Returns:
            np.ndarray: shape (E, A, T)
        """
        E, A, T = self.shape
        base = self.max_num / 2.0  # shape (A)

        # Vectorized Merton process for all envs and agents
        var = self.std ** 2  # shape (A)
        delta = 0.01
        delta_t = var  # shape (A)
        u = 0.5 * delta * delta
        a = 0
        b = 0.01
        lamda = var  # shape (A)

        # Initialize log-process
        log_X = np.zeros((E, A, T + 1), dtype=np.float32)

        # Reshape parameters for broadcasting: (1, A, 1)
        delta_t_EA = delta_t[np.newaxis, :, np.newaxis]  # (1, A, 1)
        lamda_EA = lamda[np.newaxis, :, np.newaxis]  # (1, A, 1)

        for t in range(T):
            Z = np.random.normal(0, 1, size=(E, A))
            N = np.random.poisson(lamda[np.newaxis, :], size=(E, A))
            Z_2 = np.random.normal(0, 2, size=(E, A))
            M = a * N + b * (N ** 0.5) * Z_2
            log_X[:, :, t + 1] = log_X[:, :, t] + u - 0.5 * delta * delta + (delta_t[np.newaxis, :] ** 0.5) * delta * Z + M

        data = np.exp(log_X[:, :, 1:]) * base[np.newaxis, :, np.newaxis]
        max_EAT = self._broadcast_to_EAT(self.max_num)
        if clip_nonneg:
            data = np.clip(data, 0, max_EAT)
        if round_int:
            data = np.round(data)
        return data.astype(np.float32)

    def get_theoretical_mean_std(self):
        """
        Returns:
            (mean, std): tuple of arrays of shape (A)
        """
        if self.theo_mean is None:
            self.theo_mean = self.max_num / 2.0
            self.theo_std = self.std
        return self.theo_mean, self.theo_std


class KimMertonGenerator(VectorizedGenerator):
    """
    Kim-Merton: mean/std follow Merton process, then X ~ N(mean, std^2)

    Args:
        shape: (E, A, T)
        mean_std: scalar or array of shape (A)
        mean_max: scalar or array of shape (A)
        std_std: scalar or array of shape (A)
        std_max: scalar or array of shape (A)
        max_num: scalar or array of shape (A)
    """

    def __init__(self, shape, mean_std=4, mean_max=30, std_std=1, std_max=5, max_num=INF):
        super().__init__(shape, max_num)
        self.mean_std = self._normalize_param(mean_std)
        self.mean_max = self._normalize_param(mean_max)
        self.std_std = self._normalize_param(std_std)
        self.std_max = self._normalize_param(std_max)

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Returns:
            np.ndarray: shape (E, A, T)
        """
        # Generate mean and std via Merton (parameters already normalized to shape (A))
        mean_gen = MertonGenerator(self.shape, std=self.mean_std, max_num=self.mean_max)
        std_gen = MertonGenerator(self.shape, std=self.std_std, max_num=self.std_max)

        means = mean_gen.generate(round_int=False, clip_nonneg=True)
        stds = std_gen.generate(round_int=False, clip_nonneg=True)

        # Sample from N(mean, std)
        data = np.random.normal(means, stds)
        max_EAT = self._broadcast_to_EAT(self.max_num)
        if clip_nonneg:
            data = np.clip(data, 0, max_EAT)
        if round_int:
            data = np.round(data)
        return data.astype(np.float32)

    def get_theoretical_mean_std(self):
        """
        Returns:
            (mean, std): tuple of arrays of shape (A)
        """
        if self.theo_mean is None:
            self.theo_mean = self.mean_max / 2.0
            self.theo_std = self.mean_std / 2.0
        return self.theo_mean, self.theo_std


class ShanshuGenerator(VectorizedGenerator):
    """
    Shanshu: load from CSV data.
    Note: For vectorization, we sample random windows from the dataset.

    Args:
        shape: (E, A, T)
        SKU_id: str
        agent_idx: int or list of int (if per-agent different)
        max_num: scalar or array of shape (A)
    """

    def __init__(self, shape, SKU_id, max_num=INF):
        super().__init__(shape, max_num)
        self.SKU_id = SKU_id
        E, A, T = shape
        # Load data for each agent (CSV has no header, single column of values)
        self.sales_list = []
        for agent_idx in range(A):
            data_path = SHANSHU_GEN_SOURCE.format(SKU_id, agent_idx)
            # Read CSV without header, first column is the sales data
            df = pd.read_csv(data_path, header=None)
            sales = df.iloc[:, 0].values.astype(np.float32)
            self.sales_list.append(sales)

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Returns:
            np.ndarray: shape (E, A, T)
        """
        E, A, T = self.shape
        data = np.zeros((E, A, T), dtype=np.float32)

        # For each env and agent, sample a random contiguous window
        len_sales = len(self.sales_list[0])
        for e in range(E):
            start = np.random.randint(0, len_sales - T + 1)
            for a in range(A):
                sales = self.sales_list[a]
                assert len(sales) >= T, f"Shanshu data length ({len(sales)}) < required T ({T}) for agent {a}"
                data[e, a, :] = sales[start:start + T]

        max_EAT = self._broadcast_to_EAT(self.max_num)
        if clip_nonneg:
            data = np.clip(data, 0, max_EAT)
        if round_int:
            data = np.round(data)
        return data.astype(np.float32)

    def get_theoretical_mean_std(self):
        """
        Returns:
            (mean, std): tuple of arrays of shape (A)
        """
        if self.theo_mean is None:
            # Compute mean and std for each agent from their respective data
            means = []
            stds = []
            for a in range(self.A):
                sales = self.sales_list[a]
                means.append(np.mean(sales))
                stds.append(np.std(sales))
            self.theo_mean = np.array(means, dtype=np.float32)
            self.theo_std = np.array(stds, dtype=np.float32)
        return self.theo_mean, self.theo_std


class CompositeGenerator(VectorizedGenerator):
    """
    Composite generator that combines multiple generators, one per agent.
    Used when spec is a list, where each element specifies a different generator for each agent.

    Args:
        shape: (E, A, T)
        generators: list of VectorizedGenerator instances, one per agent (length A)
    """

    def __init__(self, shape, generators):
        # Don't call super().__init__ with max_num since each generator has its own
        self.shape = shape
        self.E, self.A, self.T = shape
        self.generators = generators
        self.theo_mean = None
        self.theo_std = None

        if len(generators) != self.A:
            raise ValueError(f"Number of generators ({len(generators)}) must match number of agents ({self.A})")

    def generate(self, round_int=True, clip_nonneg=True):
        """
        Generate data by combining outputs from each agent's generator.

        Returns:
            np.ndarray: shape (E, A, T)
        """
        # Each generator has shape (E, 1, T), concatenate along agent axis
        agent_data_list = []
        for gen in self.generators:
            # Generate shape (E, 1, T) from each generator
            data = gen.generate(round_int=round_int, clip_nonneg=clip_nonneg)
            agent_data_list.append(data)

        # Concatenate along axis 1 (agent axis): (E, 1, T) * A -> (E, A, T)
        return np.concatenate(agent_data_list, axis=1).astype(np.float32)

    def get_theoretical_mean_std(self):
        """
        Get theoretical mean and std by combining from each agent's generator.

        Returns:
            (mean, std): tuple of arrays of shape (A)
        """
        if self.theo_mean is None:
            means = []
            stds = []
            for gen in self.generators:
                m, s = gen.get_theoretical_mean_std()
                # Each generator returns shape (1,), take the scalar value
                means.append(m[0] if hasattr(m, '__len__') else m)
                stds.append(s[0] if hasattr(s, '__len__') else s)
            self.theo_mean = np.array(means, dtype=np.float32)
            self.theo_std = np.array(stds, dtype=np.float32)
        return self.theo_mean, self.theo_std


def _create_single_generator(shape, spec):
    """
    Internal function to create a single generator from spec dict.

    Args:
        shape: tuple (E, A, T) - typically (E, 1, T) when called from CompositeGenerator
        spec: dict with keys 'gen_func' and 'gen_args'

    Returns:
        VectorizedGenerator instance
    """
    if not isinstance(spec, dict) or 'gen_func' not in spec:
        # Default: constant 0
        return ConstantGenerator(shape, value=0)

    gen_func = spec['gen_func']
    gen_args = spec.get('gen_args', {})

    if gen_func == 'uniform':
        return UniformGenerator(shape, **gen_args)
    elif gen_func == 'normal':
        return NormalGenerator(shape, **gen_args)
    elif gen_func == 'poisson':
        return PoissonGenerator(shape, **gen_args)
    elif gen_func == 'constant':
        return ConstantGenerator(shape, **gen_args)
    elif gen_func == 'kim':
        return KimGenerator(shape, **gen_args)
    elif gen_func == 'merton':
        return MertonGenerator(shape, **gen_args)
    elif gen_func == 'kim_merton':
        return KimMertonGenerator(shape, **gen_args)
    elif gen_func == 'shanshu':
        return ShanshuGenerator(shape, **gen_args)
    else:
        raise ValueError(f"Unknown gen_func: {gen_func}")


def create_generator(shape, spec):
    """
    Factory function to create vectorized generator from spec dict or list.

    Args:
        shape: tuple (E, A, T)
        spec: dict with keys 'gen_func' and 'gen_args', OR
              list of dicts where each element specifies generator for corresponding agent

    Returns:
        VectorizedGenerator instance

    Examples:
        # Single spec for all agents
        spec = {'gen_func': 'normal', 'gen_args': {'mean': 10, 'std': 2}}

        # Per-agent specs (list)
        spec = [
            {'gen_func': 'normal', 'gen_args': {'mean': 10, 'std': 2}},  # agent 0
            {'gen_func': 'poisson', 'gen_args': {'mean': 15}},           # agent 1
            {'gen_func': 'uniform', 'gen_args': {'min_num': 5, 'max_num': 20}},  # agent 2
        ]
    """
    E, A, T = shape

    # Handle list spec: each element is a spec for one agent
    if isinstance(spec, list):
        if len(spec) < A:
            raise ValueError(f"spec list length ({len(spec)}) must be >= number of agents ({A})")

        # Create a generator for each agent with shape (E, 1, T)
        # Only use the first A specs
        generators = []
        for agent_spec in spec[:A]:
            agent_shape = (E, 1, T)
            gen = _create_single_generator(agent_shape, agent_spec)
            generators.append(gen)

        return CompositeGenerator(shape, generators)

    # Handle single dict spec (original behavior)
    return _create_single_generator(shape, spec)
