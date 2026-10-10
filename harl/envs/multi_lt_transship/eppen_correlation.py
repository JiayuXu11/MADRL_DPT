"""Training-demand correlation for the optional Eppen risk-pooling extension."""
from pathlib import Path

import numpy as np

from harl.envs.multi_lt_transship import SHANSHU_GEN_SOURCE


def correlation_from_sequences(sequences):
    """Pearson matrix on aligned observations; constant series stay independent."""
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError('Eppen correlation requires finite (agents, time) training data with at least two periods')
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered ** 2, axis=1))
    standardized = np.divide(centered, norms[:, None], out=np.zeros_like(centered), where=norms[:, None] > 0)
    correlation = standardized @ standardized.T
    np.fill_diagonal(correlation, 1.0)
    return correlation


def load_training_correlation(args, num_agents):
    """Read only the aligned gen_source training prefix, never eval/test files."""
    spec = args.get('demand_generators', {})
    if not isinstance(spec, dict) or spec.get('gen_func') != 'shanshu':
        raise ValueError('eppen_use_demand_correlation requires a shanshu SKU training source')
    sku = spec['gen_args']['SKU_id']
    sequences = [np.loadtxt(Path(SHANSHU_GEN_SOURCE.format(sku, agent)), delimiter=',', ndmin=1)
                 for agent in range(num_agents)]
    if any(sequence.ndim != 1 for sequence in sequences):
        raise ValueError('Eppen training source must contain one demand column per agent')
    length = min(map(len, sequences))
    requested = args.get('eppen_correlation_train_length')
    if requested is not None:
        if isinstance(requested, bool) or int(requested) != requested or int(requested) < 2:
            raise ValueError('eppen_correlation_train_length must be an integer of at least two')
        length = min(length, int(requested))
    correlation = correlation_from_sequences([sequence[:length] for sequence in sequences])
    return correlation, length


def pooled_demand_variance(standard_deviation, correlation):
    """Return sigma' R sigma per environment, including both covariance triangles."""
    variance = np.einsum('ei,ij,ej->e', standard_deviation, correlation, standard_deviation)
    return np.maximum(variance, 0.0)[:, None]
