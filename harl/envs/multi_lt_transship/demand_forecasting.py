"""Demand forecasts for heuristic baselines only. RL keeps its original code.

The returned value at index ``t`` is computed from observations strictly before
``t``.  ARIMA parameters and XGBoost trees are fitted on an independent set of
training trajectories supplied by the caller; the evaluated trajectory is only
used to update the forecasting state.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


SUPPORTED_FORECAST_METHODS = ("ema", "arima", "xgboost")
_FORECAST_CACHE: Dict[Tuple[object, ...], Tuple[np.ndarray, np.ndarray]] = {}
_MODEL_CACHE: Dict[Tuple[object, ...], object] = {}

BASELINE_FORECAST_DEFAULTS = {
    "alpha": 0.3,
    "forecast_random_seed": 20260904,
    "forecast_train_episodes": 30,
    "forecast_train_length": 1000,
    "arima_order": [3, 1, 1],
    "xgboost_lags": 8,
    "xgboost_n_estimators": 120,
    "xgboost_max_depth": 2,
    "xgboost_learning_rate": 0.035,
}


def make_training_demand(args, num_agents):
    from harl.envs.multi_lt_transship import ENV_ARGS_DEFAULT
    from harl.envs.multi_lt_transship.generator_vec import create_generator

    config = {**BASELINE_FORECAST_DEFAULTS, **args}
    state = np.random.get_state()
    try:
        np.random.seed(int(config["forecast_random_seed"]))
        shape = (int(config["forecast_train_episodes"]), num_agents,
                 int(config["forecast_train_length"]))
        generator = create_generator(
            shape, config.get("demand_generators", ENV_ARGS_DEFAULT["demand_generators"])
        )
        # Empirical series are finite; keep contiguous windows without padding
        # or repeating observations to meet the synthetic-data training length.
        from harl.envs.multi_lt_transship.generator_vec import ShanshuGenerator
        generators = [generator, *getattr(generator, "generators", [])]
        lengths = [len(sales) for gen in generators
                   if isinstance(gen, ShanshuGenerator) for sales in gen.sales_list]
        train_length = min([shape[2], *lengths])
        if train_length < 2:
            raise ValueError("Forecast training requires at least 2 observations per agent")
        if train_length < shape[2]:
            warnings.warn(
                f"forecast_train_length={shape[2]} exceeds available demand history; "
                f"using {train_length} contiguous observations per training episode.",
                RuntimeWarning, stacklevel=2,
            )
            for gen in generators:
                gen.shape = (gen.shape[0], gen.shape[1], train_length)
                gen.T = train_length
                if isinstance(gen, ShanshuGenerator):


                    gen.sales_list = [sales[:min(lengths)] for sales in gen.sales_list]
        return generator.generate(round_int=True, clip_nonneg=True)
    finally:
        np.random.set_state(state)


def baseline_forecasts(demand, args, method):
    """Zero-initialized forecasts, separate from environment/RL observations."""
    config = {**BASELINE_FORECAST_DEFAULTS, **args}
    zeros = np.zeros(demand.shape[1], dtype=np.float32)
    training = make_training_demand(config, demand.shape[1]) if method != "ema" else None
    return forecast_demand(
        demand, zeros, zeros, method, training_demand=training,
        alpha=float(config["alpha"]),
        forecast_variance_alpha=float(config["alpha"]),
        arima_order=config["arima_order"],
        xgboost_lags=config["xgboost_lags"],
        xgboost_n_estimators=config["xgboost_n_estimators"],
        xgboost_max_depth=config["xgboost_max_depth"],
        xgboost_learning_rate=config["xgboost_learning_rate"],
        random_seed=config["forecast_random_seed"],
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _as_agent_vector(value: np.ndarray, num_agents: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, num_agents)
    if array.size != num_agents:
        raise ValueError(f"{name} must be scalar or have {num_agents} values")
    return array


def _ema_predictions(
    demand: np.ndarray, initial_mean: np.ndarray, alpha: float
) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
    num_envs, num_agents, horizon = demand.shape
    predictions = np.empty((num_envs, num_agents, horizon + 1), dtype=np.float32)
    predictions[:, :, 0] = np.broadcast_to(initial_mean, (num_envs, num_agents))
    one_minus_alpha = 1.0 - alpha
    for t in range(1, horizon + 1):
        predictions[:, :, t] = (
            alpha * demand[:, :, t - 1]
            + one_minus_alpha * predictions[:, :, t - 1]
        )
    return predictions


def _forecast_error_std(
    demand: np.ndarray,
    predictions: np.ndarray,
    initial_std: np.ndarray,
    variance_alpha: float,
) -> np.ndarray:
    """Use the original baseline's updated-mean variance recurrence."""
    if not 0.0 < variance_alpha <= 1.0:
        raise ValueError(
            f"forecast_variance_alpha must be in (0, 1], got {variance_alpha}"
        )
    num_envs, num_agents, horizon = demand.shape
    variance = np.empty((num_envs, num_agents, horizon + 1), dtype=np.float32)
    variance[:, :, 0] = np.broadcast_to(
        np.square(initial_std), (num_envs, num_agents)
    )
    for t in range(1, horizon + 1):
        residual = demand[:, :, t - 1] - predictions[:, :, t]
        variance[:, :, t] = (
            variance_alpha * np.square(residual)
            + (1.0 - variance_alpha) * variance[:, :, t - 1]
        )
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def _arima_trend(order: Sequence[int]) -> str:
    return "c" if int(order[1]) == 0 else "t"


def _fit_arima_models(
    training_demand: np.ndarray, order: Tuple[int, int, int]
) -> Tuple[np.ndarray, ...]:
    cache_key = ("arima-model", order, _array_digest(training_demand))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError(
            "ARIMA forecasting requires statsmodels. Install project dependencies "
            "or run `python -m pip install statsmodels`."
        ) from exc

    _, num_agents, _ = training_demand.shape
    fitted_parameters = []
    for agent_idx in range(num_agents):
        # Pool trajectories with one missing observation between episodes.
        # This is an approximate panel fit: missing values do not reset states.
        panel = np.concatenate(
            [
                np.concatenate(
                    [training_demand[env_idx, agent_idx], np.array([np.nan])]
                )
                for env_idx in range(training_demand.shape[0])
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ARIMA(
                panel,
                order=order,
                trend=_arima_trend(order),
                enforce_stationarity=(order == (3, 1, 1)),
                enforce_invertibility=(order == (3, 1, 1)),
            ).fit(method_kwargs={"warn_convergence": False, "maxiter": 100})
        if not np.all(np.isfinite(result.params)) or not result.mle_retvals.get("converged", False):
            raise RuntimeError(f"ARIMA{order} fit failed for agent {agent_idx}: {result.mle_retvals}")
        fitted_parameters.append(np.asarray(result.params, dtype=np.float64))
    fitted = tuple(fitted_parameters)
    _MODEL_CACHE[cache_key] = fitted
    return fitted


def _arima_predictions(
    demand: np.ndarray,
    training_demand: np.ndarray,
    order: Tuple[int, int, int],
) -> np.ndarray:
    if order == (3, 1, 1):
        parameters = _fit_arima_models(training_demand, order)
        return _arima311_zero_history(demand, parameters)
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError(
            "ARIMA forecasting requires statsmodels. Install project dependencies "
            "or run `python -m pip install statsmodels`."
        ) from exc

    parameters = _fit_arima_models(training_demand, order)
    horizon = demand.shape[2]
    predictions = np.zeros((*demand.shape[:2], horizon + 1), dtype=np.float32)
    for env_idx in range(demand.shape[0]):
        for agent_idx in range(demand.shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(
                    demand[env_idx, agent_idx],
                    order=order,
                    trend=_arima_trend(order),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.filter(parameters[agent_idx])
                one_step = np.asarray(
                    result.get_prediction().predicted_mean, dtype=np.float32
                )
            predictions[env_idx, agent_idx, 1:horizon] = np.maximum(
                one_step[1:horizon], 0.0
            )
            predictions[env_idx, agent_idx, horizon] = max(
                0.0, float(result.forecast(1)[0])
            )
    return predictions


def _arima311_zero_history(demand, parameters):
    """Conditional ARIMA(3,1,1), with missing centered differences/errors zero.

    The pre-series demand level and first forecast are zero. Innovations use
    the unclipped model prediction; nonnegative clipping affects output only.
    Parameters are statsmodels' drift, ar.L1..L3, ma.L1, sigma2.
    """
    horizon = demand.shape[2]
    predictions = np.zeros((*demand.shape[:2], horizon + 1), dtype=np.float32)
    for agent, params in enumerate(parameters):
        drift, phi1, phi2, phi3, theta, _ = params
        ar_history = np.zeros((demand.shape[0], 3), dtype=np.float64)
        previous_level = np.zeros(demand.shape[0], dtype=np.float64)
        raw_prediction = np.zeros(demand.shape[0], dtype=np.float64)
        for t in range(1, horizon + 1):
            observed = demand[:, agent, t - 1].astype(np.float64)
            innovation = observed - raw_prediction
            centered_difference = observed - previous_level - drift
            ar_history[:, 1:] = ar_history[:, :-1].copy()
            ar_history[:, 0] = centered_difference
            raw_prediction = (observed + drift
                              + ar_history @ np.array([phi1, phi2, phi3])
                              + theta * innovation)
            predictions[:, agent, t] = np.maximum(raw_prediction, 0.0)
            previous_level = observed
    return predictions


def _xgb_feature(history: np.ndarray, t: int, lags: int) -> np.ndarray:
    available = min(t, lags)
    tail = np.zeros(lags, dtype=np.float32)
    if available:
        tail[-available:] = history[t - available:t]
    recent = np.asarray(history[max(0, t - 15) : t], dtype=np.float32)
    return np.concatenate(
        [
            tail,
            np.array(
                [np.asarray(history[max(0, t - 5):t], dtype=np.float32).sum() / 5.0,
                 recent.std() if recent.size else 0.0,
                 tail[-1] - tail[-2] if t >= 2 else 0.0],
                dtype=np.float32,
            ),
        ]
    )


def _fit_xgboost_models(
    training_demand: np.ndarray,
    lags: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_seed: int,
) -> Tuple[object, ...]:
    cache_key = (
        "xgboost-model",
        lags,
        n_estimators,
        max_depth,
        learning_rate,
        random_seed,
        _array_digest(training_demand),
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost forecasting requires xgboost. Install project dependencies "
            "or run `python -m pip install xgboost`."
        ) from exc

    models = []
    for agent_idx in range(training_demand.shape[1]):
        features = []
        targets = []
        for env_idx in range(training_demand.shape[0]):
            series = training_demand[env_idx, agent_idx]
            for t in range(1, series.shape[0]):
                features.append(_xgb_feature(series, t, lags))
                targets.append(series[t])
        model = XGBRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            subsample=0.8,
            colsample_bytree=0.9,
            min_child_weight=10,
            reg_lambda=10,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=int(random_seed),
        )
        model.fit(np.asarray(features, dtype=np.float32), np.asarray(targets))
        models.append(model)
    fitted = tuple(models)
    _MODEL_CACHE[cache_key] = fitted
    return fitted


def _xgboost_predictions(
    demand: np.ndarray,
    training_demand: np.ndarray,
    lags: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_seed: int,
) -> np.ndarray:
    if lags < 2:
        raise ValueError("xgboost_lags must be at least 2")
    models = _fit_xgboost_models(
        training_demand,
        lags,
        n_estimators,
        max_depth,
        learning_rate,
        random_seed,
    )
    horizon = demand.shape[2]
    predictions = np.zeros((*demand.shape[:2], horizon + 1), dtype=np.float32)
    for env_idx in range(demand.shape[0]):
        for agent_idx in range(demand.shape[1]):
            series = demand[env_idx, agent_idx]
            for t in range(1, horizon + 1):
                prediction = models[agent_idx].predict(
                    _xgb_feature(series, t, lags).reshape(1, -1)
                )[0]
                predictions[env_idx, agent_idx, t] = max(
                    0.0, float(prediction)
                )
    return predictions


def forecast_demand(
    demand: np.ndarray,
    initial_mean: np.ndarray,
    initial_std: np.ndarray,
    method: str,
    *,
    training_demand: np.ndarray = None,
    alpha: float = 0.2,
    forecast_variance_alpha: float = 0.2,
    arima_order: Iterable[int] = (3, 1, 1),
    xgboost_lags: int = 8,
    xgboost_n_estimators: int = 120,
    xgboost_max_depth: int = 2,
    xgboost_learning_rate: float = 0.035,
    random_seed: int = 20260904,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return causal one-step mean and error-standard-deviation forecasts.

    Args:
        demand: Realized demand with shape ``(environments, agents, horizon)``.
        initial_mean: EMA forecast before the first observation; learned methods start at zero.
        initial_std: Error standard deviation available before the first observation.
        method: ``ema``, ``arima``, or ``xgboost``.
        training_demand: Independent trajectories used to fit ARIMA parameters or
            XGBoost trees.  It is required for both learned methods.
    """
    demand = np.asarray(demand, dtype=np.float32)
    if demand.ndim != 3:
        raise ValueError("demand must have shape (environments, agents, horizon)")
    num_agents = demand.shape[1]
    initial_mean = _as_agent_vector(initial_mean, num_agents, "initial_mean")
    initial_std = _as_agent_vector(initial_std, num_agents, "initial_std")
    method = str(method).lower()
    if method not in SUPPORTED_FORECAST_METHODS:
        raise ValueError(
            f"Unsupported demand forecast method {method!r}; "
            f"choose one of {SUPPORTED_FORECAST_METHODS}"
        )

    order = tuple(int(value) for value in arima_order)
    if len(order) != 3:
        raise ValueError("arima_order must contain exactly three integers")
    training_digest = None
    if method in {"arima", "xgboost"}:
        if training_demand is None:
            raise ValueError(f"training_demand is required for {method}")
        training_demand = np.asarray(training_demand, dtype=np.float32)
        if training_demand.ndim != 3 or training_demand.shape[1] != num_agents:
            raise ValueError(
                "training_demand must have shape (training environments, agents, horizon)"
            )
        training_digest = _array_digest(training_demand)

    config_key = json.dumps(
        {
            "method": method,
            "alpha": alpha,
            "variance_alpha": forecast_variance_alpha,
            "arima_order": order,
            "xgb_lags": xgboost_lags,
            "xgb_estimators": xgboost_n_estimators,
            "xgb_depth": xgboost_max_depth,
            "xgb_lr": xgboost_learning_rate,
            "seed": random_seed,
        },
        sort_keys=True,
    )
    cache_key = (
        config_key,
        _array_digest(demand),
        _array_digest(initial_mean),
        _array_digest(initial_std),
        training_digest,
    )
    cached = _FORECAST_CACHE.get(cache_key)
    if cached is not None:
        return cached[0].copy(), cached[1].copy()

    if method == "ema":
        predictions = _ema_predictions(demand, initial_mean, float(alpha))
    elif method == "arima":
        predictions = _arima_predictions(
            demand,
            training_demand,
            order,
        )
    else:
        predictions = _xgboost_predictions(
            demand,
            training_demand,
            int(xgboost_lags),
            int(xgboost_n_estimators),
            int(xgboost_max_depth),
            float(xgboost_learning_rate),
            int(random_seed),
        )

    error_std = _forecast_error_std(
        demand,
        predictions,
        initial_std,
        float(forecast_variance_alpha),
    )
    predictions = predictions.astype(np.float32)
    _FORECAST_CACHE[cache_key] = (predictions.copy(), error_std.copy())
    return predictions, error_std
