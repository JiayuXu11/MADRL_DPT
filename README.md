# MARLIM

Core implementation of the RL inventory/transshipment policies and heuristic benchmarks, with fixed experiment configurations and the demand, lead-time, shipping-loss, and distance data needed to run them.

## Installation

Run commands from this repository's root. Python 3.8.18 is the reference interpreter. Install PyTorch for your CPU or CUDA environment, then install the project:

```bash
python -m pip install -e .
```

The implementation uses NumPy, SciPy, Gym spaces, PyTorch, statsmodels, and XGBoost. `requirements-reference.txt` records the installed package versions used for verification. Heuristic allocation uses SciPy's linear-programming solver. CUDA is enabled in the RL configurations; PyTorch falls back to CPU when CUDA is unavailable. For an explicit CPU run, pass `--cuda False` to `train.py`.

## Experiment suites

Each JSON file under `configs/` contains a complete experiment configuration. `configs/suites.json` lists the scenarios in execution order.

| Suite | Scenarios | RL runs (seeds 1–5) | Heuristic runs |
| --- | ---: | ---: | ---: |
| `backlog` | 24 | 120 | 216 |
| `backlog_additional` | 21 | 105 | 189 |
| `lost` | 27 | 135 | 324 |
| `beverage` | 8 | 40 | 72 |
| `ririshun` | 8 | 40 | 72 |

The main backlog grid varies retailer count, lead time, freight cost, backlog penalty, holding cost, and minimum order quantity. The additional backlog grid uses 200-period episodes and uniform shipping loss on [0, 0.2]. Its lead-time cases have means 5, 8, 11, and 15. The lost-sales grid also varies selling price, with ordering cost 3 and fixed ordering cost 0. All grids vary one factor at a time around their configured default.

Beverage supplier: `SKU006`, `SKU019`, `SKU022`, `SKU023`, `SKU029`, `SKU045`, `SKU046`, `SKU062`.

RiRiShun Logistics: `1936c558`, `26f47370`, `3b6fda08`, `3e0092db`, `a4830d20`, `b01ff11b`, `b0bdce95`, `bf67f87f`.

## RL commands

```bash
bash scripts/rl/backlog.sh
bash scripts/rl/backlog_additional.sh
bash scripts/rl/lost.sh
bash scripts/rl/beverage.sh
bash scripts/rl/ririshun.sh
```

Each script directly lists the training commands for every scenario and seeds 1–5. The scripts use Bash and can be run on Linux or with Git Bash on Windows. Activate the Python environment before running them. A single command can also be run directly, for example:

```bash
python train.py --load_config configs/beverage/SKU006.json --exp_name beverage_SKU006 --seed 1
```

Training defaults are 20,000 transitions per retailer per iteration and up to 20,000 training iterations. `train.py` derives the number of parallel environments from episode length and the parameter-sharing setting. Checkpoints are saved when validation reward improves. Early stopping restores the best validation checkpoint before testing; a run that reaches the iteration budget tests its final policy. Training logs, saved configurations, checkpoints, and evaluation/test outputs are written under `results/`.

## Heuristic commands

```bash
bash scripts/heuristic/backlog.sh
bash scripts/heuristic/backlog_additional.sh
bash scripts/heuristic/lost.sh
bash scripts/heuristic/beverage.sh
bash scripts/heuristic/ririshun.sh
```

Each suite evaluates EMA, ARIMA, and XGBoost with Qtl (`cal_quantile_k`), Search (`search`), and Eppen (`eppen`). Lost sales additionally evaluates capped base-stock. Synthetic EMA uses weight 0.3. Real-data EMA uses weight 0.9 for SKU023, 0.6 for a4830d20, 0.3 for 1936c558, and 0.1 for the remaining included SKUs. ARIMA and XGBoost use residual-variance smoothing weight 0.3. ARIMA order is [3, 1, 1]; XGBoost uses 8 lags, 120 trees, depth 2, and learning rate 0.035. The environment's observation EMA remains the weight recorded in the environment configuration (0.2).

Heuristics use quantity-valued actions, 20 configured parallel environments, forced participation in transshipment, and Anup allocation. Real-data evaluation uses all available episode columns. These heuristic settings are applied independently of the RL action and allocation configuration.

ARIMA and XGBoost fit on training demand only. Forecast training uses 30 trajectories of up to 1,000 periods; real-data histories cap this length at the available contiguous training history. Search and capped base-stock select policy parameters on the evaluation split and then evaluate the selected policy on the test split. These evaluation steps are part of the benchmark implementation.

For real-data Eppen, the fixed Pearson correlation matrix is estimated from aligned `gen_source` training sequences, and pooled variance is computed as sigma-transpose × R × sigma. Synthetic Eppen sums independent marginal demand variances. This distinction is recorded by `eppen_use_demand_correlation` in every configuration.

The shell scripts list every forecast/strategy combination explicitly. A single heuristic command can be run directly, for example:

```bash
python test_baseline_vec.py --load_config configs/ririshun/1936c558.json --exp_name ririshun_1936c558__ema__eppen --run_function eppen --estimate_demand_method ema --alpha 0.3 --eppen_use_demand_correlation True
```

Heuristic outputs are written under `results_heu/`, with a row per completed run in `baseline_summary.csv`. Shell scripts stop if a command fails.

## Data and evaluation

The environment reads CSV files under `harl/envs/multi_lt_transship/`:

- `gen_source/SKU/AGENT.csv`: one training demand sequence per retailer.
- `eval_data/DATASET/agent_INDEX/`: evaluation `demand.csv`, `LT.csv`, and `shippingloss.csv`.
- `test_data/DATASET/agent_INDEX/`: test files with the same layout.
- `distance/`: ordered distance matrices used by the environment.

Evaluation and test demand files store one episode per column. Real-data episodes are overlapping 50-period windows within each split; both RL and heuristics evaluate all available columns. Synthetic evaluations use the configured number of environments. The supplied files preserve the existing experiment splits and sequences. `data_manifest.json` records the SHA-256 checksum of every bundled runtime data file.

Costs are per retailer per period: negate the mean cumulative reward across retailers and divide by episode length. RL comparisons aggregate independently trained seeds; heuristic runs retain each forecast/strategy combination. A reported best-forecast value is the minimum over those forecast-specific test costs and should be labeled as such.

## Verification

```bash
python -m unittest discover -s tests -v
```

The checks validate data checksums and shapes, suite coverage, correlation settings, and the Eppen pooled-variance calculation. Full RL training takes substantially longer than these checks; numerical results depend on the installed libraries and hardware.

The RL infrastructure builds on the HARL framework by PKU-MARL: https://github.com/PKU-MARL/HARL.
