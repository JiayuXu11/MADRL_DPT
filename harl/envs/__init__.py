from absl import flags
from harl.envs.multi_lt_transship.multi_lt_transship_logger import MultiLtTransshipLogger

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "multi_lt_transship": MultiLtTransshipLogger,
    "multi_lt_transship_mechanism": MultiLtTransshipLogger,
}
