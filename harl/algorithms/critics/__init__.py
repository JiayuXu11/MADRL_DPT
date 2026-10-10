"""Critics used by the reproduction experiments."""
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.soft_twin_continuous_q_critic import SoftTwinContinuousQCritic

CRITIC_REGISTRY = {"happo": VCritic, "hasac": SoftTwinContinuousQCritic}
