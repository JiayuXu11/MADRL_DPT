"""Runner registry."""

from harl.runners.on_policy_ha_runner_mechanism import OnMixPolicyHaRunner

RUNNER_REGISTRY = {
    "mix": OnMixPolicyHaRunner,
}
