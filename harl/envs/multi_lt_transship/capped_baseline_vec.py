"""Capped base-stock extension; the existing baseline classes stay unchanged.

Xin (2021), Operations Research 69(1):61-70, doi:10.1287/opre.2020.2019.
The paper uses q=min((S-IP)^+, r). Here S retains the existing forecast,
rounding and MOQ conventions, while r is an order cap selected on evaluation data.
"""

import numpy as np

from harl.envs.multi_lt_transship.baseline_vec import VecBaseline


class VecCappedBaseStock(VecBaseline):
    def __init__(self, args):
        super().__init__(args)
        self.cap_ratio = args.get('capped_cap_ratio', None)
        self.cap_basis = args.get('capped_cap_basis', 'reference')
        self.cap_binding_count = 0
        self.cap_decision_count = 0
        self.cap_positive_order_count = 0
        self.actual_order_cap = None
        if self.cap_basis not in ('reference', 'forecast'):
            raise ValueError('capped_cap_basis must be reference or forecast')
        if self.cap_ratio is not None:
            self.cap_ratio = float(self.cap_ratio)
            if not np.isfinite(self.cap_ratio) or self.cap_ratio < 0:
                raise ValueError('capped_cap_ratio must be finite and nonnegative, or None')

    def get_demand_est(self, env):
        mean, variance = super().get_demand_est(env)
        self._cap_demand_mean = mean
        return mean, variance

    def get_order_action(self, env, obs):
        # Exactly one forecast update per decision, including the uncapped case.
        order = super().get_order_action(env, obs)
        self.cap_decision_count += order.size
        self.cap_positive_order_count += int(np.count_nonzero(order > 0))
        if self.cap_ratio is None:
            return order
        mean = (env.demand_mean_const if self.cap_basis == 'reference'
                else self._cap_demand_mean)
        cap = np.floor(self.cap_ratio * np.maximum(mean, 0))
        if self.cap_basis == 'reference':
            self.actual_order_cap = np.asarray(cap).tolist()
        self.cap_binding_count += int(np.count_nonzero(order > cap))
        capped = np.minimum(order, cap)
        # A positive order must satisfy MOQ AND the cap. If impossible, order 0.
        capped = np.where(capped >= env.MOQ, capped, 0)
        return capped.astype(np.int32)
