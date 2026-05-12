import unittest
from unittest.mock import Mock

import numpy as np

import interbank


class BalanceIdentityOnDebugBanksTestCase(unittest.TestCase):
    @staticmethod
    def _nan_to_zero(value):
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return 0.0
        return float(value)

    def test_balances_hold_on_every_debug_banks_call(self):
        model = interbank.Model(T=10, N=5, seed=1)
        model.log.interactive = False
        model.log.define_log(log='DEBUG')

        def assert_balances_side_effect():
            for i in range(model.config.N):
                lhs = model.C[i] + model.L[i] + model.R[i] + model.s[i] + model.loaned[i]
                rhs = model.D[i] + model.E[i] + model.rationing[i] + model.d[i] + model.d2[i]
                self.assertAlmostEqual(
                    lhs,
                    rhs,
                    places=6,
                    msg=f"Balance mismatch at t={model.t}, bank={i}: {model.log.format_number(model.C[i])}C"
                            f"+{model.log.format_number(model.L[i])}L+{model.log.format_number(model.R[i])}R"
                            f"+{model.log.format_number(model.s[i])}s != {model.log.format_number(model.D[i])}D"
                            f"+{model.log.format_number(model.E[i])}E+{model.log.format_number(model.rationing[i])}rat" 
                            f"+{model.log.format_number(model.d[i])}d+{model.log.format_number(model.d2[i])}d2",
                )

        model.log.debug_banks = Mock(side_effect=assert_balances_side_effect)

        model.run()

        self.assertGreater(model.log.debug_banks.call_count, 0)


if __name__ == "__main__":
    unittest.main()
