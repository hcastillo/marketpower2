import unittest

import interbank


class ValidateModelCorrelationsTestCase(unittest.TestCase):
    def test_default_execution_has_positive_key_correlations(self):
        model = interbank.Model()
        model.log.interactive = False
        model.run()

        correlation_map = {label: result for label, result in model.stats.correlation}

        for label in ("psi", "prob_bankruptcy", "bankruptcies"):
            self.assertIn(label, correlation_map)
            result = correlation_map[label]
            self.assertIsNotNone(result)
            statistic = result.statistic if hasattr(result, "statistic") else result[0]
            self.assertGreater(statistic, 0, f"Expected positive ir/{label} correlation, got {statistic}")


if __name__ == "__main__":
    unittest.main()
