import unittest

import numpy as np

import interbank
import interbank_testclass


class NoReplacementCompactionTestCase(interbank_testclass.InterbankTest):
    def setUp(self):
        self.configureTest(N=4, T=5, seed=7)
        self.model.config.allow_replacement_of_bankrupted = False

    def test_replace_failed_banks_compacts_vectors_and_remaps_lenders(self):
        self.model.C = np.array([10.0, 11.0, 12.0, 13.0])
        self.model.D = np.array([20.0, 21.0, 22.0, 23.0])
        self.model.failed = np.array([0, 1, 0, 0], dtype=int)
        self.model.lenders = np.array([1, -1, 1, 2], dtype=int)

        self.model.replace_failed_banks()

        self.assertEqual(self.model.config.N, 3)
        np.testing.assert_allclose(self.model.C, np.array([10.0, 12.0, 13.0]))
        np.testing.assert_allclose(self.model.D, np.array([20.0, 22.0, 23.0]))
        np.testing.assert_array_equal(self.model.failed, np.array([0, 0, 0], dtype=int))
        np.testing.assert_array_equal(self.model.lenders, np.array([-1, -1, 1], dtype=int))


class NoReplacementRunStopTestCase(unittest.TestCase):
    def test_run_stops_when_two_banks_remain_and_exports_num_banks(self):
        model = interbank.Model(T=10, N=3, seed=3)
        model.config.allow_replacement_of_bankrupted = False
        model.log.interactive = False

        original_repayments = model.do_repayments

        def forced_single_failure_per_step():
            profits_paid = original_repayments()
            if model.config.N > 2:
                model.failed[0] = 1
            return profits_paid

        model.do_repayments = forced_single_failure_per_step

        result = model.run()

        self.assertEqual(len(result), 1)
        self.assertIn("Ninitial", result.columns)
        self.assertIn("num_banks", result.columns)
        self.assertEqual(result["Ninitial"].iloc[0], 3)
        self.assertEqual(result["num_banks"].iloc[0], 2)


if __name__ == "__main__":
    unittest.main()
