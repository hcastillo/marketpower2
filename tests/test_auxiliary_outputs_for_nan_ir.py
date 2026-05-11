import os
import tempfile
import unittest

import pandas as pd

import interbank
from exp_runner import ExperimentRun


class AuxiliaryOutputsForNanIrTestCase(unittest.TestCase):
    def test_auxiliary_b_files_are_created_with_real_t_when_ir_has_nan(self):
        model = interbank.Model(T=120, N=50, seed=5)
        model.log.interactive = False

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "aux_check.gdt")
            model.stats.define_output_file(output_file)
            model.stats.define_output_format("both")
            model.run()

            path_main_gdt = output_file
            path_aux_gdt = os.path.join(temp_dir, "aux_check_b.gdt")
            path_main_csv = os.path.join(temp_dir, "aux_check.csv")
            path_aux_csv = os.path.join(temp_dir, "aux_check_b.csv")

            self.assertTrue(os.path.exists(path_main_gdt))
            self.assertTrue(os.path.exists(path_aux_gdt))
            self.assertTrue(os.path.exists(path_main_csv))
            self.assertTrue(os.path.exists(path_aux_csv))

            dataframe_aux_gdt, _ = ExperimentRun.read_gdt(path_aux_gdt)
            self.assertIn("real_t", dataframe_aux_gdt.columns)

            dataframe_aux_csv = pd.read_csv(path_aux_csv, comment="#", delimiter=";")
            self.assertIn("real_t", dataframe_aux_csv.columns)


if __name__ == "__main__":
    unittest.main()
