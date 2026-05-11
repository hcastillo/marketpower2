import os
import tempfile
import unittest

import numpy as np

import interbank
from exp_runner import ExperimentRun

# interbank_statistics.py
# En determine_cross_correlation():
# convierto ir y la otra serie a numpy float.
# recorto al mínimo tamaño común.
# aplico máscara np.isfinite(ir) & np.isfinite(other).
# calculo correlación solo con esos puntos válidos.
# si quedan menos de 2 puntos válidos => n/a.
# Corregí también el texto n/a para que diga correl ir/... (consistente con tu cambio de base).
# Nuevo test: tests/test_equity_series_shape.py
# Ejecuta un modelo real con T=1000, N=50, seed=2025.
# Guarda en .gdt temporal.
# Lee el .gdt con ExperimentRun.read_gdt.
# Sobre equity, valida:
# longitud esperada (1000),
# casi todos valores finitos (>=99%),
# variabilidad en diferencias (std(diff) > 0.25),
# alternancia suficiente de subidas/bajadas (sign_change_ratio > 0.35),
# pocos pasos exactamente planos (near_zero_step_ratio < 0.04),
# sin mesetas largas (máximo run de mismo valor redondeado a 4 decimales <= 6).

class EquitySeriesShapeTestCase(unittest.TestCase):
    def test_equity_series_has_expected_volatility_pattern(self):
        model = interbank.Model(T=1000, N=50, seed=2025)
        model.log.interactive = False

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "equity_shape.gdt")
            model.stats.define_output_file(output_file)
            model.stats.define_output_format("gdt")
            model.run()

            dataframe, _ = ExperimentRun.read_gdt(output_file)

        self.assertIn("equity", dataframe.columns)
        equity = np.asarray(dataframe["equity"], dtype=float)
        self.assertEqual(len(equity), 1000)

        finite_mask = np.isfinite(equity)
        self.assertGreaterEqual(np.mean(finite_mask), 0.99)
        equity = equity[finite_mask]

        diffs = np.diff(equity)
        self.assertGreater(np.std(diffs), 0.25)

        sign_changes = np.count_nonzero(np.sign(diffs[1:]) != np.sign(diffs[:-1]))
        sign_change_ratio = sign_changes / max(len(diffs) - 1, 1)
        self.assertGreater(sign_change_ratio, 0.35)

        near_zero_step_ratio = np.mean(np.abs(diffs) < 1e-9)
        self.assertLess(near_zero_step_ratio, 0.04)

        rounded = np.round(equity, 4)
        max_plateau = 1
        current = 1
        for index in range(1, len(rounded)):
            if rounded[index] == rounded[index - 1]:
                current += 1
                if current > max_plateau:
                    max_plateau = current
            else:
                current = 1
        self.assertLessEqual(max_plateau, 6)


if __name__ == "__main__":
    unittest.main()
