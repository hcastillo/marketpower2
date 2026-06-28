#!/usr/bin/env python
# coding: utf-8
"""
Experimento minimo (suavizado): variar p entre 0.012 y 0.121 con MC mayor,
sin el punto p=0.001 que introduce mucho ruido en el regimen desconectado.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp_runner
from interbank_lenderchange import LenderChange


class MinPRunSmooth(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 100

    ALGORITHM = LenderChange
    OUTPUT_DIRECTORY = "/experiments/2306/exp_min_p_0_01_smooth1"

    parameters = {
        "p": np.linspace(0.011, 0.121, num=12),
    }

    config = {}

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2026

    XTICKS_SCALED = True
    NAME_OF_X_SERIES = "p"


if __name__ == "__main__":
    runner = exp_runner.Runner(MinPRunSmooth)
    runner.do()
