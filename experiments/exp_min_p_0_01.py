#!/usr/bin/env python
# coding: utf-8
"""
Experimento minimo: variar p entre 0 y 1 con paso 0.1.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp_runner
from interbank_lenderchange import LenderChange


class MinPRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 10

    ALGORITHM = LenderChange
    OUTPUT_DIRECTORY = "/experiments/exp_min_p_0_01"

    parameters = {
        "p":  np.linspace(0.00001, 0.12, num=12), # [0.01, 0.05, 0.07, 0.0075, 0.08, 0.09, 0.35, 0.5, 0.9] # np.linspace(0.00001, 1, num=10), #
        
    }

    config = { }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025
    
    XTICKS_SCALED = True
    NAME_OF_X_SERIES = "p"


if __name__ == "__main__":
    runner = exp_runner.Runner(MinPRun)
    runner.do()
