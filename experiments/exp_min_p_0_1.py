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
    MC = 30

    ALGORITHM = LenderChange
    OUTPUT_DIRECTORY = "/experiments/exp_min_p_0_1"

    parameters = {
        "p":   np.linspace(0.00001, 1, num=10), #
        
    }

    config = { }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 25
    
    XTICKS_SCALED = True
    NAME_OF_X_SERIES = "p"


if __name__ == "__main__":
    runner = exp_runner.Runner(MinPRun)
    runner.do()
