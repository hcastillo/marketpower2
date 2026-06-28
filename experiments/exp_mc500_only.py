#!/usr/bin/env python
# coding: utf-8
"""Process only exp_min_p_0_01_mc500."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp_runner
from interbank_lenderchange import LenderChange


class ExpMC500(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 500
    ALGORITHM = LenderChange
    parameters = {"p": np.linspace(0.012, 0.122, num=12)}
    config = {}
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0
    SEED_FOR_EXECUTION = 2027
    XTICKS_SCALED = True
    NAME_OF_X_SERIES = "p"


if __name__ == "__main__":
    exp = ExpMC500()
    exp.OUTPUT_DIRECTORY = "/experiments/2606/exp_min_p_0_01_mc500"
    exp.error_bar = False
    exp.plot_removing_first = True
    exp.save_graph = set()
    exp.clear_results()
    exp.do(clear_previous_results=False, reverse_execution=False)
