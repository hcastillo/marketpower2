#!/usr/bin/env python
# coding: utf-8
r"""
Runs exp_min_p_0_01 with MC=500, using the custom exp_runner1.py
which only discards outliers based on num_loans, loans, and ir.
Saves results in C:\experiments\0207\exp_min_p_0_01_mc500
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp_runner1
from interbank_lenderchange import LenderChange


class ExpMC500_0207(exp_runner1.ExperimentRun):
    N = 50
    T = 1000
    MC = 500
    ALGORITHM = LenderChange
    parameters = {"p": np.r_[np.linspace(0.012, 0.122, num=12), 0.5, 1] }
    config = { "omega": 0.6 }
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0
    SEED_FOR_EXECUTION = 2027
    XTICKS_SCALED = True
    NAME_OF_X_SERIES = "p"


if __name__ == "__main__":
    exp = ExpMC500_0207()
    exp.OUTPUT_DIRECTORY = "C:/experiments/0707/exp_min_p_0_01_mc500_omega06"
    exp.error_bar = False
    exp.plot_removing_first = True
    exp.save_graph = set()
    exp.clear_results()
    exp.do(clear_previous_results=False, reverse_execution=False)
