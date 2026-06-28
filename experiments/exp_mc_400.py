#!/usr/bin/env python
# coding: utf-8
"""Run MC=400 for exp_mc."""
from pathlib import Path
import sys, os, shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp_runner
from interbank_lenderchange import LenderChange


class E(exp_runner.ExperimentRun):
    N = 50; T = 1000; MC = 400
    ALGORITHM = LenderChange
    parameters = {"p": np.linspace(0.012, 0.122, num=12)}
    config = {}
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0
    SEED_FOR_EXECUTION = 2025
    XTICKS_SCALED = True
    NAME_OF_X_SERIES = "p"


if __name__ == "__main__":
    mc = 400
    mc_dir = f"/experiments/2606/exp_mc/mc{mc}"
    print(f"MC={mc}", flush=True)
    e = E()
    e.OUTPUT_DIRECTORY = mc_dir
    e.error_bar = False
    e.plot_removing_first = False
    e.save_graph = set()
    e.clear_results()
    e.do(clear_previous_results=False, reverse_execution=False)
    base_dir = "/experiments/2606/exp_mc"
    psi_src = os.path.join(mc_dir, "psi.png")
    psi_dst = os.path.join(base_dir, f"psi_mc{mc}.png")
    if os.path.exists(psi_src):
        shutil.copy2(psi_src, psi_dst)
        print(f"  -> {psi_dst}", flush=True)
    print(f"MC={mc} done", flush=True)
