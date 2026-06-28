#!/usr/bin/env python
# coding: utf-8
"""
Recreates the three base experiments (no _smooth) in /experiments/2606/.
Each variant reuses existing individual .gdt files if present.
"""
from pathlib import Path
import sys
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp_runner
from interbank_lenderchange import LenderChange


class ExpMinP(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 100
    ALGORITHM = LenderChange
    parameters = {"p": np.linspace(0.012, 0.122, num=12)}
    config = {}
    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0
    SEED_FOR_EXECUTION = 2025
    XTICKS_SCALED = True
    NAME_OF_X_SERIES = "p"


if __name__ == "__main__":
    runner = exp_runner.Runner(ExpMinP)
    args = runner.parser.parse_args()

    variants = [
        ("exp_min_p_0_01",      100, 2025, np.linspace(0.012, 0.122, 12)),
        ("exp_min_p_0_01b",     100, 2026, np.linspace(0.011, 0.121, 12)),
        ("exp_min_p_0_01_mc500", 500, 2027, np.linspace(0.012, 0.122, 12)),
    ]

    if args.listnames:
        print("Variant: exp_min_p_0_01")
        ExpMinP().listnames()
        print("Variant: exp_min_p_0_01b")
        e = ExpMinP()
        e.parameters = {"p": np.linspace(0.011, 0.121, 12)}
        e.listnames()
        print("Variant: exp_min_p_0_01_mc500")
        ExpMinP().listnames()
    elif args.plot:
        for name, mc, seed, p_values in variants:
            print(f"\n{'='*60}\nPlotting {name} (MC={mc})\n{'='*60}")
            exp = ExpMinP()
            exp.MC = mc
            exp.SEED_FOR_EXECUTION = seed
            exp.parameters = {"p": p_values}
            exp.plot_removing_first = args.plot_removing_first
            runner.plot_only(exp, f"/experiments/2606/{name}")
    elif args.do:
        for name, mc, seed, p_values in variants:
            print(f"\n{'='*60}\nRe-processing {name} (MC={mc})\n{'='*60}")
            exp = ExpMinP()
            exp.MC = mc
            exp.SEED_FOR_EXECUTION = seed
            exp.parameters = {"p": p_values}
            exp.OUTPUT_DIRECTORY = f"/experiments/2606/{name}"
            exp.error_bar = args.errorbar
            exp.plot_removing_first = args.plot_removing_first
            if args.graph.lower() == 'all':
                exp.save_graph = 'all'
            elif args.graph:
                exp.save_graph = set(int(x) for x in args.graph.split(','))
            else:
                exp.save_graph = set()
            exp.clear_results()
            exp.do(clear_previous_results=args.clear, reverse_execution=args.reverse)
    elif args.report:
        for name, mc, seed, p_values in variants:
            runner.generate_report(f"/experiments/2606/{name}")
    else:
        runner.parser.print_help()
