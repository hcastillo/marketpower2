#!/usr/bin/env python
# coding: utf-8
"""
Compara psi para distintos MC (100, 200, 300, 400, 500) con el mismo seed,
para ver a partir de qué MC la curva deja de tener valores atípicos.

Cada MC se ejecuta en /experiments/2606/exp_mc/mc{N}/
Al final copia psi.png → psi_mc{N}.png en la carpeta raíz del experimento.

Uso:
    python experiments/exp_mc.py --do
    python experiments/exp_mc.py --plot --directory /experiments/2606/exp_mc/mc100   (etc.)
"""
from pathlib import Path
import sys
import os
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp_runner
from interbank_lenderchange import LenderChange


class ExpMC(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 100

    ALGORITHM = LenderChange
    OUTPUT_DIRECTORY = "/experiments/2606/exp_mc"

    parameters = {
        "p": np.linspace(0.012, 0.122, num=12),
    }

    config = {}

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 2025

    XTICKS_SCALED = True
    NAME_OF_X_SERIES = "p"


if __name__ == "__main__":
    runner = exp_runner.Runner(ExpMC)
    args = runner.parser.parse_args()
    if args.listnames:
        ExpMC().listnames()
    elif args.do:
        base_dir = "/experiments/2606/exp_mc"
        for mc in [100, 200, 300, 400, 500]:
            mc_dir = f"{base_dir}/mc{mc}"
            print(f"\n{'='*60}\nMC={mc}\n{'='*60}")
            exp = ExpMC()
            exp.MC = mc
            exp.OUTPUT_DIRECTORY = mc_dir
            exp.error_bar = args.errorbar
            exp.plot_removing_first = args.plot_removing_first
            exp.save_graph = set()
            if args.graph.lower() == 'all':
                exp.save_graph = 'all'
            elif args.graph:
                exp.save_graph = set(int(x) for x in args.graph.split(','))
            exp.clear_results()
            exp.do(clear_previous_results=args.clear, reverse_execution=args.reverse)
            psi_src = os.path.join(mc_dir, "psi.png")
            psi_dst = os.path.join(base_dir, f"psi_mc{mc}.png")
            if os.path.exists(psi_src):
                shutil.copy2(psi_src, psi_dst)
                print(f"  -> {psi_dst}")
    elif args.plot:
        directory = args.directory if args.directory else ExpMC.OUTPUT_DIRECTORY
        runner.plot_only(ExpMC(), directory)
    elif args.report:
        directory = args.directory if args.directory else ExpMC.OUTPUT_DIRECTORY
        runner.generate_report(directory)
    else:
        runner.parser.print_help()
