#!/usr/bin/env python
# coding: utf-8
"""
Resume missing p0122 runs for exp_min_p_0_01_smooth_mc500.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_min_p_0_01_smooth_mc500 import MinPRunSmoothMC500

experiment = MinPRunSmoothMC500()
seeds = experiment.generate_random_seeds_for_this_execution()

# p0122 is the 12th (last) parameter point -> offset 11 * MC = 5500
param_index = 11
mc = experiment.MC
offset = param_index * mc

missing = []
import os
output_dir = experiment.OUTPUT_DIRECTORY
filename_for_iteration = "p0122"
for i in range(mc):
    if not os.path.exists(f"{output_dir}/{filename_for_iteration}_{i}.gdt"):
        missing.append(i)

print(f"Missing runs for p0122: {len(missing)}")
if not missing:
    print("Nothing to do.")
    sys.exit(0)

model_parameters = {"p": 0.122}
model_configuration = {}
for idx, i in enumerate(missing):
    seed = seeds[offset + i]
    print(f"[{idx+1}/{len(missing)}] p0122 run {i} seed {seed}")
    experiment.run_model(
        f"{output_dir}/{filename_for_iteration}_{i}",
        model_configuration,
        model_parameters,
        seed,
    )
print("Done.")
