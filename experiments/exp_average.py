#!/usr/bin/env python
# coding: utf-8

import argparse
import concurrent.futures
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import interbank


OUTPUT_DIRECTORY = ROOT / "output" / "experiment_average_robust"
SUMMARY_FILENAME = "average_summary.txt"
GDT_FILENAME_TEMPLATE = "run_{index:03d}.gdt"

NUM_RUNS = 50
SEEDS = list(range(1, 11))

N = 50
T = 1000


def run_one_simulation(run_index, seed, n_banks, horizon, output_directory):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    model = interbank.Model(T=horizon, N=n_banks, seed=seed)
    model.log.interactive = False
    model.stats.define_output_directory(str(output_path))
    model.stats.define_output_format("gdt")
    model.stats.define_output_file(GDT_FILENAME_TEMPLATE.format(index=run_index))
    dataframe = model.run()

    numeric_dataframe = dataframe.apply(pd.to_numeric, errors="coerce")
    stats_mean = {
        key: float(value)
        for key, value in numeric_dataframe.mean(skipna=True).items()
        if not pd.isna(value)
    }

    current_correlation = {}
    for label, result in model.stats.correlation:
        if result is None:
            current_correlation[label] = {"r": None, "p": None}
        else:
            r_value = result.statistic if hasattr(result, "statistic") else result[0]
            p_value = result.pvalue if hasattr(result, "pvalue") else result[1]
            current_correlation[label] = {"r": float(r_value), "p": float(p_value)}

    return stats_mean, current_correlation


def build_correlation_summary(correlation_runs):
    summary = {}
    for run_values in correlation_runs:
        for label, values in run_values.items():
            if label not in summary:
                summary[label] = {"r": [], "p": []}
            if values["r"] is not None:
                summary[label]["r"].append(values["r"])
            if values["p"] is not None:
                summary[label]["p"].append(values["p"])
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run average experiment with multiprocessing")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of executions")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--clean", action="store_true", help="Remove previous run_*.gdt and summary before running")
    return parser.parse_args()


def clean_output_directory(output_directory):
    for gdt_file in output_directory.glob("run_*.gdt"):
        gdt_file.unlink()
    summary_file = output_directory / SUMMARY_FILENAME
    if summary_file.exists():
        summary_file.unlink()


def main():
    args = parse_args()
    num_runs = max(1, int(args.runs))
    workers = args.workers

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_output_directory(OUTPUT_DIRECTORY)

    statistic_means_per_run = []
    correlation_per_run = []

    jobs = []
    for run_index in range(num_runs):
        seed = SEEDS[run_index % len(SEEDS)]
        jobs.append((run_index, seed, N, T, str(OUTPUT_DIRECTORY)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one_simulation, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            stats_mean, correlation_info = future.result()
            statistic_means_per_run.append(stats_mean)
            correlation_per_run.append(correlation_info)

    stats_means_dataframe = pd.DataFrame(statistic_means_per_run)
    average_of_statistics = stats_means_dataframe.mean(skipna=True)

    correlation_summary = build_correlation_summary(correlation_per_run)

    summary_path = OUTPUT_DIRECTORY / SUMMARY_FILENAME
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("experiment_average\n")
        summary_file.write(f"runs={num_runs}\n")
        summary_file.write(f"N={N}\n")
        summary_file.write(f"T={T}\n")
        summary_file.write(f"seeds={SEEDS}\n\n")

        summary_file.write("Average correlations across runs\n")
        for label in sorted(correlation_summary.keys()):
            r_values = correlation_summary[label]["r"]
            p_values = correlation_summary[label]["p"]
            mean_r = float(np.mean(r_values)) if r_values else float("nan")
            mean_p = float(np.mean(p_values)) if p_values else float("nan")
            summary_file.write(
                f"- {label}: mean_r={mean_r:.6f} mean_p={mean_p:.6f} valid_runs={len(r_values)}\n"
            )

        summary_file.write("\nAverage of statistics (mean over each run, then mean over runs)\n")
        for stat_name in sorted(average_of_statistics.index):
            value = average_of_statistics[stat_name]
            if pd.isna(value):
                continue
            summary_file.write(f"- {stat_name}: {float(value):.6f}\n")

    print(f"Saved GDT outputs and summary in: {OUTPUT_DIRECTORY}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
