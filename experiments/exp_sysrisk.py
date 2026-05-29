#!/usr/bin/env python
# coding: utf-8
"""
Systemic Risk Experiment:
- Vary p above 0.08 to analyze systemic risk at high connectivity
- Compute kurtosis of equity and Hill exponent of equity losses
- Monitor distance to default (equity buffer per bank)
- Test: high connectivity -> heavier tails (higher kurtosis, lower Hill exponent)
"""
import concurrent.futures
import os
import time
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.stats
from progress.bar import Bar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp_runner
from interbank_lenderchange import LenderChange


class SystemicRiskRun(exp_runner.ExperimentRun):
    N = 50
    T = 1000
    MC = 20

    ALGORITHM = LenderChange
    OUTPUT_DIRECTORY = "/experiments/exp_sysrisk001"

    parameters = {
        "p":  np.linspace(0.00001, 0.1200001, num=12), # np.linspace(0.08, 1.0, num=20),
    }

    config = {"robust_ir": False}

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 0

    SEED_FOR_EXECUTION = 25

    XTICKS_SCALED = True
    NAME_OF_X_SERIES = "p"

    @staticmethod
    def hill_estimator(data, k=None):
        sorted_data = np.sort(data.astype(float))[::-1]
        positive = sorted_data > 0
        n_pos = np.sum(positive)
        if n_pos < 2:
            return np.nan
        valid = sorted_data[positive]
        if k is None:
            k = max(2, int(np.sqrt(n_pos)))
        k = min(k, n_pos - 1)
        try:
            log_data = np.log(valid[:k])
            return (1.0 / k) * np.sum(log_data - log_data[-1])
        except Exception:
            return np.nan

    @staticmethod
    def excess_kurtosis(data):
        numeric = pd.to_numeric(data, errors="coerce").dropna()
        if len(numeric) < 4:
            return np.nan
        return scipy.stats.kurtosis(numeric, fisher=True)

    def do(self, clear_previous_results=False, reverse_execution=False):
        self.log_replaced_data = ""
        initial_time = time.perf_counter()
        if clear_previous_results:
            results_to_plot = {}
            results_x_axis = []
        else:
            results_to_plot, results_x_axis = self.load(f"{self.OUTPUT_DIRECTORY}/")
        if not results_to_plot:
            self.verify_directories()
            seeds_for_random = self.generate_random_seeds_for_this_execution()
            progress_bar = Bar("Executing models", max=self.get_num_models())
            progress_bar.update()
            correlation_file = open(f"{self.OUTPUT_DIRECTORY}/results.txt", "w", encoding="utf-8")
            montecarlo_iteration_perfect_correlations = {}
            position_inside_seeds_for_random = 0
            array_of_configs = self.get_models(self.config)
            if reverse_execution:
                array_of_configs = reversed(list(array_of_configs))
            for model_configuration in array_of_configs:
                array_of_parameters = self.get_models(self.parameters)
                if reverse_execution:
                    array_of_parameters = reversed(list(array_of_parameters))
                for model_parameters in array_of_parameters:
                    result_iteration_to_check = pd.DataFrame()
                    filename_for_iteration = self.get_filename_for_iteration(
                        model_parameters, model_configuration
                    )
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        results_mc = {
                            executor.submit(
                                self.load_or_execute_model,
                                model_configuration,
                                model_parameters,
                                filename_for_iteration,
                                i,
                                clear_previous_results,
                                seeds_for_random[
                                    i + position_inside_seeds_for_random
                                ],
                            ): i
                            for i in range(self.MC)
                        }
                        for future in concurrent.futures.as_completed(results_mc):
                            result_iteration_to_check = pd.concat(
                                [result_iteration_to_check, future.result()]
                            )

                    result_iteration = pd.DataFrame()
                    position_inside_seeds_for_random -= self.MC
                    montecarlo_iteration_perfect_correlation = True
                    correlation_coefficient = p_value = 0
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        results_mc = {
                            executor.submit(
                                self.load_model_and_rerun_till_ok,
                                model_configuration,
                                model_parameters,
                                filename_for_iteration,
                                i,
                                clear_previous_results,
                                seeds_for_random,
                                position_inside_seeds_for_random,
                                result_iteration_to_check,
                            ): i
                            for i in range(self.MC)
                        }

                        for i, future in enumerate(
                            concurrent.futures.as_completed(results_mc)
                        ):
                            result_mc = future.result()
                            if "bankruptcies" in result_mc and not (
                                np.all(result_mc["bankruptcies"] == 0)
                                or np.all(
                                    result_mc["bankruptcies"]
                                    == result_mc["bankruptcies"].iloc[0]
                                )
                                or np.all(result_mc["ir"] == 0)
                                or np.all(
                                    result_mc["ir"] == result_mc["ir"].iloc[0]
                                )
                            ):
                                with warnings.catch_warnings():
                                    warnings.simplefilter(
                                        "ignore",
                                        scipy.stats.NearConstantInputWarning,
                                    )
                                    correlation_coefficient, p_value = (
                                        scipy.stats.pearsonr(
                                            result_mc["ir"],
                                            result_mc["bankruptcies"],
                                        )
                                    )
                                    (
                                        correlation_coefficient1,
                                        p_value1,
                                    ) = scipy.stats.pearsonr(
                                        result_mc["ir"][1:],
                                        result_mc["bankruptcies"][:-1],
                                    )
                                    correlation_file.write(
                                        f"{filename_for_iteration}_{i} = {model_configuration} {model_parameters}\n"
                                    )
                                    correlation_file.write(
                                        exp_runner.format_correlation_values(
                                            0, correlation_coefficient, p_value
                                        )
                                    )
                                    correlation_file.write(
                                        exp_runner.format_correlation_values(
                                            1, correlation_coefficient1, p_value1
                                        )
                                    )
                                    montecarlo_iteration_perfect_correlation = (
                                        montecarlo_iteration_perfect_correlation
                                        and (
                                            (
                                                correlation_coefficient1 > 0
                                                and p_value1 <= 0.10
                                            )
                                            or (
                                                correlation_coefficient > 0
                                                and p_value <= 0.10
                                            )
                                        )
                                    )
                            result_iteration = pd.concat(
                                [result_iteration, result_mc]
                            )

                    if montecarlo_iteration_perfect_correlation:
                        montecarlo_iteration_perfect_correlations[
                            str(model_configuration)
                            + " "
                            + str(model_parameters)
                        ] = exp_runner.format_correlation_values(
                            1, correlation_coefficient, p_value
                        )

                    # --- standard per-column mean/std ---
                    for k in result_iteration.keys():
                        if k.strip() == "t":
                            continue
                        numeric_result = pd.to_numeric(
                            result_iteration[k], errors="coerce"
                        )
                        mean_estimated = numeric_result.mean()
                        std_estimated = numeric_result.std()
                        if k in results_to_plot:
                            results_to_plot[k].append(
                                [mean_estimated, std_estimated]
                            )
                        else:
                            results_to_plot[k] = [
                                [mean_estimated, std_estimated]
                            ]
                        if k in ("bankruptcies", "ir", "ir_weighted"):
                            results_to_plot.setdefault(k + "_max_line", []).append(
                                [numeric_result.max(), 0]
                            )
                            results_to_plot.setdefault(k + "_min_line", []).append(
                                [numeric_result.min(), 0]
                            )

                    # --- kurtosis of equity distribution ---
                    kurt = self.excess_kurtosis(
                        result_iteration["equity"]
                    )
                    results_to_plot.setdefault(
                        "kurtosis_equity", []
                    ).append([kurt, 0])

                    # --- Hill exponent of equity LOSSES (negative equity changes) ---
                    eq = pd.to_numeric(result_iteration["equity"], errors="coerce").values
                    eq_mat = eq.reshape(self.MC, -1)
                    diffs = np.diff(eq_mat, axis=1).ravel()
                    losses = -diffs[diffs < 0]
                    hill_h = self.hill_estimator(losses)
                    hill_alpha = 1.0 / hill_h if hill_h and not np.isnan(hill_h) else np.nan
                    results_to_plot.setdefault(
                        "hill_equity", []
                    ).append([hill_alpha, 0])

                    # --- kurtosis of bad_debt distribution ---
                    kurt_bd = self.excess_kurtosis(
                        result_iteration["bad_debt"]
                    )
                    results_to_plot.setdefault(
                        "kurtosis_bad_debt", []
                    ).append([kurt_bd, 0])

                    # --- Hill exponent of bad_debt (loss magnitudes) ---
                    bd = pd.to_numeric(result_iteration["bad_debt"], errors="coerce").values
                    bd_pos = bd[bd > 0]
                    hill_h_bd = self.hill_estimator(bd_pos)
                    hill_alpha_bd = 1.0 / hill_h_bd if hill_h_bd and not np.isnan(hill_h_bd) else np.nan
                    results_to_plot.setdefault(
                        "hill_bad_debt", []
                    ).append([hill_alpha_bd, 0])

                    # --- distance to default (mean capital ratio E/(D+E)) ---
                    if "equity" in result_iteration and "deposits" in result_iteration:
                        equity_s = pd.to_numeric(result_iteration["equity"], errors="coerce")
                        deposits_s = pd.to_numeric(result_iteration["deposits"], errors="coerce")
                        ratio = equity_s / (deposits_s + equity_s)
                        results_to_plot.setdefault("dist_to_default", []).append([ratio.mean(), 0])
                    else:
                        results_to_plot.setdefault("dist_to_default", []).append([np.nan, 0])

                    results_x_axis.append(
                        self.get_title_for(
                            model_configuration, model_parameters
                        )
                    )
                    position_inside_seeds_for_random += self.MC
                    progress_bar.next()

            progress_bar.finish()
            if montecarlo_iteration_perfect_correlations:
                for perfect_correlation in montecarlo_iteration_perfect_correlations:
                    correlation_file.write(
                        f"{perfect_correlation} : {montecarlo_iteration_perfect_correlations[perfect_correlation]}\n"
                    )
            correlation_file.close()
            print(
                f"Saving results in {self.OUTPUT_DIRECTORY}/results.{self.OUTPUT_FORMAT}..."
            )
            if self.OUTPUT_FORMAT.lower() == "csv":
                self.save_csv(
                    results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/"
                )
            else:
                self.save_gdt(
                    results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/"
                )
        else:
            print(f"Loaded data from previous work from {self.OUTPUT_DIRECTORY}")
        results_comparing, results_comparing2 = self.load_comparing(results_x_axis)
        if self.log_replaced_data:
            print(self.log_replaced_data)
        print("Plotting...")
        self.plot(
            results_to_plot,
            results_x_axis,
            self.get_title_for(self.config, self.parameters),
            f"{self.OUTPUT_DIRECTORY}/",
            results_comparing,
            results_comparing2,
        )
        self.results_to_plot = results_to_plot
        final_time = time.perf_counter()
        print("execution_time: %2.5f secs" % (final_time - initial_time))
        return results_to_plot, results_x_axis


if __name__ == "__main__":
    runner = exp_runner.Runner(SystemicRiskRun)
    experiment = runner.do()
