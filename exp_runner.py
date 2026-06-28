#!/usr/bin/env python
# coding: utf-8
"""
Executor base class for the interbank model.
Adapted to the root interbank.py API.
"""
import argparse
import concurrent.futures
import gzip
import os
import random
import time
import warnings
from itertools import product

import lxml.builder
import lxml.etree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from progress.bar import Bar

import interbank
import interbank_lenderchange
from interbank import Model


def format_correlation_values(delay, correlation_coefficient, p_value):
    result = f"\tt={delay} "
    result += f"pearson={correlation_coefficient} p_value={p_value}"
    result += " !!!\n" if p_value < 0.1 and correlation_coefficient > 0 else "\n"
    return result


class ExperimentRun:
    N = 1
    T = 1
    MC = 1

    LIMIT_OUTLIER = 6
    MAX_EXECUTIONS_OF_MODELS_OUTLIERS = 10

    STYLE = "-"
    MARKER = "s"
    COLOR = "black"
    MARKER_COLOR = "red"

    COMPARING_DATA = ""
    COMPARING_LABEL = "Comparing"
    COMPARING_STYLE = "--"
    COMPARING_COLOR = "black"
    COMPARING_MARKER = "o"
    COMPARING_MARKER_COLOR = "red"

    COMPARING_DATA2 = ""
    COMPARING_LABEL2 = "Comparing2"
    COMPARING_STYLE2 = ":"
    COMPARING_COLOR2 = "black"
    COMPARING_MARKER2 = "D"
    COMPARING_MARKER2_COLOR = "red"

    XTICKS_DIVISOR = 1
    XTICKS_SCALED = False

    LABEL = "Invalid"
    OUTPUT_DIRECTORY = "Invalid"

    ALGORITHM = interbank_lenderchange.LenderChange

    DESCRIPTION_TITLE = ""

    NAME_OF_X_SERIES = None
    NAME_OF_Y_SERIES = None

    EXTRA_MODEL_CONFIGURATION = {}

    ALLOW_REPLACEMENT_OF_BANKRUPTED = True

    OUTPUT_FORMAT = "gdt"

    config = {}

    SEED_FOR_EXECUTION = 2025

    parameters = {
        "p": np.linspace(0.001, 0.100, num=40),
    }

    LENGTH_FILENAME_PARAMETER = 5
    LENGTH_FILENAME_CONFIG = 1

    log_replaced_data = ""

    error_bar = False

    @staticmethod
    def _transform_line_from_string(line_with_values):
        items = []
        for item in line_with_values.replace("  ", " ").strip().split(" "):
            if item == "":
                continue
            try:
                items.append(int(item))
            except ValueError:
                items.append(float(item))
        return items

    @staticmethod
    def read_gdt(filename):
        configuration = []
        with gzip.open(filename, "rb") as file_obj:
            tree = lxml.etree.parse(file_obj)
        root = tree.getroot()
        children = root.getchildren()
        values = []
        columns = []
        columns_to_remove = []
        result = pd.DataFrame()
        if len(children) == 3:
            for variable in children[1].getchildren():
                column_name = variable.values()[0].strip()
                if column_name == "leverage_" or column_name == "std_leverage_":
                    column_name = "leverage" if column_name == "leverage_" else "std_leverage"
                columns.append(column_name)
                if "parent" in variable.keys():
                    columns_to_remove.append(column_name)
                if len(variable.values()) == 2 and configuration == []:
                    configuration = variable.values()[1].replace(os.path.basename(__file__) + " ", "").split(" ")
            for value in children[2].getchildren():
                values.append(ExperimentRun._transform_line_from_string(value.text))

        if columns and values:
            result = pd.DataFrame(columns=columns, data=values)
            for column_to_remove in columns_to_remove:
                del result[column_to_remove]
        return result, configuration

    def plot(self, array_with_data, array_with_x_values, title_x, directory,
             array_comparing=None, array_comparing2=None):
        if getattr(self, 'plot_removing_first', False) and len(array_with_x_values) > 1:
            array_with_x_values = array_with_x_values[1:]
            array_with_data = {k: v[1:] for k, v in array_with_data.items()}
        plot_x_values = []
        for j in range(len(array_with_x_values)):
            plot_x_values.append(array_with_x_values[j] if (j % ExperimentRun.XTICKS_DIVISOR == 0) else " ")
        if plot_x_values:
            plot_x_values[-1] = array_with_x_values[-1]
        if self.XTICKS_SCALED and self.NAME_OF_X_SERIES:
            try:
                prefix = self.NAME_OF_X_SERIES + "="
                x_scaled = []
                for x in array_with_x_values:
                    idx = x.rfind(prefix)
                    if idx >= 0:
                        remainder = x[idx + len(prefix):].split()[0]
                        x_scaled.append(float(remainder))
                    else:
                        x_scaled.append(float(x.split("=")[-1].split()[0]))
                array_with_x_values = x_scaled
                plot_x_values = [f"{v:g}" for v in x_scaled]
            except Exception:
                pass
        for i in array_with_data:
            i = i.strip()
            if i in ["t", "psi.1", "cross_psi_ir_lenders", "cross_psi_ir"] or i.endswith("_max_line") or i.endswith("_min_line"):
                continue
            mean = []
            deviation_error = []
            mean_comparing = []
            mean_comparing2 = []
            for j in range(len(array_with_data[i])):
                mean.append(array_with_data[i][j][0])
                deviation_error.append(array_with_data[i][j][1] / 2)
                if array_comparing and i in array_comparing and j < len(array_comparing[i]):
                    mean_comparing.append(array_comparing[i][j][0])
                if array_comparing2 and i in array_comparing2 and j < len(array_comparing2[i]):
                    mean_comparing2.append(array_comparing2[i][j][0])

            plt.clf()
            fig, ax = plt.subplots()
            title = f"{i}"
            if not self.DESCRIPTION_TITLE:
                if self.NAME_OF_X_SERIES and self.XTICKS_SCALED:
                    title += f" vs {self.NAME_OF_X_SERIES} MC={self.MC}"
                else:
                    title += f" x={title_x} MC={self.MC}"
            else:
                title += f" {self.DESCRIPTION_TITLE}"

            if self.error_bar:
                ax.errorbar(array_with_x_values, mean, yerr=deviation_error,
                            linestyle=self.STYLE, marker=self.MARKER, color=self.COLOR,
                            markerfacecolor=self.MARKER_COLOR, markeredgecolor=self.MARKER_COLOR,
                            ecolor='gray', capsize=2,
                            label=self.NAME_OF_X_SERIES if self.NAME_OF_X_SERIES
                            else self.ALGORITHM.__name__ if array_comparing else "")
            else:
                ax.plot(array_with_x_values, mean,
                        linestyle=self.STYLE, marker=self.MARKER, color=self.COLOR,
                        markerfacecolor=self.MARKER_COLOR, markeredgecolor=self.MARKER_COLOR,
                        label=self.NAME_OF_X_SERIES if self.NAME_OF_X_SERIES
                        else self.ALGORITHM.__name__ if array_comparing else "")
            logarithm_plot = False
            if array_comparing and i in array_comparing:
                if len(mean_comparing) == 1:
                    ax.plot(0, mean_comparing, linestyle=self.COMPARING_STYLE,
                            marker=self.COMPARING_MARKER, color=self.COMPARING_COLOR,
                            markerfacecolor=self.COMPARING_MARKER_COLOR, markeredgecolor=self.COMPARING_MARKER_COLOR,
                            label=self.COMPARING_LABEL)
                else:
                    ax.plot(array_with_x_values, mean_comparing, linestyle=self.COMPARING_STYLE,
                            marker=self.COMPARING_MARKER, color=self.COMPARING_COLOR,
                            markerfacecolor=self.COMPARING_MARKER_COLOR, markeredgecolor=self.COMPARING_MARKER_COLOR,
                            label=self.COMPARING_LABEL)
                if abs(mean[0] - mean_comparing[0]) > 1e6 and abs(mean[-1] - mean_comparing[-1]) > 1e6:
                    ax.set_yscale("log")
                    logarithm_plot = True
                if array_comparing2 and i in array_comparing2:
                    if len(mean_comparing2) == 1:
                        ax.plot(0, mean_comparing2, linestyle=self.COMPARING_STYLE2,
                                marker=self.COMPARING_MARKER2, color=self.COMPARING_COLOR2,
                                markerfacecolor=self.COMPARING_MARKER2_COLOR, markeredgecolor=self.COMPARING_MARKER2_COLOR,
                                label=self.COMPARING_LABEL2)
                    else:
                        ax.plot(array_with_x_values, mean_comparing2, linestyle=self.COMPARING_STYLE2,
                                marker=self.COMPARING_MARKER2, color=self.COMPARING_COLOR2,
                                markerfacecolor=self.COMPARING_MARKER2_COLOR, markeredgecolor=self.COMPARING_MARKER2_COLOR,
                                label=self.COMPARING_LABEL2)
                    if abs(mean[0] - mean_comparing2[0]) > 1e6 and abs(mean[-1] - mean_comparing2[-1]) > 1e6:
                        ax.set_yscale("log")
                        logarithm_plot = True

            max_line_key = i + "_max_line"
            min_line_key = i + "_min_line"
            if max_line_key in array_with_data or min_line_key in array_with_data:
                ir_series = i.startswith("ir")
                max_color = "lightblue" if ir_series else "red"
                min_color = "lightblue" if ir_series else "green"
                if max_line_key in array_with_data:
                    max_line_vals = [array_with_data[max_line_key][j][0] for j in range(len(array_with_data[max_line_key]))]
                    ax.plot(array_with_x_values, max_line_vals, color=max_color, linestyle="--", linewidth=1, label="max")
                if min_line_key in array_with_data:
                    min_line_vals = [array_with_data[min_line_key][j][0] for j in range(len(array_with_data[min_line_key]))]
                    ax.plot(array_with_x_values, min_line_vals, color=min_color, linestyle="--", linewidth=1, label="min")

            plt.title(title + (" (log)" if logarithm_plot else ""))
            if self.XTICKS_SCALED and self.NAME_OF_X_SERIES:
                pass
            else:
                ax.set_xticks(array_with_x_values)
                ax.set_xticklabels(plot_x_values, rotation=270, fontsize=5)
            if array_comparing or max_line_key in array_with_data or min_line_key in array_with_data:
                plt.legend(loc="best")
            ax.set_xlabel(title_x, fontsize=6)
            ax.set_ylabel(i, fontsize=6)
            plt.savefig(f"{directory}{i}.png", dpi=300)
            plt.close(fig)
            with open(f"{directory}{i}.txt", "w", encoding="utf-8") as file_obj:
                file_obj.write(f"{' ':16}{self.NAME_OF_X_SERIES if self.NAME_OF_X_SERIES else i:15}(std_err) ")
                if array_comparing and i in array_comparing:
                    file_obj.write(f"{self.COMPARING_LABEL:15}")
                if array_comparing2 and i in array_comparing2:
                    file_obj.write(f"{self.COMPARING_LABEL2:15}")
                file_obj.write("\n")
                for index_x, x in enumerate(array_with_x_values):
                    file_obj.write(f"{x:15}{mean[index_x]:15.10f} ({deviation_error[index_x]:7.4f})")
                    if array_comparing and i in array_comparing:
                        file_obj.write(f"{mean_comparing[index_x]:15.10f}")
                    if array_comparing2 and i in array_comparing2:
                        file_obj.write(f"{mean_comparing2[index_x]:15.10f}")
                    file_obj.write("\n")

            with open(f"{directory}{i}.tex", "w", encoding="utf-8") as tex:
                tex.write("\\begin{frame}{" + title.replace("_", "\\_") + "}\n")
                tex.write("  \\begin{columns}[T]\n")
                tex.write("    \\begin{column}{0.75\\textwidth}\n")
                tex.write("      \\begin{figure}\n")
                tex.write("        \\centering\n")
                tex.write("        \\includegraphics[width=\\textwidth, trim=0 0 0 38, clip]{" + i + ".png}\n")
                tex.write("      \\end{figure}\n")
                tex.write("    \\end{column}\n")
                tex.write("    \\begin{column}{0.23\\textwidth}\n")
                tex.write("      \\centering\n")
                tex.write("      \\tiny\\renewcommand{\\arraystretch}{0.85}\\setlength{\\tabcolsep}{3pt}\n")
                tex.write("      \\begin{tabular}{l")
                if array_comparing and i in array_comparing:
                    tex.write("r" * (2 + (1 if array_comparing2 and i in array_comparing2 else 0)))
                else:
                    tex.write("r")
                tex.write("}\n")
                tex.write("        \\toprule\n")

                param_name = next(iter(self.parameters)) if isinstance(self.parameters, dict) else self.NAME_OF_X_SERIES
                if not param_name:
                    param_name = "$p$"
                tex.write(f"        {param_name} & {i.replace('_', '\\_')}")
                if array_comparing and i in array_comparing:
                    tex.write(f" & {self.COMPARING_LABEL}")
                if array_comparing2 and i in array_comparing2:
                    tex.write(f" & {self.COMPARING_LABEL2}")
                tex.write(" \\\\\n")
                tex.write("        \\midrule\n")

                x_vals_numeric = []
                for x in array_with_x_values:
                    try:
                        x_str = f"{float(x):.3f}"
                    except ValueError:
                        x_str = str(x)
                    x_vals_numeric.append(x_str)

                for index_x in range(len(array_with_x_values)):
                    tex.write(f"        {x_vals_numeric[index_x]} & {mean[index_x]:.4f}")
                    if array_comparing and i in array_comparing:
                        tex.write(f" & {mean_comparing[index_x]:.4f}")
                    if array_comparing2 and i in array_comparing2:
                        tex.write(f" & {mean_comparing2[index_x]:.4f}")
                    tex.write(" \\\\\n")
                tex.write("        \\bottomrule\n")
                tex.write("      \\end{tabular}\n")
                tex.write("    \\end{column}\n")
                tex.write("  \\end{columns}\n")
                tex.write("\\end{frame}\n")

        if all(k in array_with_data for k in ("bankruptcies", "bankruptcy_rationed", "bankruptcy_contagion")):
            x = array_with_x_values
            rat = [array_with_data["bankruptcy_rationed"][j][0] for j in range(len(x))]
            cont = [array_with_data["bankruptcy_contagion"][j][0] for j in range(len(x))]
            total = [array_with_data["bankruptcies"][j][0] for j in range(len(x))]
            repay_fail = [total[j] - rat[j] - cont[j] for j in range(len(x))]

            plt.clf()
            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            ax.plot(x, rat, linestyle=self.STYLE, marker="s", color="black",
                    markerfacecolor="red", markeredgecolor="red", label="Rationing")
            ax2.plot(x, cont, linestyle=self.STYLE, marker="o", color="darkgray",
                     markerfacecolor="red", markeredgecolor="red", label="Contagion")
            ax.plot(x, repay_fail, linestyle=self.STYLE, marker="o", color="lightgray",
                    markerfacecolor="white", markeredgecolor="red", label="Repayment fail.")
            plt.title(f"Bankruptcies by channel vs {title_x} MC={self.MC}")
            ax.set_xlabel(title_x, fontsize=6)
            ax.set_ylabel("Rationing + Repayment fail.", fontsize=6)
            ax2.set_ylabel("Contagion", fontsize=6)
            if self.XTICKS_SCALED and self.NAME_OF_X_SERIES:
                pass
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(plot_x_values, rotation=270, fontsize=5)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
            plt.savefig(f"{directory}bankruptcies_all.png", dpi=300)
            plt.close(fig)

            param_name = next(iter(self.parameters)) if isinstance(self.parameters, dict) and self.parameters else self.NAME_OF_X_SERIES
            if not param_name:
                param_name = "$p$"
            param_name = str(param_name).replace("_", "\\_")

            x_vals_numeric = []
            for val in x:
                try:
                    x_vals_numeric.append(f"{float(val):.3f}")
                except ValueError:
                    x_vals_numeric.append(str(val))

            with open(f"{directory}bankruptcies_all.tex", "w", encoding="utf-8") as tex:
                tex.write("\\begin{frame}{Bankruptcies by channel}\n")
                tex.write("  \\begin{columns}[T]\n")
                tex.write("    \\begin{column}{0.75\\textwidth}\n")
                tex.write("      \\begin{figure}\n")
                tex.write("        \\centering\n")
                tex.write("        \\includegraphics[width=\\textwidth, trim=0 0 0 38, clip]{bankruptcies_all.png}\n")
                tex.write("      \\end{figure}\n")
                tex.write("    \\end{column}\n")
                tex.write("    \\begin{column}{0.23\\textwidth}\n")
                tex.write("      \\centering\n")
                tex.write("      \\tiny\n")
                tex.write("      \\renewcommand{\\arraystretch}{0.85}\\setlength{\\tabcolsep}{3pt}\n")
                tex.write("      \\textcolor{red}{$\\blacksquare$} Rationing (left)\n")
                tex.write("      \\textcolor{red}{$\\bullet$} Contagion (right)\\par\n")
                tex.write("      \\textcolor{red}{$\\circ$} Repayment fail. (left)\\par\n")
                tex.write("      \\vspace{0.15cm}\n")
                tex.write("      \\begin{tabular}{lrrr}\n")
                tex.write("        \\toprule\n")
                tex.write(f"        {param_name} & Rat. & Cont. & Fail. \\\\\n")
                tex.write("        \\midrule\n")
                for j in range(len(x)):
                    tex.write(f"        {x_vals_numeric[j]} & {rat[j]:.2f} & {cont[j]:.2f} & {repay_fail[j]:.2f} \\\\\n")
                tex.write("        \\bottomrule\n")
                tex.write("      \\end{tabular}\n")
                tex.write("    \\end{column}\n")
                tex.write("  \\end{columns}\n")
                tex.write("\\end{frame}\n")

    def load(self, directory):
        if os.path.exists(f"{directory}results.csv"):
            dataframe = pd.read_csv(f"{directory}results.csv", header=1, delimiter=";")
            array_with_data = {}
            array_with_x_values = []
            name_for_x_column = dataframe.columns[0]
            for i in dataframe.columns[1:]:
                if not i.startswith("std_"):
                    array_with_data[i] = []
            for i in array_with_data.keys():
                if i != "psi.1":
                    for j in range(len(dataframe[i])):
                        array_with_data[i].append([dataframe[i][j], dataframe[f"std_{i}"][j]])
            for j in dataframe[name_for_x_column]:
                array_with_x_values.append(f"{name_for_x_column}={j}")
            return array_with_data, array_with_x_values
        gdt_path = f"{directory}results.gdt"
        if os.path.exists(gdt_path):
            result, x_vals, _ = self._load_from_gdt(gdt_path)
            return result, x_vals
        return {}, []

    def _load_from_gdt(self, gdt_path):
        dataframe, config_list = self.read_gdt(gdt_path)
        if dataframe.empty:
            return {}, [], {}
        array_with_data = {}
        array_with_x_values = []
        columns = list(dataframe.columns)
        name_for_x_column = columns[0]
        for i in columns[1:]:
            if not i.startswith("std_"):
                array_with_data[i] = []
        for i in list(array_with_data.keys()):
            std_col = f"std_{i}"
            if std_col in columns:
                for j in range(len(dataframe[i])):
                    array_with_data[i].append([dataframe[i][j], dataframe[std_col][j]])
            if not array_with_data[i]:
                del array_with_data[i]
        try:
            import ast
            with gzip.open(gdt_path, "rb") as f:
                tree = lxml.etree.parse(f)
            root = tree.getroot()
            children = root.getchildren()
            if len(children) == 3:
                first_var = children[1].getchildren()[0]
                if len(first_var.values()) == 2:
                    label_text = first_var.values()[1]
                    parsed = ast.literal_eval(label_text)
                    if isinstance(parsed, list) and len(parsed) == len(dataframe):
                        array_with_x_values = parsed
        except Exception:
            pass
        if not array_with_x_values:
            for j in dataframe[name_for_x_column]:
                array_with_x_values.append(f"{name_for_x_column}={j}")
        metadata = {"name_of_x_series": name_for_x_column}
        for item in config_list:
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    metadata[k.lower()] = int(v)
                except ValueError:
                    try:
                        metadata[k.lower()] = float(v)
                    except ValueError:
                        metadata[k.lower()] = v
            else:
                metadata.setdefault("algorithm", item)
        return array_with_data, array_with_x_values, metadata

    def _data_keys(self, array_with_data):
        return [k for k in array_with_data
                if not k.endswith("_max_line") and not k.endswith("_min_line")]

    def save_csv(self, array_with_data, array_with_x_values, directory, filename="results.csv"):
        keys = self._data_keys(array_with_data)
        with open(f"{directory}{filename}", "w", encoding="utf-8") as file_obj:
            file_obj.write(f"# MC={self.MC} N={self.N} T={self.T} {self.ALGORITHM.__name__}\n")
            file_obj.write(array_with_x_values[0].split("=")[0])
            for j in keys:
                file_obj.write(f";{j};std_{j}")
            file_obj.write("\n")
            for i in range(len(array_with_x_values)):
                value_for_line = f"{array_with_x_values[i].split('=')[1]}"
                if " " in value_for_line:
                    value_for_line = value_for_line.split(" ")[0]
                file_obj.write(f"{value_for_line}")
                for j in keys:
                    file_obj.write(f";{array_with_data[j][i][0]};{array_with_data[j][i][1]}")
                file_obj.write("\n")

    def save_gdt(self, array_with_data, array_with_x_values, directory, filename="results.gdt"):
        keys = self._data_keys(array_with_data)
        element = lxml.builder.ElementMaker()
        gretl_data = element.gretldata
        description = element.description
        variables = element.variables
        variable = element.variable
        observations = element.observations
        obs = element.obs

        model = Model()
        description1 = str(array_with_x_values)
        for item_config in self.config:
            if hasattr(model.config, item_config):
                setattr(model.config, item_config, None)
        description2 = str(model.config) + str(self.config)

        variables_xml = variables(count=f"{2 * len(keys) + 1}")
        x_var_name = (self.NAME_OF_X_SERIES if self.NAME_OF_X_SERIES
                      else array_with_x_values[0].strip().split()[-1].split("=")[0])
        variables_xml.append(variable(name=x_var_name, label=f"{description1}"))
        first = True
        for j in keys:
            name = "leverage_" if j == "leverage" else j
            if first:
                variables_xml.append(variable(name=f"{name}", label=f"{description2}"))
            else:
                variables_xml.append(variable(name=f"{name}"))
            first = False
            variables_xml.append(variable(name=f"std_{name}"))

        observations_xml = observations(count=f"{len(array_with_x_values)}", labels="false")
        for i in range(len(array_with_x_values)):
            value_for_line = array_with_x_values[i].strip().split()[-1].split("=")[1]
            string_obs = f"{value_for_line}  "
            for j in keys:
                string_obs += f"{array_with_data[j][i][0]}  {array_with_data[j][i][1]}  "
            observations_xml.append(obs(string_obs))
        header_text = (
            f"MC={self.MC} N={self.N} T={self.T} "
            f"{self.ALGORITHM.__name__ if not self.NAME_OF_X_SERIES else self.NAME_OF_X_SERIES}"
        )
        gdt_result = gretl_data(
            description(header_text),
            variables_xml,
            observations_xml,
            version="1.4",
            name="prueba",
            frequency="special:1",
            startobs="1",
            endobs=f"{len(array_with_x_values)}",
            type="cross-section",
        )
        with gzip.open(f"{directory}{filename}", "wb") as output_file:
            output_file.write(
                b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n'
            )
            output_file.write(lxml.etree.tostring(gdt_result, pretty_print=True, encoding=str).encode("ascii"))

    def run_model(self, filename, execution_config, execution_parameters, seed_random):
        model = Model()

        config_values = {}
        merged = {}
        merged.update(execution_config)
        merged.update(execution_parameters)
        merged.update(self.EXTRA_MODEL_CONFIGURATION)
        for key, value in merged.items():
            if hasattr(model.config, key):
                config_values[key] = value

        config_values.update(
            {
                "T": self.T,
                "N": self.N,
                "allow_replacement_of_bankrupted": self.ALLOW_REPLACEMENT_OF_BANKRUPTED,
                "seed": seed_random,
            }
        )
        model.configure(**config_values)
        model.save_graph = getattr(self, 'save_graph', False)

        model.lenderchange = self.ALGORITHM(model)
        model.stats.define_output_format(self.OUTPUT_FORMAT)
        model.stats.define_output_file(filename)
        model.stats.generate_plots = False
        return model.run()

    def get_num_models(self):
        return len(list(self.get_models(self.parameters))) * len(list(self.get_models(self.config)))

    def get_models(self, parameters):
        normalized_values = []
        for value in parameters.values():
            if isinstance(value, (list, tuple, range, np.ndarray)):
                normalized_values.append(value)
            else:
                normalized_values.append([value])
        return (dict(zip(parameters.keys(), values)) for values in sorted(product(*normalized_values)))

    def __filename_clean(self, value, max_length):
        value = str(value).replace("np.float64(", "").replace("np.float(", "")
        for remove in "{}[]()',: .":
            value = value.replace(remove, "")
        if value.endswith(".0"):
            value = value[:-2]
            last_digit = len(value) - 1
            while last_digit > 0 and value[last_digit].isdigit():
                last_digit -= 1
            while len(value) <= max_length:
                value = value[: last_digit + 1] + "0" + value[last_digit + 1 :]
        else:
            value = value.replace(".", "")
            while len(value) <= max_length:
                value += "0"
            if len(value) > max_length:
                value = value[:max_length]
        return value

    def get_filename_for_iteration(self, parameters, config):
        return self.get_filename_for_parameters(parameters) + self.get_filename_for_config(config)

    def get_filename_for_config(self, config):
        return self.__filename_clean(config, self.LENGTH_FILENAME_CONFIG)

    def get_filename_for_parameters(self, parameters):
        return self.__filename_clean(parameters, self.LENGTH_FILENAME_PARAMETER)

    def verify_directories(self):
        if not os.path.isdir(self.OUTPUT_DIRECTORY):
            os.makedirs(self.OUTPUT_DIRECTORY, exist_ok=True)

    def listnames(self):
        num = 0
        for config in self.get_models(self.config):
            for parameter in self.get_models(self.parameters):
                print(self.get_filename_for_iteration(parameter, config))
                num += 1
        print("total: ", num)

    def __get_value_for(self, param):
        result = ""
        if param:
            for i in param.keys():
                if hasattr(param[i], "__len__"):
                    result += i + " "
                else:
                    result += i + "=" + str(param[i]) + " "
        return result

    def get_title_for(self, param1, param2):
        return (self.__get_value_for(param1) + " " + self.__get_value_for(param2)).strip()

    def load_comparing(self, results_x_axis):
        results_comparing = None
        results_comparing2 = None
        if self.COMPARING_DATA:
            results_comparing, results_x_comparing = self.load(f"{self.COMPARING_DATA}/")
            if len(results_x_comparing) not in (len(results_x_axis), 1):
                results_comparing = None
            else:
                print(f"Loaded data to compare from {self.COMPARING_DATA}")

            if self.COMPARING_DATA2:
                results_comparing2, results_x_comparing2 = self.load(f"{self.COMPARING_DATA2}/")
                if len(results_x_comparing2) not in (len(results_x_axis), 1):
                    results_comparing2 = None
                else:
                    print(f"Loaded data to compare from {self.COMPARING_DATA2}")

        return results_comparing, results_comparing2

    @staticmethod
    def _has_numeric_content(dataframe):
        if dataframe.empty:
            return False
        numeric_dataframe = dataframe.apply(pd.to_numeric, errors="coerce")
        return numeric_dataframe.notna().sum().sum() > 0

    @classmethod
    def _read_model_csv(cls, filename):
        read_options = (
            {"header": 2},
            {"header": 2, "delimiter": ";"},
            {"header": 2, "sep": None, "engine": "python"},
        )
        for options in read_options:
            try:
                dataframe = pd.read_csv(filename, **options)
            except Exception:
                continue
            if len(dataframe.columns) == 1 and ";" in str(dataframe.columns[0]):
                try:
                    dataframe = pd.read_csv(filename, header=2, delimiter=";")
                except Exception:
                    pass
            if cls._has_numeric_content(dataframe):
                return dataframe.apply(pd.to_numeric, errors="coerce")
        return pd.DataFrame()

    def data_seems_ok(self, individual_execution, array_all_data):
        for k in array_all_data.keys():
            if k.strip() == "t":
                continue
            mean_individual_execution = pd.to_numeric(individual_execution[k], errors="coerce").mean()
            means = []
            for i in range(self.MC):
                means.append(
                    pd.to_numeric(
                        array_all_data[k][i * self.T : (i * self.T) + (self.T - 1)],
                        errors="coerce",
                    ).mean()
                )
            q1 = np.percentile(means, 25)
            q3 = np.percentile(means, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.LIMIT_OUTLIER * iqr
            upper_bound = q3 + self.LIMIT_OUTLIER * iqr
            if (
                not np.isnan(mean_individual_execution)
                and not np.isnan(iqr)
                and not (lower_bound <= mean_individual_execution <= upper_bound)
                and not (lower_bound == upper_bound)
            ):
                return False
        return True

    def load_or_execute_model(self, model_configuration, model_parameters, filename_for_iteration,
                              i, clear_previous_results=False, seed_for_this_model=None, stats_market=False):
        filename_to_open = f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}"
        if stats_market:
            filename_to_open += "b"
        output_format = self.OUTPUT_FORMAT.lower()
        if output_format == "csv":
            filename_csv = f"{filename_to_open}.csv"
            if os.path.isfile(filename_csv) and not clear_previous_results:
                result_from_csv = self._read_model_csv(filename_csv)
                if not result_from_csv.empty:
                    return result_from_csv
                warnings.warn(
                    f"Ignoring invalid CSV '{filename_csv}' and using other available source.",
                    RuntimeWarning,
                )
        else:
            filename_gdt = f"{filename_to_open}.gdt"
            if os.path.isfile(filename_gdt) and not clear_previous_results:
                result_mc, _ = self.read_gdt(filename_gdt)
                return result_mc
        if seed_for_this_model is None:
            print(f"file_not_found {filename_to_open}")
            return pd.DataFrame()
        return self.run_model(f"{filename_to_open}", model_configuration, model_parameters, seed_for_this_model)

    def load_model_and_rerun_till_ok(self, model_configuration, model_parameters, filename_for_iteration,
                                     i, clear_previous_results, seeds_for_random,
                                     position_inside_seeds_for_random, result_iteration_to_check):
        result_mc = self.load_or_execute_model(
            model_configuration,
            model_parameters,
            filename_for_iteration,
            i,
            clear_previous_results,
            seeds_for_random[i + position_inside_seeds_for_random],
        )
        offset = 1
        while (
            not self.data_seems_ok(result_mc, result_iteration_to_check)
            and offset <= self.MAX_EXECUTIONS_OF_MODELS_OUTLIERS
        ):
            self.discard_execution_of_iteration(filename_for_iteration, i)
            result_mc = self.load_or_execute_model(
                model_configuration,
                model_parameters,
                filename_for_iteration,
                i,
                clear_previous_results,
                seeds_for_random[i + position_inside_seeds_for_random] + offset,
            )
            offset += 1
        return result_mc

    def _generate_boxplot(self, results_x_axis, directory):
        if not hasattr(self, '_boxplot_data') or 'equity_borrowers' not in self._boxplot_data:
            return
        data = self._boxplot_data['equity_borrowers']
        x_labels = []
        for x in results_x_axis:
            try:
                x_labels.append(float(x.split('=')[-1].strip()))
            except (ValueError, IndexError):
                x_labels.append(x)
        fig, ax = plt.subplots()
        bp = ax.boxplot(data, labels=x_labels, patch_artist=True, showmeans=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        title = f"Boxplot of equity_borrowers vs {self.NAME_OF_X_SERIES} MC={self.MC}"
        ax.set_title(title)
        ax.set_xlabel(self.NAME_OF_X_SERIES if self.NAME_OF_X_SERIES else 'Parameter')
        ax.set_ylabel('Equity of Borrowers')
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{directory}boxplot.png", dpi=300)
        plt.close(fig)
        with open(f"{directory}boxplot.txt", "w", encoding="utf-8") as f:
            f.write(f"Boxplot: equity_borrowers vs {self.NAME_OF_X_SERIES} MC={self.MC}\n")
            f.write(f"{'x':>10} {'median':>10} {'Q1':>10} {'Q3':>10} {'min':>10} {'max':>10} {'n':>6}\n")
            for i, values in enumerate(data):
                arr = np.array(values)
                median = np.median(arr)
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                mi = np.min(arr)
                mx = np.max(arr)
                f.write(f"{x_labels[i]:>10.4f} {median:>10.4f} {q1:>10.4f} {q3:>10.4f} {mi:>10.4f} {mx:>10.4f} {len(values):>6}\n")
            f.write("\nOutliers (beyond 1.5*IQR):\n")
            for i, values in enumerate(data):
                arr = np.array(values)
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = (arr < lower) | (arr > upper)
                outliers = arr[outlier_mask]
                if len(outliers) > 0:
                    outlier_indices = ', '.join(f'#{idx}' for idx in np.where(outlier_mask)[0])
                    f.write(f"  {self.NAME_OF_X_SERIES}={x_labels[i]:.4f}: {len(outliers)} outliers ({outlier_indices}): {', '.join(f'{v:.4f}' for v in outliers)}\n")
        param_name = self.NAME_OF_X_SERIES if self.NAME_OF_X_SERIES else "x"
        with open(f"{directory}boxplot.tex", "w", encoding="utf-8") as tex:
            tex.write("\\begin{frame}{" + title.replace("_", "\\_") + "}\n")
            tex.write("  \\begin{columns}[T]\n")
            tex.write("    \\begin{column}{0.75\\textwidth}\n")
            tex.write("      \\begin{figure}\n")
            tex.write("        \\centering\n")
            tex.write("        \\includegraphics[width=\\textwidth, trim=0 0 0 38, clip]{boxplot.png}\n")
            tex.write("      \\end{figure}\n")
            tex.write("    \\end{column}\n")
            tex.write("    \\begin{column}{0.23\\textwidth}\n")
            tex.write("      \\centering\n")
            tex.write("      \\tiny\\renewcommand{\\arraystretch}{0.85}\\setlength{\\tabcolsep}{3pt}\n")
            tex.write("      \\begin{tabular}{lrrrrr}\n")
            tex.write("        \\toprule\n")
            tex.write(f"        {param_name} & median & Q1 & Q3 & min & max \\\\\n")
            tex.write("        \\midrule\n")
            for i, values in enumerate(data):
                arr = np.array(values)
                median = np.median(arr)
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                mi = np.min(arr)
                mx = np.max(arr)
                tex.write(f"        {x_labels[i]:.4f} & {median:.4f} & {q1:.4f} & {q3:.4f} & {mi:.4f} & {mx:.4f} \\\\\n")
            tex.write("        \\bottomrule\n")
            tex.write("      \\end{tabular}\n")
            tex.write("    \\end{column}\n")
            tex.write("  \\end{columns}\n")
            tex.write("\\end{frame}\n")

    def do(self, clear_previous_results=False, reverse_execution=False):
        self.log_replaced_data = ""
        self._boxplot_data = {}
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
                    self._per_run_tmp = {}
                    result_iteration_to_check = pd.DataFrame()
                    filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        results_mc = {
                            executor.submit(
                                self.load_or_execute_model,
                                model_configuration,
                                model_parameters,
                                filename_for_iteration,
                                i,
                                clear_previous_results,
                                seeds_for_random[i + position_inside_seeds_for_random],
                            ): i
                            for i in range(self.MC)
                        }
                        for future in concurrent.futures.as_completed(results_mc):
                            result_iteration_to_check = pd.concat([result_iteration_to_check, future.result()])

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

                        for i, future in enumerate(concurrent.futures.as_completed(results_mc)):
                            result_mc = future.result()
                            if "bankruptcies" in result_mc and not (
                                np.all(result_mc["bankruptcies"] == 0)
                                or np.all(result_mc["bankruptcies"] == result_mc["bankruptcies"].iloc[0])
                                or np.all(result_mc["ir"] == 0)
                                or np.all(result_mc["ir"] == result_mc["ir"].iloc[0])
                            ):
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", scipy.stats.NearConstantInputWarning)
                                    correlation_coefficient, p_value = scipy.stats.pearsonr(
                                        result_mc["ir"], result_mc["bankruptcies"]
                                    )
                                    correlation_coefficient1, p_value1 = scipy.stats.pearsonr(
                                        result_mc["ir"][1:], result_mc["bankruptcies"][:-1]
                                    )
                                    correlation_file.write(
                                        f"{filename_for_iteration}_{i} = {model_configuration} {model_parameters}\n"
                                    )
                                    correlation_file.write(format_correlation_values(0, correlation_coefficient, p_value))
                                    correlation_file.write(
                                        format_correlation_values(1, correlation_coefficient1, p_value1)
                                    )
                                    montecarlo_iteration_perfect_correlation = (
                                        montecarlo_iteration_perfect_correlation
                                        and (
                                            (correlation_coefficient1 > 0 and p_value1 <= 0.10)
                                            or (correlation_coefficient > 0 and p_value <= 0.10)
                                        )
                                    )
                            for col in result_mc.columns:
                                if col == 't':
                                    continue
                                col_mean = pd.to_numeric(result_mc[col], errors="coerce").mean()
                                self._per_run_tmp.setdefault(col, []).append(col_mean)
                            result_iteration = pd.concat([result_iteration, result_mc])

                    if montecarlo_iteration_perfect_correlation:
                        montecarlo_iteration_perfect_correlations[
                            str(model_configuration) + " " + str(model_parameters)
                        ] = format_correlation_values(1, correlation_coefficient, p_value)

                    for k in result_iteration.keys():
                        if k.strip() == "t":
                            continue
                        numeric_result = pd.to_numeric(result_iteration[k], errors="coerce")
                        mean_estimated = numeric_result.mean()
                        warnings.filterwarnings("ignore")
                        std_estimated = numeric_result.std()
                        if k in results_to_plot:
                            results_to_plot[k].append([mean_estimated, std_estimated])
                        else:
                            results_to_plot[k] = [[mean_estimated, std_estimated]]
                    results_x_axis.append(self.get_title_for(model_configuration, model_parameters))
                    position_inside_seeds_for_random += self.MC
                    for col, values in self._per_run_tmp.items():
                        self._boxplot_data.setdefault(col, []).append(values)
                    progress_bar.next()

            progress_bar.finish()
            if montecarlo_iteration_perfect_correlations:
                for perfect_correlation in montecarlo_iteration_perfect_correlations:
                    correlation_file.write(
                        f"{perfect_correlation} : {montecarlo_iteration_perfect_correlations[perfect_correlation]}\n"
                    )
            correlation_file.close()
            print(f"Saving results in {self.OUTPUT_DIRECTORY}/results.{self.OUTPUT_FORMAT}...")
            if self.OUTPUT_FORMAT.lower() == "csv":
                self.save_csv(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
            else:
                self.save_gdt(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
            if self._boxplot_data:
                self._generate_boxplot(results_x_axis, f"{self.OUTPUT_DIRECTORY}/")
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

    def get_cross_correlation_result(self, data, column_a, column_b):
        try:
            correlation_value = scipy.stats.pearsonr(data[column_a], data[column_b])
        except ValueError:
            correlation_value = None
        if correlation_value:
            return [correlation_value.statistic, correlation_value.pvalue]
        return [0, 0]

    def do_stats_market(self):
        self.log_replaced_data = ""
        results_to_plot = {k: [] for k in ["cross_psi_ir", "cross_psi_ir_lenders"]}
        results_x_axis = []
        self.verify_directories()
        progress_bar = Bar("Executing models (stats_market)", max=self.get_num_models())
        progress_bar.update()
        for model_configuration in self.get_models(self.config):
            for model_parameters in self.get_models(self.parameters):
                filename_for_iteration = self.get_filename_for_iteration(model_parameters, model_configuration)
                result_iteration = pd.DataFrame()
                for i in range(self.MC):
                    result_iteration = pd.concat(
                        [
                            result_iteration,
                            self.load_or_execute_model(
                                model_configuration,
                                model_parameters,
                                filename_for_iteration,
                                i,
                                stats_market=True,
                            ),
                        ]
                    )

                for k in result_iteration.keys():
                    if k.strip() == "t":
                        continue
                    mean_estimated = result_iteration[k].mean()
                    warnings.filterwarnings("ignore")
                    std_estimated = result_iteration[k].std()
                    if k in results_to_plot:
                        results_to_plot[k].append([mean_estimated, std_estimated])
                    else:
                        results_to_plot[k] = [[mean_estimated, std_estimated]]
                results_to_plot["cross_psi_ir"].append(
                    self.get_cross_correlation_result(result_iteration, "psi", "ir")
                )
                results_to_plot["cross_psi_ir_lenders"].append(
                    self.get_cross_correlation_result(result_iteration, "psi", "bankruptcies")
                )
                results_x_axis.append(self.get_title_for(model_configuration, model_parameters))
                progress_bar.next()

        progress_bar.finish()
        print(f"Saving results in {self.OUTPUT_DIRECTORY}/resultsb.gdt|csv...")
        self.save_csv(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/", "resultsb.csv")
        self.save_gdt(results_to_plot, results_x_axis, f"{self.OUTPUT_DIRECTORY}/", "resultsb.gdt")
        if self.log_replaced_data:
            print(self.log_replaced_data)
        self.plot(results_to_plot, results_x_axis, self.get_title_for(self.config, self.parameters),
                  f"{self.OUTPUT_DIRECTORY}/")

    def generate_random_seeds_for_this_execution(self):
        seeds_for_random = []
        random.seed(self.SEED_FOR_EXECUTION)
        for _ in self.get_models(self.config):
            for _ in self.get_models(self.parameters):
                for _ in range(self.MC):
                    seeds_for_random.append(random.randint(1000, 99999))
        return seeds_for_random

    def discard_execution_of_iteration(self, filename_for_iteration, i):
        if os.path.exists(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv"):
            base, ext = os.path.splitext(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.csv")
        elif os.path.exists(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt"):
            base, ext = os.path.splitext(f"{self.OUTPUT_DIRECTORY}/{filename_for_iteration}_{i}.gdt")
        else:
            base, ext = "?", "???"
        offset = 1
        while True:
            new_name = f"{base}_discarded{offset}{ext}"
            if not os.path.exists(new_name):
                os.rename(f"{base}{ext}", new_name)
                break
            offset += 1

    def clear_results(self):
        try:
            os.remove(self.OUTPUT_DIRECTORY + "/results.csv")
        except FileNotFoundError:
            pass
        try:
            os.remove(self.OUTPUT_DIRECTORY + "/results.gdt")
        except FileNotFoundError:
            pass


class Runner:
    def __init__(self, experiment_runner):
        self.experiment_runner = experiment_runner
        self.parser = argparse.ArgumentParser(description="Executes MC experiments using interbank model")
        self.parser.add_argument("--do", default=False, action=argparse.BooleanOptionalAction,
                                 help=f"Execute the experiment and saves the results in {experiment_runner.OUTPUT_DIRECTORY}")
        self.parser.add_argument("--listnames", default=False, action=argparse.BooleanOptionalAction,
                                 help="Print combinations to generate")
        self.parser.add_argument("--clear_results", default=False, action=argparse.BooleanOptionalAction,
                                 help="Ignore generated results.csv and create it again")
        self.parser.add_argument("--clear", default=False, action=argparse.BooleanOptionalAction,
                                 help="Ignore generated models and create them again")
        self.parser.add_argument("--errorbar", default=False, action=argparse.BooleanOptionalAction,
                                 help="Plot also the errorbar (deviation error)")
        self.parser.add_argument("--reverse", default=False, action=argparse.BooleanOptionalAction,
                                 help="Execute the experiment in opposite order")
        self.parser.add_argument("--statsb", default=False, action=argparse.BooleanOptionalAction,
                                 help="Generate also resultsb.txt|resultsb.gdt")
        self.parser.add_argument("--plot_removing_first", default=False, action=argparse.BooleanOptionalAction,
                                 help="Remove the first x-axis value from plots")
        self.parser.add_argument("--plot", default=False, action=argparse.BooleanOptionalAction,
                                 help="Load existing results.gdt and regenerate plots only")
        self.parser.add_argument("--directory", type=str, default=None,
                                 help="Override OUTPUT_DIRECTORY (path to results.gdt)")
        self.parser.add_argument("--graph", type=str, default='',
                                 help="Save Erdos-Renyi graphs as PNG and JSON. Comma-separated time steps (e.g. 5,10,15) or 'all' for every step")
        self.parser.add_argument("--report", default=False, action=argparse.BooleanOptionalAction,
                                 help="Generate report.tex from existing .tex files in OUTPUT_DIRECTORY")

    def generate_report(self, directory):
        directory = directory.rstrip("/\\") + "/"
        order = ["bankruptcies", "bankruptcies_all", "ir", "psi", "prob_bankruptcy", "bad_debt", "profits", "rationing", "boxplot"]
        preamble = r"""\documentclass{beamer}
\usetheme{Warsaw}
\definecolor{pantone2955}{RGB}{0,51,96}
\setbeamercolor{structure}{fg=pantone2955}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{booktabs}
\usepackage{times}
\usepackage[T1]{fontenc}

\begin{document}
"""
        with open(f"{directory}report.tex", "w", encoding="utf-8") as f:
            f.write(preamble)
            for name in order:
                tex_path = f"{directory}{name}.tex"
                if os.path.isfile(tex_path):
                    f.write(f"\\input{{{name}.tex}}\n")
                else:
                    print(f"Warning: {tex_path} not found, skipping")
            f.write("\\end{document}\n")
        print(f"Generated {directory}report.tex")

    def plot_only(self, experiment, directory):
        directory = directory.rstrip("/\\") + "/"
        results_to_plot, results_x_axis, metadata = experiment._load_from_gdt(f"{directory}results.gdt")
        if not results_to_plot:
            print("No results found. Run with --do first.")
            return
        print(f"Loaded data from {directory}")
        experiment.MC = metadata.get("mc", experiment.MC)
        experiment.NAME_OF_X_SERIES = metadata.get("name_of_x_series", experiment.NAME_OF_X_SERIES)
        results_comparing, results_comparing2 = experiment.load_comparing(results_x_axis)
        title_x = experiment.NAME_OF_X_SERIES if experiment.NAME_OF_X_SERIES else ""
        experiment.plot(
            results_to_plot, results_x_axis,
            title_x,
            f"{directory}/",
            results_comparing, results_comparing2,
        )

    def do(self):
        args = self.parser.parse_args()
        experiment = self.experiment_runner()
        if args.clear_results:
            experiment.clear_results()
        experiment.error_bar = args.errorbar
        experiment.plot_removing_first = args.plot_removing_first
        if args.graph.lower() == 'all':
            experiment.save_graph = 'all'
        elif args.graph:
            experiment.save_graph = set(int(x) for x in args.graph.split(','))
        else:
            experiment.save_graph = set()
        directory = args.directory if args.directory else experiment.OUTPUT_DIRECTORY
        if args.listnames:
            experiment.listnames()
        elif args.plot:
            self.plot_only(experiment, directory)
            return experiment
        elif args.do:
            experiment.OUTPUT_DIRECTORY = directory
            experiment.clear_results()
            experiment.do(clear_previous_results=args.clear, reverse_execution=args.reverse)
            if args.statsb:
                experiment.do_stats_market()
            return experiment
        elif args.report:
            self.generate_report(directory)
        elif args.statsb:
            experiment.do_stats_market()
        else:
            self.parser.print_help()
