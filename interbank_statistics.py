#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Generates a simulation of an interbank network
# Usage: interbank.py --help
#
#
# author: hector@bith.net
# date:   04/2023, 09/2025, 03/2026
import gzip
import warnings
import numpy as np
import os
import sys
import pandas as pd
import scipy.stats
import lxml.etree
import lxml.builder
from interbank_log import Log


class Statistics:
    def __init__(self, in_model):
        self.graph_format = '.svg'
        self.output_format = '.gdt'
        self.output_directory = 'output'
        self.generate_plots = False
        self.export_datafile = None
        self.model = in_model
        self.reserves = []
        self.grade_avg = []
        self.communities = []
        self.communities_not_alone = []
        self.gcs = []
        self.potential_lenders = []
        self.num_loans = []
        self.loans = []
        self.psi = []
        self.bankruptcies = []
        self.ir = []
        self.ir_avg = []
        self.var_D1 = []
        self.var_D2 = []
        self.var_D = []
        self.d1 = []
        self.d2 = []
        self.asset_i = []
        self.asset_j = []
        self.bad_debt = []
        self.capacity = []
        self.prob_bankruptcy = []
        self.deposits = []
        self.liquidity = []
        self.rationing = []
        self.num_of_rationed = []
        self.equity = []
        self.profits = []
        self.correlation = []

    # ---------------------------------------------------------------------------
    # Cross-correlation: psi vs other variables
    # ---------------------------------------------------------------------------

    # Pairs of (attribute_name, display_label) to correlate against ir
    CORRELATION_PAIRS = [
        ('psi',             'psi'),
        ('prob_bankruptcy', 'prob_bankruptcy'),
        ('bankruptcies',    'bankruptcies'),
        ('bad_debt',       'bad_debt'),
    ]

    def determine_cross_correlation(self):
        """Compute Pearson r between ir and each variable in CORRELATION_PAIRS."""
        self.correlation = []
        if len(self.ir) < 2:
            return
        ir_values = np.asarray(self.ir, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr, label in self.CORRELATION_PAIRS:
                other = getattr(self, attr, [])
                if len(other) < 2:
                    self.correlation.append((label, None))
                    continue
                other_values = np.asarray(other, dtype=float)
                min_length = min(len(ir_values), len(other_values))
                if min_length < 2:
                    self.correlation.append((label, None))
                    continue
                ir_slice = ir_values[:min_length]
                other_slice = other_values[:min_length]
                valid_mask = np.isfinite(ir_slice) & np.isfinite(other_slice)
                if np.count_nonzero(valid_mask) < 2:
                    self.correlation.append((label, None))
                    continue
                try:
                    result = scipy.stats.pearsonr(ir_slice[valid_mask], other_slice[valid_mask])
                    self.correlation.append((label, result))
                except ValueError:
                    self.correlation.append((label, None))

    def get_cross_correlation_result(self, index):
        """Return a formatted string for correlation[index], or empty string."""
        if index >= len(self.correlation):
            return ''
        label, result = self.correlation[index]
        if result is None:
            return f'correl ir/{label} n/a'
        r = result.statistic if hasattr(result, 'statistic') else result[0]
        p = result.pvalue   if hasattr(result, 'pvalue')    else result[1]
        if r > 0 and p < 0.05:
            stars = '**'
        elif r > 0 and p < 0.10:
            stars = '* '
        else:
            stars = '  '
        return f'correl ir/{label} r={r:.3f} p={p:.3f} {stars}'

    def print_cross_correlation_summary(self):
        if not self.correlation:
            return
        print('Cross-correlations:')
        for index in range(len(self.correlation)):
            result = self.get_cross_correlation_result(index)
            if result:
                print(f'- {result}')

    def init(self):
        for attr, value in list(self.__dict__.items()):
            if isinstance(value, list):
                setattr(self, attr, [])

    def finish(self):
        self.determine_cross_correlation()
        if self.model.log.interactive:
            self.print_cross_correlation_summary()
        self.save( export_datafile=self.export_datafile, export_description=str(self.model.config))
        result = pd.DataFrame()
        if not self.model.log.interactive:
            for element_name, element in self.enumerate_time_series_results():
                result[element_name] = element
        return result

    def enumerate_results(self):
        for element, value in self.__dict__.items():
            if isinstance(value, np.ndarray) or isinstance(value, list):
                yield element, value

    def enumerate_time_series_results(self):
        for element_name, element_values in self.enumerate_results():
            if element_name == 'correlation' or element_name.isupper():
                continue
            yield element_name, element_values

    def get_plots(self):
        #TODO
        if self.generate_plots:
            pass

    def build_equity_validity_line(self):
        equity_values = np.asarray(self.equity, dtype=float)
        if equity_values.size == 0:
            return 'Validity KO: finite values >=99%: 0.00%; std(diff)>0.25: nan; sign_change_ratio > 0.35: nan; near_zero_step_ratio < 0.04: nan'

        finite_mask = np.isfinite(equity_values)
        finite_ratio = float(np.mean(finite_mask))
        filtered_equity = equity_values[finite_mask]
        diffs = np.diff(filtered_equity) if filtered_equity.size >= 2 else np.array([])

        std_diff = float(np.std(diffs)) if diffs.size > 0 else np.nan
        sign_change_ratio = np.nan
        if diffs.size >= 2:
            sign_changes = np.count_nonzero(np.sign(diffs[1:]) != np.sign(diffs[:-1]))
            sign_change_ratio = sign_changes / (len(diffs) - 1)
        near_zero_step_ratio = float(np.mean(np.abs(diffs) < 1e-9)) if diffs.size > 0 else np.nan

        is_valid = (
            finite_ratio >= 0.99
            and np.isfinite(std_diff)
            and std_diff > 0.25
            and np.isfinite(sign_change_ratio)
            and sign_change_ratio > 0.35
            and np.isfinite(near_zero_step_ratio)
            and near_zero_step_ratio < 0.04
        )

        validity_text = 'OK' if is_valid else 'KO'
        return (
            f'Validity {validity_text}: '
            f'finite values >=99%: {finite_ratio * 100:.2f}%; '
            f'std(diff)>0.25: {std_diff:.2f}; '
            f'sign_change_ratio > 0.35: {sign_change_ratio:.2f}; '
            f'near_zero_step_ratio < 0.04: {near_zero_step_ratio:.2f}'
        )
    
    def generate_gdt_file(self, filename, header, selected_indices=None, include_real_t=False):
        element = lxml.builder.ElementMaker()
        gretl_data = element.gretldata
        xml_description = element.description
        xml_variables = element.variables
        variable = element.variable
        xml_observations = element.observations
        observation = element.obs
        data_series = list(self.enumerate_time_series_results())
        num_variables = len(data_series) + (1 if include_real_t else 0)
        variables = xml_variables(count=f'{num_variables}')
        header_text = '\n'.join(header)
        if selected_indices is None:
            selected_indices = list(range(self.model.config.T))

        next_variable_index = 1
        if include_real_t:
            variables.append(variable(name='real_t', label=header_text))
            next_variable_index += 1

        for i, (variable_name, observations) in enumerate(data_series, next_variable_index):
            if variable_name == 'leverage':
                variable_name += '_'
            if i == 1:
                variables.append(variable(name=variable_name, label=header_text))
            elif i - 2 < len(self.correlation):
                variables.append(variable(name=variable_name,
                                          label=self.get_cross_correlation_result(i - 2)))
            else:
                variables.append(variable(name=variable_name))
        xml_observations = xml_observations(count=f'{len(selected_indices)}', labels='false')
        warned_short_variables = set()
        for i in selected_indices:
            string_obs = ''
            if include_real_t:
                string_obs += f'{i:3} '
            for variable_name, observations in data_series:
                if i < len(observations):
                    string_obs += f'{observations[i]} '
                else:
                    if variable_name not in warned_short_variables:
                        warnings.warn(
                            f"'{variable_name}' shorter than expected ({len(observations)}<{self.model.config.T}); filling with nan.",
                            RuntimeWarning,
                        )
                        warned_short_variables.add(variable_name)
                    string_obs += 'nan '
            xml_observations.append(observation(string_obs))
        gdt_result = gretl_data(xml_description(header_text), variables,
                                xml_observations, version='1.4', name='interbank',
                                frequency='special:1', startobs='1', endobs='{}'.format(len(selected_indices)),
                                type='time-series')
        with gzip.open(filename, 'w') as export_datafile:
            export_datafile.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
            export_datafile.write(lxml.etree.tostring(gdt_result, encoding=str).encode('ascii'))

    def save_auxiliary_file_with_valid_ir(self, export_datafile, header):
        if len(self.ir) < 2:
            return
        valid_ir_indices = [
            index for index in range(min(self.model.config.T, len(self.ir))) if np.isfinite(self.ir[index])
        ]
        if len(valid_ir_indices) == self.model.config.T:
            return
        if len(valid_ir_indices) < 2:
            return
        return valid_ir_indices

    def generate_csv_or_txt(self, filename, header, delimiter, selected_indices=None, include_real_t=False):
        file_header = ''
        for line_header in header:
            file_header += '# {}\n'.format(line_header)
        file_header += "# pd.read_csv('file{}',header={}', delimiter='{}')\nt".format(self.output_format,
                                                                                       len(header) + 1, delimiter)
        data_series = list(self.enumerate_time_series_results())
        if selected_indices is None:
            selected_indices = list(range(self.model.config.T))
        with open(filename, 'w', encoding='utf-8') as save_file:
            save_file.write(file_header)
            if include_real_t:
                save_file.write(f'{delimiter}real_t')
            for element_name, _ in data_series:
                save_file.write('{}{}'.format(delimiter, element_name))
            save_file.write('\n')
            i_seq = 0
            warned_short_variables = set()
            for i in selected_indices:
                save_line = True
                line = '{}'.format(i_seq)
                if include_real_t:
                    line += '{}{}'.format(delimiter, i)
                for name_element, element in data_series:
                    if i < len(element):
                        line += '{}{}'.format(delimiter, element[i])
                    else:
                        if name_element not in warned_short_variables:
                            warnings.warn(
                                f"'{name_element}' shorter than expected ({len(element)}<{self.model.config.T}); filling with nan.",
                                RuntimeWarning,
                            )
                            warned_short_variables.add(name_element)
                        line += '{}nan'.format(delimiter)
                if save_line:
                    i_seq += 1
                    save_file.write(line + '\n')

    def get_export_path(self, filename, ending_name=''):
        if not os.path.dirname(filename):
            filename = '{}/{}'.format(self.output_directory if self.output_directory else '.', filename)
        path, extension = os.path.splitext(filename)
        if ending_name:
            return path + ending_name
        else:
            return path + self.output_format.lower()

    def save(self, export_datafile=None, export_description=None):
        if self.export_datafile:
            if export_description:
                header = ['{}'.format(export_description)]
            else:
                header = ['{} T={} N={}'.format(__name__, self.model.config.T, self.model.config.N)]
            header.append(self.build_equity_validity_line())
            valid_ir_indices = self.save_auxiliary_file_with_valid_ir(export_datafile, header)
            if self.output_format.lower() == '.both':
                self.output_format = '.csv'
                self.generate_csv_or_txt(self.get_export_path(export_datafile), header, ';')
                if valid_ir_indices:
                    self.generate_csv_or_txt(
                        self.get_export_path(export_datafile, '_b.csv'),
                        header,
                        ';',
                        selected_indices=valid_ir_indices,
                        include_real_t=True,
                    )
                self.output_format = '.gdt'
                self.generate_gdt_file(self.get_export_path(export_datafile), header)
                if valid_ir_indices:
                    self.generate_gdt_file(
                        self.get_export_path(export_datafile, '_b.gdt'),
                        header,
                        selected_indices=valid_ir_indices,
                        include_real_t=True,
                    )
            elif self.output_format.lower() == '.csv':
                self.generate_csv_or_txt(self.get_export_path(export_datafile), header, ';')
                if valid_ir_indices:
                    self.generate_csv_or_txt(
                        self.get_export_path(export_datafile, '_b.csv'),
                        header,
                        ';',
                        selected_indices=valid_ir_indices,
                        include_real_t=True,
                    )
            elif self.output_format.lower() == '.txt':
                self.generate_csv_or_txt(self.get_export_path(export_datafile), header, '\t')
            else:
                self.generate_gdt_file(self.get_export_path(export_datafile), header)
                if valid_ir_indices:
                    self.generate_gdt_file(
                        self.get_export_path(export_datafile, '_b.gdt'),
                        header,
                        selected_indices=valid_ir_indices,
                        include_real_t=True,
                    )


    def define_plot_format(self, plot_format):
        match plot_format.lower():
            case 'none':
                self.plot_format = None
            case 'svg':
                self.plot_format = '.svg'
            case 'png':
                self.plot_format = '.png'
            case 'gif':
                self.plot_format = '.gif'
            case 'pdf':
                self.plot_format = '.pdf'
            case 'agr':
                self.plot_format = '.agr'
            case _:
                print('Invalid plot file format: {}'.format(plot_format))
                sys.exit(-1)

    def define_output_directory(self, output_directory):
        if output_directory:
            self.output_directory = output_directory

    def define_output_file(self, output_file):
        if output_file:
            if os.path.dirname(output_file):
                self.output_directory = os.path.dirname(output_file)
            self.export_datafile = os.path.basename(output_file)

    def define_output_format(self, output_format):
        match output_format.lower():
            case 'both':
                self.output_format = '.both'
            case 'gdt':
                self.output_format = '.gdt'
            case 'csv':
                self.output_format = '.csv'
            case 'txt':
                self.output_format = '.txt'
            case _:
                print('Invalid output file format: {}'.format(output_format))
                sys.exit(-1)

    def compute_psi(self):
        self.psi.append( np.nanmean(self.model.psi) )

    def compute_bankruptcies(self):
        self.bankruptcies.append(np.nansum(self.model.failed))

    def compute_d1(self):
        self.d1.append(np.nansum(self.model.d))
        self.var_D1.append(np.sum(self.model.varD1))

    def compute_ir(self):
        self.ir.append( np.nanmean(self.model.interest_rate) )

    def compute_d2(self):
        self.d2.append(np.nansum(self.model.d))
        self.var_D2.append(np.sum(self.model.varD2))
        self.var_D.append(self.model.varD1[-1] + self.model.varD2[-1])

    def compute_liquidity(self):
        self.liquidity.append(np.nansum(self.model.C))

    def compute_rationing(self, num_of_rationed, total_rationed):
        self.rationing.append(total_rationed)
        self.num_of_rationed.append(num_of_rationed)

    def compute_profits(self, profits_paid):
        self.profits.append(profits_paid)

    def compute_deposits(self):
        self.deposits.append(np.nansum(self.model.D))

    def compute_reserves(self):
        self.reserves.append(np.nansum(self.model.R))

    def compute_bad_debt(self):
        self.bad_debt.append(abs(np.nansum(self.model.bad_debt)))

    def compute_equity(self):
        self.equity.append(np.nansum(self.model.E))

    def compute_assets(self):
        assets = self.model.C+self.model.L+self.model.R
        # i=lenders, so then psi!=null
        self.asset_i.append( np.sum( assets[~np.isnan(self.model.psi)]))
        self.asset_j.append( np.sum( assets[np.isnan(self.model.psi)]))

    def compute_potential_lenders(self):
        self.potential_lenders.append(np.sum(self.model.s>0))

    def compute_num_loans(self):
        self.num_loans.append(np.sum((self.model.l > 0) & ~np.isnan(self.model.l)))
        self.loans.append(np.nansum(self.model.l))

    def compute_ir_avg(self):
        if np.any(self.model.l > 0):
            self.ir_avg.append(np.mean(self.model.interest_rate[self.model.l > 0]))
        else:
            self.ir_avg.append(np.nan)

    def compute_graph(self):
        self.gcs.append(self.model.lenderchange.determine_current_graph_gcs())
        self.grade_avg.append(self.model.lenderchange.determine_current_graph_grade_avg())
        self.communities.append(self.model.lenderchange.determine_current_communities())
        self.communities_not_alone.append(self.model.lenderchange.determine_current_communities_not_alone())

    def compute_capacity(self):
        self.capacity.append(np.nanmean(self.model.capacity))

    def compute_prob_bankruptcy(self):
        self.prob_bankruptcy.append(1 - np.nanmean(self.model.prob_bankruptcy))

