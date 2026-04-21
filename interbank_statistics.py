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
import numpy as np
import os
import sys
import pandas as pd
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
        self.psi = []
        self.lenders = []
        self.potential_lenders = []
        self.borrowers = []
        self.interest_rates = []
        self.var_d1 = []
        self.var_d2 = []
        self.var_d = []
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

    def init(self):
        for attr in dir(self):
            if isinstance(getattr(self,attr), list):
                setattr(self, attr, [])

    def finish(self):
        self.save(self.export_datafile, str(self.model.config))
        result = pd.DataFrame()
        if not self.model.log.interactive:
            for element_name, element in self.enumerate_results():
                result[element_name] = element
        return result

    def enumerate_results(self):
        for element in dir(self):
            if isinstance(getattr(self, element), np.ndarray) or isinstance(getattr(self, element), list):
               yield element, getattr(self, element)

    def get_plots(self):
        #TODO
        if self.generate_plots:
            pass
    
    def generate_gdt_file(self, filename, header, num_of_observations=None):
        element = lxml.builder.ElementMaker()
        gretl_data = element.gretldata
        xml_description = element.description
        xml_variables = element.variables
        variable = element.variable
        xml_observations = element.observations
        observation = element.obs
        num_variables = 0
        for variable_name, _ in self.enumerate_results():
            num_variables += 1

        variables = xml_variables(count=f'{num_variables}')
        header_text = ''
        for item in header:
            header_text += item + ' '
        # header_text will be present as label in the first variable
        # correlation_result will be present as label in the second variable
        if num_of_observations:
            range_of_observations = range(num_of_observations)
        else:
            range_of_observations = range(self.model.config.T)
        i = 1
        for variable_name, observations in self.enumerate_results():
            if variable_name == 'leverage':
                variable_name += '_'
            if i == 1:
                variables.append(variable(name='{}'.format(variable_name), label='{}'.format(header_text)))
            else:
                variables.append(variable(name='{}'.format(variable_name)))
            if len(observations)<len(range_of_observations):
                self.model.log.error("stats ",
                  f"{variable_name} is shorter than expected length ({len(observations)}<{len(range_of_observations)})")
                sys.exit(-1)
            i = i + 1
        xml_observations = xml_observations(count='{}'.format(self.model.config.T), labels='false')
        for i in range_of_observations:
            string_obs = ''
            for variable_name, observations in self.enumerate_results():
                if variable_name == 'real_t':
                    string_obs += f'{i:3} '
                else:
                    string_obs += f'{observations[i]} '
            xml_observations.append(observation(string_obs))
        gdt_result = gretl_data(xml_description(header_text), variables,
                                xml_observations, version='1.4', name='interbank',
                                frequency='special:1', startobs='1', endobs='{}'.format(self.model.config.T),
                                type='time-series')
        with gzip.open(filename, 'w') as export_datafile:
            export_datafile.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
            export_datafile.write(lxml.etree.tostring(gdt_result, encoding=str).encode('ascii'))

    def generate_csv_or_txt(self, filename, header, delimiter):
        file_header = ''
        for line_header in header:
            file_header += '# {}\n'.format(line_header)
        file_header += "# pd.read_csv('file{}',header={}', delimiter='{}')\nt".format(self.output_format,
                                                                                      len(header) + 1, delimiter)
        with open(filename, 'w', encoding='utf-8') as save_file:
            save_file.write(file_header)
            for element_name, _ in self.enumerate_results():
                save_file.write('{}{}'.format(delimiter, element_name))
            save_file.write('\n')
            i_seq = 0
            for i in range(self.model.config.T):
                save_line = True
                line = '{}'.format(i_seq)
                for name_element, element in self.enumerate_results():
                    if name_element == 'real_t':
                        line += '{}{}'.format(delimiter, i)
                    else:
                        line += '{}{}'.format(delimiter, element[i])
                if save_line:
                    i_seq += 1
                    save_file.write(line + '\n')

    def get_export_path(self, filename, ending_name=''):
        if not os.path.dirname(filename):
            filename = '{}/{}'.format(self.output_directory, filename)
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
            if self.output_format.lower() == '.both':
                self.output_format = '.csv'
                self.generate_csv_or_txt(self.get_export_path(export_datafile), header, ';')
                self.output_format = '.gdt'
                self.generate_gdt_file(self.get_export_path(export_datafile), header)
            elif self.output_format.lower() == '.csv':
                self.generate_csv_or_txt(self.get_export_path(export_datafile), header, ';')
            elif self.output_format.lower() == '.txt':
                self.generate_csv_or_txt(self.get_export_path(export_datafile), header, '\t')
            else:
                self.generate_gdt_file(self.get_export_path(export_datafile), header)


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

    def define_output_file(self, output_file):
        if output_file:
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

    def compute_var_d1(self):
        self.var_d1.append(np.sum(self.model.varD1))

    def compute_interest_rate(self):
        self.interest_rates.append( np.nanmean(self.model.interest_rate) )

    def compute_var_d2(self):
        self.var_d2.append(np.sum(self.model.varD2))
        self.var_d.append(self.var_d1[-1] + self.var_d2[-1])

    def compute_liquidity(self):
        self.liquidity.append(np.sum(self.model.C))

    def compute_rationing(self, num_of_rationed, total_rationed):
        self.rationing.append(total_rationed)
        self.num_of_rationed.append(num_of_rationed)

    def compute_deposits(self):
        self.deposits.append(np.sum(self.model.D))

    def compute_reserves(self):
        self.reserves.append(np.sum(self.model.R))

    def compute_bad_debt(self):
        self.bad_debt.append(np.sum(np.sum(self.model.bad_debt)))

    def compute_equity(self):
        self.equity.append(np.sum(self.model.E))

    def compute_assets(self):
        assets = self.model.C+self.model.L+self.model.R
        # i=lenders, so then psi!=null
        self.asset_i.append( np.sum( assets[~np.isnan(self.model.psi)]))
        self.asset_j.append( np.sum( assets[np.isnan(self.model.psi)]))

    def compute_potential_lenders(self):
        # model.lenders>=0 -> that bank has lender in this step
        self.potential_lenders.append(np.sum(self.model.s>0))

    def compute_lenders_and_borrowers(self):
        self.borrowers.append(np.sum(self.model.lenders>=0))
        self.lenders.append(np.sum(np.isnan(self.model.haircut)))

    def compute_graph(self):
        self.gcs.append(self.model.lenderchange.determine_current_graph_gcs())
        self.grade_avg.append(self.model.lenderchange.determine_current_graph_grade_avg())
        self.communities.append(self.model.lenderchange.determine_current_communities())
        self.communities_not_alone.append(self.model.lenderchange.determine_current_communities_not_alone())

    def compute_capacity(self):
        self.capacity.append(np.nanmean(self.model.capacity))

    def compute_prob_bankruptcy(self):
        self.prob_bankruptcy.append(np.nanmean(self.model.prob_bankruptcy))

