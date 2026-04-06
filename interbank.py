#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Generates a simulation of an interbank network
# Usage: interbank.py --help
#
#
# author: hector@bith.net
# date:   04/2023, 09/2025, 03/2026
import random

import numpy as np
import os
import sys
import pandas as pd
import logging
import networkx as nx
import matplotlib.pyplot as plt
import json
import warnings
import lxml


class Config:
    """
        Configuration parameters for the interbank network
    """
    T: int = 10  # time (1000)
    N: int = 10 # number of banks (50)

    reserves: float = 0.02

    # probability of attachment of lender change
    p = 0.8

    # screening costs:
    m = 0.015

    # seed applied for random values (set during initialize)
    seed: int = 5

    # shocks parameters: mi=0.7 omega=0.6 for perfect balance
    # less omega, more negative is the shock
    mu: float = 0.7  # mi µ
    omega: float = 0.6  # omega ω

    # screening costs
    phi: float = 0.025  # phi Φ
    chi: float = 0.015  # chi Χ

    xi: float = 0.3  # xi ξ liquidation cost of collateral
    rho: float = 0.3  # rho ρ fire sale cost

    beta: float = 5  # β beta intensity of breaking the connection (5)
    alfa: float = 0.1  # α alfa below this level of E or D, we will bankrupt the bank

    # If true, psi variable will be ignored:
    psi_endogenous = True
    psi: float = 0.3  # market power parameter : 0 perfect competence .. 1 monopoly

    # banks initial parameters
    # L + C + R = D + E
    # but R = 0.02*D and C_i0= 30-2.7=27.3 and R=2.7
    #L_i0: float = 120  # long term assets
    C_i0: float = 5  # capital BEFORE RESERVES ESTIMATION, after it will be 27.3
    L_i0: float = 5
    # R_i0=2.7
    D_i0: float = 9  # deposits
    E_i0: float = 1  # equity
    r_i0: float = 0.02  # initial rate

    # what elements are in the results.csv file, and also which are plot.
    # 1 if also plot, 0 not to plot:
    ELEMENTS_STATISTICS_NO_PLOT = ['best_lender_clients', 'fitness', 'policy', 'leverage', 'systemic_leverage',
                                   'sum_loans', 'potential_lenders', 'active_lenders', 'active_borrowers',
                                   'prob_bankruptcy']
    ELEMENTS_STATISTICS_NO_SHOW = ['gcs', 'grade_avg', 'communities', 'communities_not_alone']

    # only if we have a graph for lender_change:
    ELEMENTS_STATISTICS_GRAPHS = {
        'grade_avg': True, 'communities': True, 'communities_not_alone': True, 'gcs': True
    }
    ELEMENTS_STATISTICS_LOG = {'equity'}

    ELEMENTS_TRANSLATIONS = {'bankruptcy': 'bankruptcies',
                             'P': 'prob_change_lender',
                             'B': 'bad_debt'}


class Log:
    """
    The class acts as a logger and helpers to represent the data and evol from the Model.
    """
    logger = logging.getLogger('interbank')
    modules = []
    model = None
    logLevel = 'DEBUG'
    progress_bar = None

    def __init__(self, its_model):
        self.model = its_model

    def do_progress_bar(self, message, maximum):
        from progress.bar import Bar
        self.progress_bar = Bar(message, max=maximum)

    @staticmethod
    def format_number(number):
        result = '{}'.format(number)
        while len(result) > 5 and result[-1] == '0':
            result = result[:-1]
        while len(result) > 5 and result.find('.') > 0:
            result = result[:-1]
        while len(result) < 5:
            result = ' '+result
        return result

    def debug_bank(self, i):
        format_value = self.model.bank_str
        if i<self.model.config.N:
            result =f"bank#{i} C={format_value(i,"C")} L={format_value(i,"L")} R={format_value(i,"R")} |" +\
                    f" D={format_value(i,"D")} E={format_value(i,"E")} "
            if self.model.incrD[i]>0:
                result +=(f" d={format_value(i,"d")} "
                          f"{'lender=' + str(self.model.lenders[i]) if self.model.lenders[i] >= 0 else ''}")
                if self.model.prob_bankruptcy[i]>=0:
                    result +=f" p={format_value(i,"prob_bankruptcy")} "
                if self.model.leverage[i]>=0:
                    result +=f" λ={format_value(i,"leverage")} "
                if self.model.haircut[i]>=0:
                    result +=f" h={format_value(i,"haircut")} "
                if self.model.capacity[i]>=0:
                    result +=f" c={format_value(i,"capacity")} "
                if self.model.interest_rate[i]>=0:
                    result +=f" r={format_value(i,"interest_rate")} "
            else:
                result +=f" s={format_value(i,"s")}"
                if self.model.psi[i]>=0:
                    result += f" ψ={format_value(i,"psi")}"
            return result
        else:
            return ""

    def debug_banks(self):
        for i in range(self.model.config.N):
            self.debug("------", self.debug_bank(i))

    @staticmethod
    def get_level(option):
        try:
            return getattr(logging, option.upper())
        except AttributeError:
            logging.error(" '--log' must contain a valid logging level and {} is not.".format(option.upper()))
            sys.exit(-1)

    def debug(self, module, text):
        if self.modules == [] or module in self.modules:
            if isinstance(text, list):
                for textline in text:
                    self.debug(module, textline)
            elif text:
                self.logger.debug('t={}/{} {}'.format(self.model.t, module, text))

    def info(self, module, text):
        if self.modules == [] or module in self.modules:
            if text:
                self.logger.info(' t={}/{} {}'.format(self.model.t, module, text))

    def error(self, module, text):
        if text:
            self.logger.error('t={}/{} {}'.format(self.model.t, module, text))

    def define_log(self, log: str, logfile: str = '', modules: str = ''):
        self.modules = modules.split(',') if modules else []
        formatter = logging.Formatter('%(levelname)s-' + '- %(message)s')
        self.logLevel = Log.get_level(log.upper())
        self.logger.setLevel(self.logLevel)
        if logfile:
            if not os.path.dirname(logfile):
                logfile = '{}/{}'.format(self.model.statistics.OUTPUT_DIRECTORY, logfile)
            fh = logging.FileHandler(logfile, 'a', 'utf-8')
            fh.setLevel(self.logLevel)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(self.logLevel)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

class Statistics:
    def __init__(self, in_model):
        self.stats_market = False
        self.graphs = {}
        self.graphs_pos = None
        self.plot_format = None
        self.graph_format = '.svg'
        self.output_format = '.gdt'
        self.create_gif = False
        self.OUTPUT_DIRECTORY = 'output'
        self.NUMBER_OF_ITEMS_IN_ANIMATED_GRAPH = 40
        self.correlation = []  # cross correlation of interest rate against bankruptcies
        self.model = in_model
        self.detailed_banks = []
        self.detailed_times = []
        self.statistics_stats_market = None
        # # this other only used if there is a graph associated with the lender change:
        self.grade_avg = []
        self.communities = []
        self.communities_not_alone = []
        self.gcs = []

    def reset(self, output_directory=None):
        if output_directory:
            self.OUTPUT_DIRECTORY = output_directory
        if not os.path.isdir(self.OUTPUT_DIRECTORY):
            os.mkdir(self.OUTPUT_DIRECTORY)

        self.best_lender = np.full(self.model.config.T, -1, dtype=int)
        self.best_lender_clients = np.full(self.model.config.T, -1, dtype=int)
        self.potential_credit_channels = np.zeros(self.model.config.T, dtype=int)
        self.active_borrowers = np.zeros(self.model.config.T, dtype=int)
        self.prob_bankruptcy = np.zeros(self.model.config.T, dtype=float)
        self.active_lenders = np.zeros(self.model.config.T, dtype=int)
        self.potential_lenders = np.zeros(self.model.config.T, dtype=int)
        self.c = np.zeros(self.model.config.T, dtype=float)
        self.demand = np.zeros(self.model.config.T, dtype=float)
        self.ir = np.zeros(self.model.config.T, dtype=float)
        self.ir_effective = np.zeros(self.model.config.T, dtype=float)
        self.asset_i = np.zeros(self.model.config.T, dtype=float)
        self.asset_j = np.zeros(self.model.config.T, dtype=float)
        self.equity = np.zeros(self.model.config.T, dtype=float)
        self.equity_lenders = np.zeros(self.model.config.T, dtype=float)
        self.max_e = np.zeros(self.model.config.T, dtype=float)
        self.max_lambda = np.zeros(self.model.config.T, dtype=float)
        self.var_deposits = np.zeros(self.model.config.T, dtype=float)
        self.var_dep_shock1 = np.zeros(self.model.config.T, dtype=float)
        self.var_dep_shock2 = np.zeros(self.model.config.T, dtype=float)
        self.liquidity = np.zeros(self.model.config.T, dtype=float)
        self.rationing = np.zeros(self.model.config.T, dtype=float)
        self.leverage = np.zeros(self.model.config.T, dtype=float)
        self.systemic_leverage = np.zeros(self.model.config.T, dtype=float)
        self.profits = np.zeros(self.model.config.T, dtype=float)
        self.B = np.zeros(self.model.config.T, dtype=float)
        self.sum_loans = np.zeros(self.model.config.T, dtype=float)
        self.num_loans = np.zeros(self.model.config.T, dtype=int)
        self.deposits = np.zeros(self.model.config.T, dtype=float)
        self.reserves = np.zeros(self.model.config.T, dtype=float)
        self.num_banks = np.zeros(self.model.config.T, dtype=int)
        self.bankruptcy = np.zeros(self.model.config.T, dtype=int)
        self.bankruptcy_rationed = np.zeros(self.model.config.T, dtype=int)
        self.num_of_rationed = np.zeros(self.model.config.T, dtype=int)
        self.psi = np.zeros(self.model.config.T, dtype=float)
        self.psi_effective = np.zeros(self.model.config.T, dtype=float)
        self.grade_avg = np.zeros(self.model.config.T, dtype=float)
        self.communities = np.zeros(self.model.config.T, dtype=int)
        self.communities_not_alone = np.zeros(self.model.config.T, dtype=int)
        self.gcs = np.zeros(self.model.config.T, dtype=int)

    def compute_credit_channels_and_best_lender(self):
        lenders = {}
        for bank in self.model.banks:
            if bank.lender is not None:
                if bank.lender in lenders:
                    lenders[bank.lender] += 1
                else:
                    lenders[bank.lender] = 1
        best = -1
        best_value = -1
        for lender in lenders.keys():
            if self.stats_market and not self.model.banks[lender].is_real_lender_or_borrower():
                continue
            if lenders[lender] > best_value:
                best = lender
                best_value = lenders[lender]
        self.best_lender[self.model.t] = best
        self.best_lender_clients[self.model.t] = best_value
        credit_channels = self.model.config.lender_change.get_credit_channels()
        if not self.stats_market:
            if credit_channels is None:
                self.potential_credit_channels[self.model.t] = len(self.model.banks)
            else:
                self.potential_credit_channels[self.model.t] = credit_channels
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_credit_channels_and_best_lender()

    def compute_statistics_of_graph(self):
        if self.model.config.lender_change.GRAPH_NAME:
            self.communities[self.model.t] = self.model.config.lender_change.determine_current_communities()
            self.communities_not_alone[self.model.t] = (
                self.model.config.lender_change.determine_current_communities_not_alone())
            self.gcs[self.model.t] = self.model.config.lender_change.determine_current_graph_gcs()
            self.grade_avg[self.model.t] = self.model.config.lender_change.determine_current_graph_grade_avg()
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_statistics_of_graph()

    def compute_ir_assets_psi_potential_lenders(self):
        asset_i = []
        asset_j = []
        ir = []
        potential_lenders = 0
        psi = []
        for bank in self.model.banks:
            if self.stats_market and not bank.is_real_lender_or_borrower():
                continue
            if bank.incrD >= 0:
                potential_lenders += 1
                if bank.get_loan_interest():
                    ir.append(bank.get_loan_interest())
                    self.compute_individual_banks_statistics(bank, 'ir', bank.get_loan_interest())
                if self.model.config.psi_endogenous:
                    self.compute_individual_banks_statistics(bank, 'psi', bank.psi)
                    psi.append(bank.psi)
                asset_i.append(bank.asset_i_avg_ir)
                self.compute_individual_banks_statistics(bank, 'asset_i', bank.asset_i_avg_ir)
            else:
                asset_j.append(bank.asset_j_avg_ir)
                self.compute_individual_banks_statistics(bank, 'asset_j', bank.asset_j_avg_ir)
        self.ir[self.model.t] = np.mean(ir) if ir else \
            (np.nan if self.stats_market else 0.0)
        self.asset_i[self.model.t] = np.mean(asset_i) if asset_i else 0.0
        self.asset_j[self.model.t] = np.mean(asset_j) if asset_j else 0.0
        self.potential_lenders[self.model.t] = potential_lenders
        if self.model.config.psi_endogenous:
            self.psi[self.model.t] = np.mean(psi) if psi else (np.nan if self.stats_market else 0.0)
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_ir_assets_psi_potential_lenders()

    def compute_loans(self):
        sum_loans = 0
        num_loans = 0
        ir_effective = []
        num_of_banks_that_are_lenders = 0
        num_of_banks_that_are_borrowers = 0
        equity_lenders = []
        psi_effective = []
        for bank in self.model.banks:
            if self.stats_market and not bank.is_real_lender_or_borrower():
                continue
            if bank.active_borrowers:
                if self.model.config.psi_endogenous:
                    self.compute_individual_banks_statistics(bank, 'psi_effective', bank.psi)
                    psi_effective.append(bank.psi)
                num_of_banks_that_are_lenders += 1
                this_bank_loans = 0
                this_bank_num_loans = 0
                for bank_that_is_borrower in bank.active_borrowers:
                    this_bank_loans += bank.active_borrowers[bank_that_is_borrower]
                    this_bank_num_loans += 1
                self.compute_individual_banks_statistics(bank, 'sum_loans', this_bank_loans)
                self.compute_individual_banks_statistics(bank, 'num_loans', this_bank_num_loans)
                sum_loans += this_bank_loans
                if this_bank_loans:
                    equity_lenders.append(bank.E)
                num_loans += this_bank_num_loans
            elif bank.l > 0:
                self.compute_individual_banks_statistics(bank, 'l', bank.l)
                num_of_banks_that_are_borrowers += 1
            if bank.get_loan_interest() and bank.s > 0:
                self.compute_individual_banks_statistics(bank, 'ir_effective', bank.get_loan_interest())
                ir_effective.append(bank.get_loan_interest())
        self.ir_effective[self.model.t] = np.mean(ir_effective) if ir_effective else \
            (np.nan if self.stats_market else 0.0)
        self.active_borrowers[self.model.t] = num_of_banks_that_are_borrowers
        self.active_lenders[self.model.t] = num_of_banks_that_are_lenders
        self.equity_lenders[self.model.t] = np.mean(equity_lenders) if equity_lenders else np.nan
        self.sum_loans[self.model.t] = sum_loans
        self.num_loans[self.model.t] = num_loans
        if self.model.config.psi_endogenous:
            self.psi_effective[self.model.t] = np.mean(psi_effective) if psi_effective\
                else (np.nan if self.stats_market else 0.0)
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_loans()

    def compute_leverage(self):
        leverage_of_lenders = []
        for bank in self.model.banks:
            if self.stats_market and not bank.is_real_lender_or_borrower():
                continue
            if not bank.failed:
                if bank.l == 0:
                    amount_of_loan = 0
                    if bank.get_lender() is not None and bank.get_lender().l > 0:
                        amount_of_loan = bank.get_lender().l
                    leverage_of_lenders.append(amount_of_loan / bank.E)
                    self.compute_individual_banks_statistics(bank, 'leverage', amount_of_loan / bank.E)

        self.leverage[self.model.t] = np.mean(leverage_of_lenders) if leverage_of_lenders else np.nan
        # systemic_leverage = how the system is in relation to the total population of banks (big value  of 10 borrowers
        # against a population of 100 banks means that there is a risk
        self.systemic_leverage[self.model.t] = sum(leverage_of_lenders) / len(self.model.banks) \
            if len(self.model.banks) > 0 else 0
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_equity()

    def compute_equity(self):
        sum_of_equity = 0
        for bank in self.model.banks:
            if self.stats_market and not bank.is_real_lender_or_borrower():
                continue
            if not bank.failed:
                sum_of_equity += bank.E
                self.compute_individual_banks_statistics(bank, 'equity', bank.E)
        self.equity[self.model.t] = sum_of_equity
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_equity()

    def compute_liquidity(self):
        total_liquidity = 0
        for bank in self.model.banks:
            if self.stats_market and not bank.is_real_lender_or_borrower():
                continue
            if not bank.failed:
                total_liquidity += bank.C
                self.compute_individual_banks_statistics(bank, 'liquidity', bank.C)
        self.liquidity[self.model.t] = total_liquidity
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_liquidity()

    def compute_fitness(self):
        self.fitness[self.model.t] = np.nan
        if self.model.config.N > 0:
            total_fitness = 0
            num_items = 0
            for bank in self.model.banks:
                if self.stats_market and not bank.is_real_lender_or_borrower():
                    continue
                if not bank.failed:
                    total_fitness += bank.mu
                    self.compute_individual_banks_statistics(bank, 'fitness', bank.mu)
                    num_items += 1
            self.fitness[self.model.t] = total_fitness / num_items if num_items > 0 else np.nan
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_fitness()

    def compute_policy(self):
        self.policy[self.model.t] = self.model.eta
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_policy()

    def compute_bad_debt(self):
        total_bad_debt = 0
        for bank in self.model.banks:
            if self.stats_market and not bank.is_real_lender_or_borrower():
                continue
            if bank.B:
                total_bad_debt += bank.B
                self.compute_individual_banks_statistics(bank, 'B', bank.B)
        self.B[self.model.t] = total_bad_debt
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_bad_debt()

    def compute_deposits_and_reserves(self):
        total_deposits = 0
        total_reserves = 0
        for bank in self.model.banks:
            if self.stats_market and not bank.is_real_lender_or_borrower():
                continue
            if not bank.failed:
                total_deposits += bank.D
                total_reserves += bank.R
                self.compute_individual_banks_statistics(bank, 'deposits', bank.D)
                self.compute_individual_banks_statistics(bank, 'reserves', bank.R)
        self.deposits[self.model.t] = total_deposits
        self.reserves[self.model.t] = total_reserves
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_deposits_and_reserves()

    def compute_probability_of_lender_change(self):
        if self.P is not None:
            probabilities_lc = []
            for bank in self.model.banks:
                if self.stats_market and not bank.is_real_lender_or_borrower():
                    continue
                probabilities_lc.append(bank.P)
            self.P[self.model.t] = np.mean(probabilities_lc) if probabilities_lc else np.nan
            self.P_max[self.model.t] = max(probabilities_lc) if probabilities_lc else np.nan
            self.P_min[self.model.t] = min(probabilities_lc)
            if self.statistics_stats_market:
                self.statistics_stats_market.compute_probability_of_lender_change()

    def compute_prob_bankruptcy_max_e_and_c(self):
        max_e = 0.0
        max_lambda = 0.0
        for bank in self.model.banks:
            if self.stats_market and not bank.is_real_lender_or_borrower():
                continue
            if bank.E > max_e:
                max_e = bank.E
            if bank.lambda_ > max_lambda:
                max_lambda = bank.lambda_
        probabilities_bankruptcy = []
        probabilities_lc = []
        lender_capacities = []
        demand_loan = 0
        for bank in self.model.banks:
            if self.stats_market and not bank.is_real_lender_or_borrower():
                continue
            if not bank.failed:
                if bank.d > 0:
                    demand_loan += bank.d
                    probabilities_bankruptcy.append(1 - bank.prob_surviving)
                else:
                    lender_capacities.append(np.mean(bank.c_avg_ir))
            if self.P is not None:
                probabilities_lc.append(bank.P)
        if type(self) is not StatisticsOnlyMarket:
            self.model.max_e = max_e
        self.prob_bankruptcy[self.model.t] = np.mean(probabilities_bankruptcy) if probabilities_bankruptcy else np.nan
        self.c[self.model.t] = np.mean(lender_capacities) if lender_capacities else np.nan
        self.demand[self.model.t] = demand_loan
        self.num_banks[self.model.t] = len(probabilities_bankruptcy)
        self.max_e[self.model.t] = max_e
        self.max_lambda[self.model.t] = max_lambda
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_prob_bankruptcy_max_e_and_c()

    def compute_individual_banks_statistics(self, bank, statistic, value):
        if bank.id in self.detailed_banks or self.model.t in self.detailed_times:
            self.detailed_banks_results.loc[self.model.t, statistic + f'_{bank.id}'] = value

    def compute_another_bankruptcy(self, bank, reason):
        self.bankruptcy[self.model.t] += 1
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_another_bankruptcy(bank, reason)
            self.compute_individual_banks_statistics(bank, 'bankruptcies', 1)

    def compute_increment_d(self, bank_incr_d, shock):
        self.var_deposits[self.model.t] += bank_incr_d
        if shock=="shock1":
            self.var_dep_shock1[self.model.t] += bank_incr_d
        else:
            self.var_dep_shock2[self.model.t] += bank_incr_d
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_increment_d(bank_incr_d, shock)

    def compute_profits(self, total_profits):
        self.profits[self.model.t] = total_profits
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_profits(total_profits)

    def compute_replaced_banks(self, num_banks_failed_rationed):
        self.bankruptcy_rationed[self.model.t] = num_banks_failed_rationed
        self.num_banks[self.model.t] = len(self.model.banks)
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_replaced_banks(num_banks_failed_rationed)

    def compute_rationed_rationing(self, num_of_rationed, amount_rationed):
        self.num_of_rationed[self.model.t] = num_of_rationed
        self.rationing[self.model.t] = amount_rationed
        if self.statistics_stats_market:
            self.statistics_stats_market.compute_rationed_rationing(num_of_rationed, amount_rationed)

    def get_cross_correlation_result(self, t):
        if t in [0, 1] and len(self.correlation) > t:
            status = '  '
            if self.correlation[t][0] > 0:
                if self.correlation[t][1] < 0.05:
                    status = '**'
                elif self.correlation[t][1] < 0.10:
                    status = '* '
            if t == 0:
                return (f'correl psi/ir {self.correlation[0][0]:4.2} '
                        f'p_value={self.correlation[0][1]:4.2} {status}')
            else:
                return (f'correl psi/liquidity {self.correlation[1][0]:4.2} '
                        f'p_value={self.correlation[1][1]:4.2} {status}')
        else:
            return " "

    def determine_cross_correlation(self):
        if not self.model.config.psi_endogenous:
            self.correlation = []
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", scipy.stats.ConstantInputWarning)
                    warnings.simplefilter("ignore", scipy.stats.NearConstantInputWarning)
                    self.correlation = [
                        # correlation_coefficient = [-1..1] and p_value < 0.10
                        scipy.stats.pearsonr(self.psi, self.ir),
                        scipy.stats.pearsonr(self.psi, self.liquidity),
                    ]
            except ValueError:
                self.correlation = []
        if self.statistics_stats_market:
            self.statistics_stats_market.determine_cross_correlation()


    def export_data(self, export_datafile=None, export_description=None, generate_plots=True):
        if export_datafile:
            self.save_data(export_datafile, export_description)
            if generate_plots:
                self.get_plots(export_datafile)
        if self.statistics_stats_market:
            self.statistics_stats_market.export_data(export_datafile, export_description, generate_plots)

        def draw(original_graph, new_guru_look_for=False, title=None, show=False):
            """ Draws the graph using a spring layout that reuses the previous one layout, to show similar position for
                the same ids of nodes along time. If the graph is undirected (Barabasi) then no """
            graph_to_draw = original_graph.copy()
            plt.clf()
            if title:
                plt.title(title)
            guru = None
            if self.node_positions is None:
                self.node_positions = nx.spring_layout(graph_to_draw, pos=self.node_positions)
            if not hasattr(original_graph, "type") and original_graph.is_directed():
                # guru should not have out edges, and surely by random graphs it has:
                guru, _ = self.find_guru(graph_to_draw)
                for (i, j) in list(graph_to_draw.out_edges(guru)):
                    graph_to_draw.remove_edge(i, j)
            if hasattr(original_graph, "type") and original_graph.type == "barabasi_albert":
                graph_to_draw, guru = self.get_graph_from_guru(graph_to_draw.to_undirected())
            if hasattr(original_graph, "type") and original_graph.type == "erdos_renyi" and graph_to_draw.is_directed():
                for node in list(graph_to_draw.nodes()):
                    if not graph_to_draw.edges(node) and not graph_to_draw.in_edges(node):
                        graph_to_draw.remove_node(node)
                new_guru_look_for = True
            if not self.node_colors or new_guru_look_for:
                self.node_colors = []
                guru, guru_node_edges = self.find_guru(graph_to_draw)
                for node in graph_to_draw.nodes():
                    if node == guru:
                        self.node_colors.append('darkorange')
                    elif self.__len_edges(graph_to_draw, node) == 0:
                        self.node_colors.append('lightblue')
                    elif self.__len_edges(graph_to_draw, node) == 1:
                        self.node_colors.append('steelblue')
                    else:
                        self.node_colors.append('royalblue')
            if hasattr(original_graph, "type") and original_graph.type == "barabasi_pref":
                nx.draw(graph_to_draw, pos=self.node_positions, node_color=self.node_colors, with_labels=True)
            else:
                nx.draw(graph_to_draw, pos=self.node_positions, node_color=self.node_colors, arrowstyle='->',
                        arrows=True, with_labels=True)

            if show:
                plt.show()
            return guru

    def get_graph(self, t):
        """
        Extracts from the model the graph that corresponds to the network in this instant
        """
        if 'unittest' in sys.modules.keys():
            return None
        else:
            self.graphs[t] = nx.DiGraph(directed=True)
            for bank in self.model.banks:
                if bank.lender is not None:
                    self.graphs[t].add_edge(bank.lender, bank.id)
            self.draw(self.graphs[t], new_guru_look_for=True, title='t={}'.format(t))
            if Utils.is_spyder():
                plt.show()
                filename = None
            else:
                filename = sys.argv[0] if self.model.export_datafile is None else self.model.export_datafile
                filename = self.get_export_path(filename, '_{}{}'.format(t, self.graph_format))
                plt.savefig(filename)
            plt.close()
            return filename

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
        if self.statistics_stats_market:
            self.statistics_stats_market.define_output_format(output_format)

    def create_gif_with_graphs(self, list_of_files):
        if len(list_of_files) == 0 or not self.create_gif:
            return
        else:
            if len(list_of_files) > self.NUMBER_OF_ITEMS_IN_ANIMATED_GRAPH:
                positions_of_images = len(list_of_files) / self.NUMBER_OF_ITEMS_IN_ANIMATED_GRAPH
            else:
                positions_of_images = 1
            filename_output = sys.argv[0] if self.model.export_datafile is None else self.model.export_datafile
            filename_output = self.get_export_path(filename_output, '.gif')
            images = []
            from PIL import Image
            for idx, image_file in enumerate(list_of_files):
                if not idx % positions_of_images == 0:
                    continue
                images.append(Image.open(image_file))
            images[0].save(fp=filename_output, format='GIF',
                           append_images=images[1:], save_all=True, duration=100, loop=0)

    def get_export_path(self, filename, ending_name=''):
        if not os.path.dirname(filename):
            filename = '{}/{}'.format(self.OUTPUT_DIRECTORY, filename)
        path, extension = os.path.splitext(filename)
        if ending_name:
            return path + ending_name
        else:
            return path + self.output_format.lower()

    def __generate_csv_or_txt(self, export_datafile, header, delimiter):
        file_header = ''
        for line_header in header:
            file_header += '# {}\n'.format(line_header)
        if self.stats_market:
            file_header += '# stats_market=1'
            export_datafile = export_datafile.replace('.txt', 'b.txt').replace('.csv', 'b.csv')
        file_header += "# pd.read_csv('file{}',header={}', delimiter='{}')\nt".format(self.output_format,
                                                                                      len(header) + 1, delimiter)
        with open(export_datafile, 'w', encoding='utf-8') as save_file:
            save_file.write(file_header)
            for element_name, _ in self.enumerate_statistics_results(stats_market=True):
                save_file.write('{}{}'.format(delimiter, element_name))
            save_file.write('\n')
            i_seq = 0
            for i in range(self.model.config.T):
                save_line = True
                line = '{}'.format(i_seq)
                for name_element, element in self.enumerate_statistics_results(stats_market=True):
                    if name_element == 'real_t':
                        line += '{}{}'.format(delimiter, i)
                    else:
                        line += '{}{}'.format(delimiter, element[i])
                        if self.stats_market and name_element == 'sum_loans' and element[i] == 0:
                            save_line = False
                            break
                if save_line:
                    i_seq += 1
                    save_file.write(line + '\n')

    def __generate_gdt_file(self, filename, enumerate_results, header, num_of_observations=None):
        element = lxml.builder.ElementMaker()
        gretl_data = element.gretldata
        xml_description = element.description
        xml_variables = element.variables
        variable = element.variable
        xml_observations = element.observations
        observation = element.obs
        num_variables = 0
        for variable_name, _ in enumerate_results(self.stats_market):
            if variable_name in self.model.config.ELEMENTS_STATISTICS_LOG:
                num_variables += 2
            else:
                num_variables += 1

        variables = xml_variables(count=f'{num_variables}')
        header_text = ''
        if self.stats_market:
            filename = filename.replace('.gdt', 'b.gdt')
            header_text = 'stats_market=1 '
        for item in header:
            header_text += item + ' '
        # header_text will be present as label in the first variable
        # correlation_result will be present as label in the second variable
        i = 1
        for variable_name, _ in enumerate_results(self.stats_market):
            if variable_name == 'leverage':
                variable_name += '_'
            if i == 1:
                variables.append(variable(name='{}'.format(variable_name), label='{}'.format(header_text)))
            elif i in [2, 3]:
                variables.append(variable(name='{}'.format(variable_name),
                                          label=self.get_cross_correlation_result(i - 2)))
            else:
                variables.append(variable(name='{}'.format(variable_name)))
            # if it is for example, equity, we add also the log as another extra variable:
            if variable_name in self.model.config.ELEMENTS_STATISTICS_LOG:
                variables.append(variable(name='l_{}'.format(variable_name),
                                          label=f"log of {variable_name}",
                                          parent=f"{variable_name}", transform="logs"))
            i = i + 1
        xml_observations = xml_observations(count='{}'.format(self.model.config.T), labels='false')
        num_obs_without_nans = 0
        if num_of_observations:
            range_of_observations = range(num_of_observations)
        else:
            range_of_observations = range(self.model.config.T)
        for i in range_of_observations:
            string_obs = ''
            save_instance = True
            for variable_name, variable in enumerate_results(self.stats_market):
                if variable_name == 'real_t':
                    string_obs += f'{i:3} '
                else:
                    string_obs += f'{variable[i]} '
                    if variable_name in self.model.config.ELEMENTS_STATISTICS_LOG:
                        try:
                            string_obs += f'{math.log(variable[i])} '
                        except ValueError:
                            string_obs += 'nan '
                    if self.stats_market and variable_name == 'sum_loans' and variable[i] == 0:
                        save_instance = False
                        break
            if save_instance:
                xml_observations.append(observation(string_obs))
                num_obs_without_nans += 1
        if num_obs_without_nans != self.model.config.T:
            xml_observations.set('count', str(num_obs_without_nans))
        gdt_result = gretl_data(xml_description(header_text), variables,
                                xml_observations, version='1.4', name='interbank',
                                frequency='special:1', startobs='1', endobs='{}'.format(self.model.config.T),
                                type='time-series')
        with gzip.open(filename, 'w') as output_file:
            output_file.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE gretldata SYSTEM "gretldata.dtd">\n')
            output_file.write(lxml.etree.tostring(gdt_result, encoding=str).encode('ascii'))

    def __generate_gdt(self, export_datafile, header):
        self.__generate_gdt_file(export_datafile, self.enumerate_statistics_results, header)

    def __enumerate_results_detailed_banks(self, stats_market):
        if type(self) is not StatisticsOnlyMarket and not stats_market:
            for variable_name in list(self.detailed_banks_results.columns):
                yield variable_name, self.detailed_banks_results[variable_name].tolist()
            yield 't', self.detailed_banks_results.index.to_list()

    def generate_detailed_banks(self, export_datafile):
        if not self.detailed_banks_results.empty and type(self) is not StatisticsOnlyMarket:
            if self.output_format in ('.csv', '.txt'):
                self.detailed_banks_results.to_csv(
                    export_datafile.replace('.csv', '_detailed.csv').replace('.txt', '_detailed.csv'), index=True)
            elif self.output_format == '.gdt':
                self.__generate_gdt_file(export_datafile.replace('.gdt', '_detailed.gdt'),
                                         self.__enumerate_results_detailed_banks, "",
                                         len(self.detailed_banks_results.index.to_list()))

    @staticmethod
    def __transform_line_from_string(line_with_values):
        items = []
        for i in line_with_values.replace('  ', ' ').strip().split(' '):
            try:
                items.append(int(i))
            except ValueError:
                if i != '':
                    items.append(float(i))
        return items

    @staticmethod
    def read_gdt(filename):
        configuration = []
        tree = lxml.etree.parse(filename)
        root = tree.getroot()
        children = root.getchildren()
        values = []
        columns = []
        columns_to_remove = []
        result = pd.DataFrame()
        if len(children) == 3:
            for variable in children[1].getchildren():
                column_name = variable.values()[0].strip()
                if column_name == 'leverage_':
                    column_name = 'leverage'
                columns.append(column_name)
                # as the value is later in a string as "123.23 2123.5 323.2 4166.6 52562.2", we add it
                # to columns_to_remove and we remove from the data later:
                if 'parent' in variable.keys():
                    columns_to_remove.append(column_name)
                if len(variable.values()) == 2 and configuration == []:
                    configuration = (variable.values()[1].
                                     replace(sys.argv[0] + ' ', '').
                                     replace(os.path.basename(sys.argv[0]) + ' ', '').
                                     split(' '))
            for value in children[2].getchildren():
                values.append(Statistics.__transform_line_from_string(value.text))

        if columns and values:
            result = pd.DataFrame(columns=columns, data=values)
            for column_to_remove in columns_to_remove:
                del result[column_to_remove]

        return result, configuration

    def save_data(self, export_datafile=None, export_description=None):
        if export_datafile:
            if export_description:
                header = ['{}'.format(export_description)]
            else:
                header = ['{} T={} N={}'.format(__name__, self.model.config.T, self.model.config.N)]
            if self.output_format.lower() == '.both':
                self.output_format = '.csv'
                self.__generate_csv_or_txt(self.get_export_path(export_datafile), header, ';')
                self.generate_detailed_banks(self.get_export_path(export_datafile))
                self.output_format = '.gdt'
                self.__generate_gdt(self.get_export_path(export_datafile), header)
                self.generate_detailed_banks(self.get_export_path(export_datafile))
            elif self.output_format.lower() == '.csv':
                self.__generate_csv_or_txt(self.get_export_path(export_datafile), header, ';')
                self.generate_detailed_banks(self.get_export_path(export_datafile))
            elif self.output_format.lower() == '.txt':
                self.__generate_csv_or_txt(self.get_export_path(export_datafile), header, '\t')
                self.generate_detailed_banks(self.get_export_path(export_datafile))
            else:
                self.__generate_gdt(self.get_export_path(export_datafile), header)
                self.generate_detailed_banks(self.get_export_path(export_datafile))

    def enumerate_statistics_results(self, stats_market=False):
        for element in dir(self):
            if isinstance(getattr(self, element), np.ndarray) and element not in Config.ELEMENTS_STATISTICS_NO_SHOW:
                if not (stats_market and self.get_name(element).endswith('_effective')):
                    yield self.get_name(element), getattr(self, element)
        if stats_market:
            yield 'real_t', 'real_t'
        if not stats_market and self.model.config.lender_change.GRAPH_NAME:
            for element in Config.ELEMENTS_STATISTICS_GRAPHS:
                yield self.get_name(element), getattr(self, element)

    def get_name(self, variable):
        try:
            return self.model.config.ELEMENTS_TRANSLATIONS[variable]
        except KeyError:
            return variable

    def unget_name(self, name):
        try:
            return {v: k for k, v in self.model.config.ELEMENTS_TRANSLATIONS.items()}[name]
        except KeyError:
            return name

    def get_data(self):
        result = pd.DataFrame()
        for variable_name, variable in self.enumerate_statistics_results():
            result[variable_name] = np.array(variable)
        return result.iloc[0:self.model.t]

    def plot_pygrace(self, xx, yy_s, variable, title, export_datafile, x_label, y_label):
        import pygrace.project
        plot = pygrace.project.Project()
        graph = plot.add_graph()
        graph.title.text = title.capitalize().replace('_', ' ')
        for yy, color, ticks, title_y in yy_s:
            data = []
            if isinstance(yy, tuple):
                for i in range(len(yy[0])):
                    data.append((yy[0][i], yy[1][i]))
            else:
                for i in range(len(xx)):
                    data.append((xx[i], yy[i]))
            dataset = graph.add_dataset(data, legend=title_y)
            dataset.symbol.fill_color = color
        graph.xaxis.label.text = x_label
        graph.yaxis.label.text = y_label
        graph.set_world_to_limits()
        graph.autoscale()
        if export_datafile:
            if self.plot_format:
                plot.saveall(self.get_export_path(export_datafile, '_{}{}'.format(variable.lower(), self.plot_format)))

    def plot_pyplot(self, xx, yy_s, variable, title, export_datafile, x_label, y_label):
        if self.plot_format == '.agr':
            self.plot_pygrace(xx, yy_s, variable, title, export_datafile, x_label, y_label)
        else:
            plt.clf()
            plt.figure(figsize=(14, 6))
            for yy, color, ticks, title_y in yy_s:
                if isinstance(yy, tuple):
                    plt.plot(yy[0], yy[1], ticks, color=color, label=title_y, linewidth=0.2)
                else:
                    plt.plot(xx, yy, ticks, color=color, label=title_y)
            plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            plt.title(title.capitalize().replace('_', ' '))
            if len(yy_s) > 1:
                plt.legend()
            if export_datafile:
                if self.plot_format:
                    plt.savefig(self.get_export_path(export_datafile,
                                                     '_{}{}'.format(variable.lower(), self.plot_format)))
            else:
                plt.show()
            plt.close()

    def plot_result(self, variable, title, export_datafile=None):
        xx = []
        yy = []
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(getattr(self, variable)[i])
        self.plot_pyplot(xx, [(yy, 'blue', '-', '')], variable, title, export_datafile, 'Time', '')

    def get_plots(self, export_datafile):
        for variable in dir(self):
            if isinstance(getattr(self, variable), np.ndarray) and variable not in Config.ELEMENTS_STATISTICS_NO_PLOT:
                if 'plot_{}'.format(variable) in dir(Statistics):
                    eval('self.plot_{}(export_datafile)'.format(variable))
                else:
                    self.plot_result(variable, self.get_name(variable), export_datafile)
        if self.model.config.lender_change.GRAPH_NAME:
            for variable in Config.ELEMENTS_STATISTICS_GRAPHS:
                if Config.ELEMENTS_STATISTICS_GRAPHS[variable]:
                    if 'plot_{}'.format(variable) in dir(Statistics):
                        eval('self.plot_{}(export_datafile)'.format(variable))
                    else:
                        self.plot_result(variable, self.get_name(variable), export_datafile)

    def plot_p(self, export_datafile=None):
        if self.P:
            xx = []
            yy = []
            yy_min = []
            yy_max = []
            yy_std = []
            for i in range(self.model.config.T):
                xx.append(i)
                yy.append(self.P[i])
                yy_min.append(self.P_min[i])
                yy_max.append(self.P_max[i])
            self.plot_pyplot(xx, [(yy, 'blue', '-', 'Avg prob with $\\gamma$'),
                                  (yy_min, 'cyan', ':', 'Max and min prob'), (yy_max, 'cyan', ':', '')],
                             'prob_change_lender',
                             'Prob of change lender ' + self.model.config.lender_change.describe(),
                             export_datafile, 'Time', '')

    def plot_num_banks(self, export_datafile=None):
        if not self.model.config.allow_replacement_of_bankrupted:
            self.plot_result('num_banks', self.get_name('num_banks'), export_datafile)

    def plot_best_lender(self, export_datafile=None):
        xx = []
        yy = []
        yy2 = []
        max_duration = 0
        final_best_lender = -1
        current_lender = -1
        current_duration = 0
        time_init = 0
        for i in range(self.model.config.T):
            xx.append(i)
            yy.append(self.best_lender[i] / self.model.config.N)
            yy2.append(self.best_lender_clients[i] / self.model.config.N)
            if current_lender != self.best_lender[i]:
                if max_duration < current_duration:
                    max_duration = current_duration
                    time_init = i - current_duration
                    final_best_lender = current_lender
                current_lender = self.best_lender[i]
                current_duration = 0
            else:
                current_duration += 1
        xx3 = []
        yy3 = []
        for i in range(time_init, time_init + max_duration):
            xx3.append(i)
            yy3.append(self.best_lender[i] / self.model.config.N)
        self.plot_pyplot(xx, [(yy, 'blue', '-', 'id'), (yy2, 'red', '-', 'Num clients'),
                              ((xx3, yy3), 'orange', '-', '')], 'best_lender',
                         'Best Lender (blue) #clients (red)', export_datafile,
                         'Time (best lender={} at t=[{}..{}])'.format(
                             final_best_lender, time_init, time_init + max_duration), 'Best Lender')

class GraphStatistics:
    @staticmethod
    def giant_component_size(graph):
        """weakly connected componentes of the directed graph using Tarjan's algorithm:
           https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm"""
        if graph.is_directed():
            return nx.average_clustering(graph)
        else:
            return len(max(nx.connected_components(graph), key=len))

    @staticmethod
    def get_all_credit_channels(graph):
        return graph.number_of_edges()

    @staticmethod
    def avg_clustering_coef(graph):
        """clustering coefficient 0..1, 1 for totally connected graphs, and 0 for totally isolated
           if ~0 then a small world"""
        try:
            return nx.average_clustering(graph, count_zeros=True)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def communities(graph):
        """Communities using greedy modularity maximization"""
        return list(nx.weakly_connected_components(graph if graph.is_directed() else graph.to_directed()))

    @staticmethod
    def grade_avg(graph):
        communities = GraphStatistics.communities(graph)
        total = 0
        for community in communities:
            total += len(community)
        return total / len(communities)

    @staticmethod
    def communities_not_alone(graph):
        """Number of communities that are not formed only with an isolated node"""
        total = 0
        for community in GraphStatistics.communities(graph):
            total += len(community) > 1
        return total

    @staticmethod
    def describe(graph, interact=False):
        if isinstance(graph, str):
            graph_name = graph
            try:
                graph = load_graph_json(graph_name)
            except FileNotFoundError:
                print("json file does not exist: %s" % graph)
                sys.exit(0)
            except (UnicodeDecodeError, json.decoder.JSONDecodeError) as e:
                print("json file does not contain a valid graph: %s" % graph)
                sys.exit(0)
        else:
            graph_name = '?'
        communities = GraphStatistics.communities(graph)
        string_result = f"giant={GraphStatistics.giant_component_size(graph)} " + \
                        f" comm_not_alone={GraphStatistics.communities_not_alone(graph)}" + \
                        f" comm={len(communities)}" + \
                        f" gcs={GraphStatistics.giant_component_size(graph)}"
        if interact:
            import code
            print(string_result)
            print(f"grade_avg={GraphStatistics.grade_avg(graph)}")
            print("communities=", list(GraphStatistics.communities(graph)))
            print(f"\n{graph_name} loaded into 'graph'\n")
            code.interact(local=locals())
            sys.exit(0)
        else:
            return string_result

class LenderChange:
    SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP = True
    GRAPH_NAME = "erdos_renyi"
    
    def __init__(self, model):
        self.p = model.config.p
        self.model = model
        self.node_positions = None
        self.node_colors = None

    def __len_edges(self, graph, node):
        if hasattr(graph, "in_edges"):
            return len(graph.in_edges(node))
        else:
            return len(graph.edges(node))

    def find_guru(self, graph):
        """It returns the guru ID and also a color_map with red for the guru, lightblue if weight<max/2 and blue others """
        guru_node = None
        guru_node_edges = 0
        for node in graph.nodes():
            edges_node = self.__len_edges(graph, node)
            if guru_node_edges < edges_node:
                guru_node_edges = edges_node
                guru_node = node
        return guru_node, guru_node_edges
    
    def __str__(self):
        return f"erdos_renyi p={self.p}"

    def __get_graph_from_guru(self, input_graph, output_graph, current_node, previous_node):
        """ It generates a new graph starting from the guru"""
        if self.__len_edges(input_graph, current_node) > 1:
            for (_, destination) in input_graph.edges(current_node):
                if destination != previous_node:
                    self.__get_graph_from_guru(input_graph, output_graph, destination, current_node)
        if previous_node is not None:
            output_graph.add_edge(current_node, previous_node)
            
    def get_graph_from_guru(self, input_graph):
        guru, _ = self.find_guru(input_graph)
        output_graph = nx.DiGraph()
        self.__get_graph_from_guru(input_graph, output_graph, guru, None)
        return output_graph, guru
    
    def draw(self, original_graph, new_guru_look_for=False, title=None, show=False):
        """ Draws the graph using a spring layout that reuses the previous one layout, to show similar position for
            the same ids of nodes along time. If the graph is undirected (Barabasi) then no """
        graph_to_draw = original_graph.copy()
        plt.clf()
        if title:
            plt.title(title)
        # if not self.node_positions:
        guru = None
        if self.node_positions is None:
            self.node_positions = nx.spring_layout(graph_to_draw, pos=self.node_positions)
        if not hasattr(original_graph, "type") and original_graph.is_directed():
            # guru should not have out edges, and surely by random graphs it has:
            guru, _ = self.find_guru(graph_to_draw)
            for (i, j) in list(graph_to_draw.out_edges(guru)):
                graph_to_draw.remove_edge(i, j)
        if hasattr(original_graph, "type") and original_graph.type == "barabasi_albert":
            graph_to_draw, guru = self.get_graph_from_guru(graph_to_draw.to_undirected())
        if hasattr(original_graph, "type") and original_graph.type == "erdos_renyi" and graph_to_draw.is_directed():
            for node in list(graph_to_draw.nodes()):
                if not graph_to_draw.edges(node) and not graph_to_draw.in_edges(node):
                    graph_to_draw.remove_node(node)
            new_guru_look_for = True
        if not self.node_colors or new_guru_look_for:
            self.node_colors = []
            guru, guru_node_edges = self.find_guru(graph_to_draw)
            for node in graph_to_draw.nodes():
                if node == guru:
                    self.node_colors.append('darkorange')
                elif self.__len_edges(graph_to_draw, node) == 0:
                    self.node_colors.append('lightblue')
                elif self.__len_edges(graph_to_draw, node) == 1:
                    self.node_colors.append('steelblue')
                else:
                    self.node_colors.append('royalblue')
        if hasattr(original_graph, "type") and original_graph.type == "barabasi_pref":
            nx.draw(graph_to_draw, pos=self.node_positions, node_color=self.node_colors, with_labels=True)
        else:
            nx.draw(graph_to_draw, pos=self.node_positions, node_color=self.node_colors, arrowstyle='->',
                    arrows=True, with_labels=True)
        if show:
            plt.show()
        return guru

    def save_graph_png(self, graph, description, filename, add_info=False):
        if add_info:
            if not description:
                description = ""
            description += " " + GraphStatistics.describe(graph)

        guru = self.draw(graph, new_guru_look_for=True, title=description)
        plt.rcParams.update({'font.size': 6})
        plt.rcParams.update(plt.rcParamsDefault)
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.savefig(filename)
        plt.close('all')
        return guru

    def save_graph_json(self, graph, filename):
        if graph:
            graph_json = nx.node_link_data(graph, edges="links")
            with open(filename, 'w') as f:
                json.dump(graph_json, f)

    def generate_banks_graph(self):
        result = nx.erdos_renyi_graph(n=self.model.config.N, p=self.model.config.p)
        return result, f"erdos_renyi p={self.model.config.p:5.3} {GraphStatistics.describe(result)}"

    #TODO: quitar self.model
    # quitar export_datafile
    def setup_links(self, save_graph=True):
        """ It creates a Erdos Renyi graph with p defined in parameter['p']. No changes in relationships before end"""
        self.banks_graph, description = self.generate_banks_graph()
        self.banks_graph.type = self.GRAPH_NAME

        if self.model.export_datafile and save_graph:
             filename_for_file = f"_{self.GRAPH_NAME}"
             if self.SAVE_THE_DIFFERENT_GRAPH_OF_EACH_STEP:
                 filename_for_file += f"_{self.model.t}"
             destination_file_json = self.model.statistics.get_export_path(self.model.export_datafile,
                                                                           f"{filename_for_file}.json")
             if not os.path.isfile(destination_file_json):
                 self.save_graph_json(self.banks_graph, destination_file_json)
                 self.save_graph_png(self.banks_graph, description, destination_file_json.replace('.json','.png'))

        self.model.log.debug("links ", f"{self.GRAPH_NAME} (x,y)=x could lends to y: "+str(self.banks_graph.edges()) if self.banks_graph.edges() else "no edges in graph")
        for i in range(self.model.config.N):
            if self.model.incrD[i]<0:
                possible_lenders = []
                for (_, j) in self.banks_graph.edges(i):
                    if self.model.incrD[j] > 0:
                        possible_lenders.append(j)
                if possible_lenders:
                    self.model.lenders[i] = random.choice(possible_lenders)
                else:
                    self.model.lenders[i] = -1
            else:
                # si es lender, no hay lender:
                self.model.lenders[i] = -1
        self.model.log.debug("links ", "lenders (-1=None): "+str(self.model.lenders))
        return self.banks_graph

    def determine_current_communities(self):
        return len(GraphStatistics.communities(self.banks_graph))

    def determine_current_communities_not_alone(self):
        return GraphStatistics.communities_not_alone(self.banks_graph)

    def determine_current_graph_gcs(self):
        return GraphStatistics.giant_component_size(self.banks_graph)

    def determine_current_graph_grade_avg(self):
        return GraphStatistics.grade_avg(self.banks_graph)


class Model:
    export_datafile = None

    def __init__(self):
        self.t = 0
        self.C = np.zeros(Config.N, dtype=float)
        self.D = np.zeros(Config.N, dtype=float)
        self.incrD = np.zeros(Config.N, dtype=float)
        self.E = np.zeros(Config.N, dtype=float)
        self.L = np.zeros(Config.N, dtype=float)
        self.R = np.zeros(Config.N, dtype=float)
        self.s = np.zeros(Config.N, dtype=float)
        self.d = np.zeros(Config.N, dtype=float)
        self.lenders = np.zeros(Config.N, dtype=int)
        self.prob_bankruptcy = np.zeros(Config.N, dtype=float)
        self.leverage = np.zeros(Config.N, dtype=float)
        self.haircut = np.zeros(Config.N, dtype=float)
        self.capacity = np.zeros(Config.N, dtype=float)
        self.interest_rate = np.zeros(Config.N, dtype=float)
        self.psi = np.zeros(Config.N, dtype=float)
        self.config = Config()
        self.lenderchange = LenderChange(self)
        self.stats = Statistics(self)
        self.log = Log(self)
        
    def bank(self, i, attr):
        matrix = getattr(self, attr)
        return matrix[i]

    def bank_str(self, i, attr):
        matrix = getattr(self, attr)
        return Log.format_number(matrix[i])

    def initialize_model(self):
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        for i in range(self.config.N):
            self.C[i] = self.config.C_i0
            self.L[i] = self.config.L_i0
            self.D[i] = self.config.D_i0
            self.E[i] = self.config.E_i0
            self.R[i] = self.config.r_i0 * self.D[i]
            self.C[i] = self.C[i] - self.R[i]

    def setup_links(self):
        self.lenderchange.setup_links(save_graph=False)

    def shock1(self):
        rand_values = np.random.rand(Config.N)
        shock = self.config.mu + self.config.omega * rand_values #  * ([10]*self.config.N)
        newD = self.D * shock
        self.incrD = newD - self.D
        newR = self.config.reserves * newD
        self.incrR = newR - self.R
        self.D = newD

        #TODO esto sería equivalente a lo posterior:
        #self.banks_C = np.where(self.banks_incrD>0, self.banks_C + self.banks_incrD - self.banks_incrR,
        #                        self.banks_D += self.banks_incrR)
        ##self.C += self.incrD - self.incrR
        # y ahora si C es <0, con una mácara np.where, sacar el dinero que les falta
        for i in range(Config.N):
            if self.incrD[i] >= 0:
                self.C[i] += self.incrD[i] - self.incrR[i]
                self.s[i] = self.C[i]
                self.d[i] = 0
                if self.incrD[i] > 0:
                    self.log.debug("shock1", '{} wins ΔD={:.4f}'.format(i, self.incrD[i]))
                else:
                    self.log.debug("shock1", '{} has no shock'.format(i))
            else:
                self.s[i] = 0
                if self.incrD[i] - self.incrR[i] + self.C[i] >= 0:
                    self.d[i] = 0
                    self.C[i] += self.incrD[i] - self.incrR[i]
                    self.log.debug("shock1", '{} loses ΔD={:.4f}, covered by capital'.
                                   format(i, self.incrD[i]))
                else:
                    self.d[i] = abs(self.incrD[i] - self.incrR[i] + self.C[i])
                    self.C[i] = 0
                    self.log.debug("shock1", '{} loses ΔD={:.4f} has C={:.4f} and needs {:.4f}'.format(
                        i, self.incrD[i], self.C[i], self.d[i]))
                    self.C[i] = 0

    def do_interest_rate(self):
        # 1. estimate probability of bankruptcy:
        #TODO se podria calcular la prob de bankruptcy para cada borrower distinto
        #  contando únicamente sus posibles lenders y así sería más heterogéneo
        max_e=0
        for i in range(self.config.N):
            if self.incrD[i]>0:
                if self.E[i]>max_e:
                    max_e=self.E[i]
        for i in range(self.config.N):
            if self.incrD[i]<0:
                self.prob_bankruptcy[i] = self.E[i] / max_e
            else:
                self.prob_bankruptcy[i] = -1.0
        # 2. leverage:
        max_leverage = 0
        for i in range(self.config.N):
            if self.incrD[i]<0:
                self.leverage[i] = self.d[i] / self.E[i]
                if self.leverage[i] > max_leverage:
                    max_leverage=self.leverage[i]
            else:
                self.leverage[i] = -1.0
        # 3. haircut:
        for i in range(self.config.N):
            if self.incrD[i]<0 and max_leverage:
                self.haircut[i] = self.leverage[i] / max_leverage
            else:
                self.haircut[i] = -1
        # 4. capacity:
        for i in range(self.config.N):
            if self.incrD[i]<0 and max_leverage:
                self.capacity[i] = (1 - self.haircut[i]) * self.d[i]
            else:
                self.capacity[i] = -1

        # 5. psi (market power of lenders):
        max_e_lenders = 0
        for i in range(self.config.N):
            if self.E[i]>max_e_lenders:
                max_e_lenders = self.E[i]
        for i in range(self.config.N):
            if self.incrD[i]>=0:
                self.psi[i] = self.E[i] / max_e_lenders
            else:
                self.psi[i] = -1

        # 6. interest_rate, for borrowers with its lender:
        for i in range(self.config.N):
            if self.d[i]>0 and self.lenders[i]>=0:
                psi = self.psi[self.lenders[i]]
                if psi==1:
                    psi=0.99
                self.interest_rate[i] = self.config.m / (self.prob_bankruptcy[i] * (1 - psi))
            else:
                self.interest_rate[i] = -1

    def run(self):
        self.initialize_model()
        self.log.debug_banks()
        for t in range(self.config.T):
            self.t=t
            self.shock1()
            self.setup_links()
            self.do_interest_rate()
            self.log.debug_banks()
            #for i in range(self.config.N):
            #    if self.capacity[i]>0 and (self.banks_d[i]-self.capacity[i])<=0:
            #        print(f"{i} c={self.capacity[i]} d={self.banks_d[i]}")


if __name__ == '__main__':
    model = Model()
    model.log.define_log('DEBUG', None, None)
    model.run()
