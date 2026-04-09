#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Generates a simulation of an interbank network
# Usage: interbank.py --help
#
#
# author: hector@bith.net
# date:   04/2023, 09/2025, 03/2026
import argparse
import json
import logging
import random
import re
import sys
import warnings

import numpy as np
import interbank_lenderchange as lc
from interbank_log import Log
from interbank_statistics import Statistics as Stats

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
    #phi: float = 0.025  # phi Φ
    #chi: float = 0.015  # chi Χ

    #xi: float = 0.3  # xi ξ liquidation cost of collateral
    #rho: float = 0.3  # rho ρ fire sale cost

    #beta: float = 5  # β beta intensity of breaking the connection (5)
    #alfa: float = 0.1  # α alfa below this level of E or D, we will bankrupt the bank

    # banks initial parameters
    # L + C + R = D + E
    # but R = 0.02*D and C_i0= 30-2.7=27.3 and R=2.7
    C_i0: float = 5  # capital BEFORE RESERVES ESTIMATION, after it will be 27.3
    L_i0: float = 5
    # R_i0=2.7
    D_i0: float = 9  # deposits
    E_i0: float = 1  # equity
    r_i0: float = 0.02  # initial rate

    # if false when a bank dies it's not replaced: TODO
    allow_replacement_of_bankrupted : bool = True
    
    def __str__(self, separator=''):
        description = sys.argv[0] if __name__ == '__main__' else ''
        for attr, value in self:
            if attr == '__firstlineno__':
                continue
            description += ' {}={}{}'.format(attr, value, separator)
        return description + ' '

    def __iter__(self):
        for attr in dir(self):
            value = getattr(self, attr)
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                yield attr, value

    def get_current_value(self, name_config):
        current_value = None
        try:
            current_value = getattr(self, name_config)
        except AttributeError:
            logging.error("Config has no '{}' parameter".format(name_config))
            sys.exit(-1)
        # if current_value is None, then we cannot guess which type is the previous value:
        if current_value is None:
            try:
                current_value = self.__annotations__[name_config]
            except KeyError:
                return False
            if current_value is int:
                return 0
            elif current_value is bool:
                return False
            else:
                return 0.0
        return current_value
    
    def define_values_from_args(self, config_list):
        if config_list:
            config_list.sort()
            for item in config_list:
                if item == '':
                    pass
                elif item == '?':
                    print(self.__str__(separator='\n'))
                    sys.exit(0)
                else:
                    try:
                        name_config, value_config = item.split('=')
                    except ValueError:
                        name_config, value_config = ('-', '-')
                        logging.error('A Config value should be passed as parameter=value: {}'.format(item))
                        sys.exit(-1)
                    current_value = self.get_current_value(name_config)
                    try:
                        if isinstance(current_value, bool):
                            if value_config.lower() in ('y', 'yes', 't', 'true', 'on', '1'):
                                setattr(self, name_config, True)
                            elif value_config.lower() in ('n', 'no', 'false', 'f', 'off', '0'):
                                setattr(self, name_config, False)
                        elif isinstance(current_value, int):
                            setattr(self, name_config, int(value_config))
                        elif isinstance(current_value, float):
                            setattr(self, name_config, float(value_config))
                        else:
                            setattr(self, name_config, float(value_config))
                    except ValueError:
                        print(current_value, type(current_value), isinstance(current_value, bool),
                              isinstance(current_value, int))
                        logging.error('Value given for {} is not valid: {}'.format(name_config, value_config))
                        sys.exit(-1)
                    if name_config == 'psi':
                        setattr(self, 'psi_endogenous', 'False')


class Model:
    """
        It contains the banks and has the logic to execute the simulation
            import interbank
            model = interbank.Model()
            model.configure(param=x)
            result = model.run()
    """
    export_datafile = None

    def __init__(self):
        self.config = Config()
        self.stats = Stats(self)
        self.log = Log(self)
        self.lenderchange = lc.LenderChange(self)
        self.init()

    def configure_json(self, json_string: str):
        json_string = (json_string.strip().
                       replace('=', ':').replace(' ', ', ').
                       replace('True', 'true').replace('False', 'false'))
        if not json_string.startswith('{'):
            json_string = '{' + json_string
        if not json_string.endswith('}'):
            json_string += '}'
        self.configure(**json.loads(re.sub('(?<=\\{|\\s)(\\w+)(?=\\s*:)', '"\\1"', json_string)))

    def configure(self, **configuration):
        for attribute in configuration:
            if hasattr(self.config, attribute):
                setattr(self.config, attribute, configuration[attribute])
            else:
                raise LookupError('attribute in config not found: %s ' % attribute)
        
    def bank(self, i, attr):
        matrix = getattr(self, attr)
        return matrix[i]

    def bank_str(self, i, attr):
        matrix = getattr(self, attr)
        return Log.format_number(matrix[i])

    def init(self):
        self.t = 0
        self.C = np.zeros(self.config.N, dtype=float)
        self.bad_debt = np.zeros(self.config.N, dtype=float)
        self.failed = np.zeros(self.config.N, dtype=int)
        self.D = np.zeros(self.config.N, dtype=float)
        self.varD = np.zeros(self.config.N, dtype=float)
        self.E = np.zeros(self.config.N, dtype=float)
        self.L = np.zeros(self.config.N, dtype=float)
        self.R = np.zeros(self.config.N, dtype=float)
        self.s = np.zeros(self.config.N, dtype=float)
        self.d = np.zeros(self.config.N, dtype=float)
        self.lenders = np.zeros(self.config.N, dtype=int)
        self.l = np.zeros(self.config.N, dtype=float)
        self.rationing = np.zeros(self.config.N, dtype=float)
        self.prob_bankruptcy = np.zeros(self.config.N, dtype=float)
        self.leverage = np.zeros(self.config.N, dtype=float)
        self.haircut = np.zeros(self.config.N, dtype=float)
        self.capacity = np.zeros(self.config.N, dtype=float)
        self.interest_rate = np.zeros(self.config.N, dtype=float)
        self.psi = np.zeros(self.config.N, dtype=float)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="All-NaN axis encountered")
        self.log.do_progress_bar('Simulating t=0..{}'.format(self.config.T), self.config.T)
        for i in range(self.config.N):
            self.C[i] = self.config.C_i0
            self.L[i] = self.config.L_i0
            self.D[i] = self.config.D_i0
            self.E[i] = self.config.E_i0
            self.R[i] = self.config.r_i0 * self.D[i]
            self.C[i] = self.C[i] - self.R[i]
        self.stats.init()

    def finish(self):
        return self.stats.finish()

    def setup_links(self):
        self.lenderchange.setup_links(save_graph=False)

    def do_shock2(self):
        rand_values = np.random.rand(self.config.N)
        shock_values = self.config.mu + self.config.omega * rand_values
        newD = np.zeros(self.config.N, dtype=float)
        newR = np.zeros(self.config.N, dtype=float)
        #newD = np.where( self.d>0, self.D * shock_values, self.D)
        #self.varD = np.where( self.d>0, newD - self.D, 0)
        #newR = np.where(self.d>0, self.config.reserves * newD, self.R)
        #self.incrR = np.where( self.d>0, newR - self.R, 0)
        #self.D = newD
        for i in range(self.config.N):
            if self.d[i]>0:
                # only when it was a borrower we obtain a shock:
                newD[i] = self.D[i]*shock_values[i]
                self.varD[i] = newD[i] - self.D[i]
                newR[i] = self.config.reserves * self.D[i]
                self.varR[i] = self.R[i] - self.R[i]
                self.D[i] = newD[i]
                self.R[i] = newR[i]

                self.C[i] += self.varD[i] - self.varR[i]
            else:
                self.varD[i] = 0
                self.varR[i] = 0

        self.log.debug("shock2", f"ΔD={self.log.format_number(self.varD)}")

    def do_shock1(self):
        rand_values = np.random.rand(self.config.N)
        shock_values = self.config.mu + self.config.omega * rand_values
        newD = self.D * shock_values
        self.varD = newD - self.D
        newR = self.config.reserves * newD
        self.varR = newR - self.R
        self.D = newD
        self.log.debug("shock1", f"ΔD={self.log.format_number(self.varD)}")

        self.C += self.varD - self.varR
        self.R = newR
        # if C<0, then d=|C| -> borrowers
        self.d = np.where( self.C<0, abs(self.C), 0)
        # others, lenders:
        self.s = np.where( self.C>=0, self.C, 0)
        # and C<0 is impossible: those are zero now:
        self.C[ self.C<0 ] = 0
        self.R[ self.C<0 ] = 0
        # equivalent to:
        # for i in range(Config.N):
        #     if self.incrD[i] >= 0:
        #         self.C[i] += self.incrD[i] - self.incrR[i]
        #         self.s[i] = self.C[i]
        #         self.d[i] = 0
        #         if self.incrD[i] > 0:
        #             self.log.debug("shock1", '{} wins ΔD={:.4f}'.format(i, self.incrD[i]))
        #         else:
        #             self.log.debug("shock1", '{} has no shock'.format(i))
        #     else:
        #         self.s[i] = 0
        #         if self.incrD[i] - self.incrR[i] + self.C[i] >= 0:
        #             self.d[i] = 0
        #             self.C[i] += self.incrD[i] - self.incrR[i]
        #             self.log.debug("shock1", '{} loses ΔD={:.4f}, covered by capital'.
        #                            format(i, self.incrD[i]))
        #         else:
        #             self.d[i] = abs(self.incrD[i] - self.incrR[i] + self.C[i])
        #             self.C[i] = 0
        #             self.log.debug("shock1", '{} loses ΔD={:.4f} has C={:.4f} and needs {:.4f}'.format(
        #                 i, self.incrD[i], self.C[i], self.d[i]))
        #             self.C[i] = 0

    def do_interest_rate(self):
        #TODO se podria calcular la prob de bankruptcy para cada borrower distinto
        #     contando únicamente sus posibles lenders y así sería más heterogéneo
        # 1. probability of bankruptcy:
        max_e_borrowers = np.nanmax( self.E[self.d>0]) if len(self.E[self.d>0])>0 else np.nan
        self.prob_bankruptcy = np.where( self.d>0, self.E/ max_e_borrowers, np.nan)
        # max_e=0
        # for i in range(self.config.N):
        #     if self.d[i]>0:
        #         if self.E[i]>max_e:
        #             max_e=self.E[i]
        # for i in range(self.config.N):
        #     if self.d[i]>0:
        #         self.prob_bankruptcy[i] = self.E[i] / max_e
        #     else:
        #         self.prob_bankruptcy[i] = np.nan
        # 2. leverage:
        self.leverage = np.where( self.d>0, self.d / self.E, np.nan)
        max_leverage = np.nanmax(self.leverage) if len(self.leverage)>0 else np.nan
        # max_leverage = 0
        # for i in range(self.config.N):
        #     if self.d[i]>0:
        #         self.leverage[i] = self.d[i] / self.E[i]
        #         if self.leverage[i] > max_leverage:
        #             max_leverage=self.leverage[i]
        #     else:
        #         self.leverage[i] = np.nan
        # 3. haircut:
        self.haircut = np.where( self.d>0, self.leverage / max_leverage, np.nan)
        # for i in range(self.config.N):
        #     if self.d[i]>0 and max_leverage:
        #         self.haircut[i] = self.leverage[i] / max_leverage
        #     else:
        #         self.haircut[i] = np.nan
        # 4. capacity:
        self.capacity = np.where( self.d>0, (1 - self.haircut) * self.d, np.nan)
        # for i in range(self.config.N):
        #     if self.d[i]>0 and max_leverage:
        #         self.capacity[i] = (1 - self.haircut[i]) * self.d[i]
        #     else:
        #         self.capacity[i] = np.nan
        # 5. psi (market power of lenders):
        max_e_lenders = np.nanmax(self.E[self.s>0])
        self.psi = np.where( self.s>0, self.E / max_e_lenders, np.nan)
        # for i in range(self.config.N):
        #     if self.E[i]>max_e_lenders:
        #         max_e_lenders = self.E[i]
        # for i in range(self.config.N):
        #     if self.s[i]>0:
        #         self.psi[i] = self.E[i] / max_e_lenders
        #     else:
        #         self.psi[i] = np.nan
        # 6. interest_rate, for borrowers with its lender:
        for i in range(self.config.N):
            if self.d[i]>0 and self.lenders[i]>=0:
                psi = self.psi[self.lenders[i]]
                if psi==1:
                    psi=0.99
                self.interest_rate[i] = self.config.m / (self.prob_bankruptcy[i] * (1 - psi))
            else:
                self.interest_rate[i] = np.nan


    def do_loans(self):
        num_of_rationed = 0
        total_rationed = 0
        total_demanded = 0
        total_loans = 0
        for i in range(self.config.N):
            if self.d[i] > 0:
                total_demanded += self.d[i]
                # if no lender:
                string_result = f"#{i} "
                if self.lenders[i] ==-1:
                    self.l[i] = np.nan
                    self.rationing[i] = self.d[i]
                    total_rationed += self.rationing[i]
                    num_of_rationed += 1
                    string_result += f"rationed with no lender "
                elif self.s[ self.lenders[i]] >= self.capacity[i]:
                    # this includes d=c, and rationing will be 0
                    #
                    # borrower:
                    string_result += (f"s>=c ({self.log.format_number(self.s[ self.lenders[i]])}>"
                                      f"{self.log.format_number(self.capacity[i])}) ")
                    self.l[i] = self.capacity[i]
                    self.rationing[i] = self.d[i] -  self.capacity[i]
                    total_rationed += self.rationing[i]
                    num_of_rationed += 1
                    # lender:
                    self.C[ self.lenders[i]] -= self.capacity[i]
                    self.s[ self.lenders[i]] -= self.capacity[i]
                else: # if self.s[ self.lenders[i]] < self.capacity[i]:
                    # borrower:
                    string_result += (f"s<c ({self.log.format_number(self.s[ self.lenders[i]])}>"
                                      f"{self.log.format_number(self.capacity[i])}) ")
                    self.l[i] = self.s[ self.lenders[i]]
                    self.rationing[i] = self.d[i] - self.s[ self.lenders[i]]
                    # lender:
                    self.C[ self.lenders[i]] = 0
                    self.s[ self.lenders[i]] = 0

                if self.rationing[i]>0:
                    total_rationed += self.rationing[i]
                    num_of_rationed += 1
                    string_result += f" rationed={self.log.format_number(self.rationing[i])},"
                self.log.debug("loans ",f"{string_result}l={self.capacity[i]},"
                                        f"lender.C={self.log.format_number(self.C[ self.lenders[i]])},"
                                        f"lender.s={self.log.format_number(self.s[ self.lenders[i]])}")

        return num_of_rationed, total_rationed


    def do_repayments(self):
        for i in range(self.config.N):
            # bank was borrower: d>0
            # new shock is negative varD<0 or positive>0:
            if self.d[i] > 0:
                # borrower has not obtained enough loan and still ows money from shock1:
                if self.rationing[i] > 0:
                    self.C[i] -= self.rationing[i]
                    if self.C[i] > 0:
                        self.rationing[i] = 0
                        self.log.debug("repayments", f"#{i} cancels "
                                                     f"rationing={self.log.format_number(self.rationing[i])} and has "
                                                     f"still {self.log.format_number(self.C[i])}")
                    else:
                        self.E[i] += self.C[i]
                        self.C[i] = 0
                        if self.E[i] > 0:
                            self.log.debug("repayments", f"#{i} uses E to pay rationing, "
                                                         f"E={self.log.format_number(self.E[i])},C=0")
                        else:
                            self.log.debug("repayments", f"#{i} uses E to pay rationing but fails")
                            self.failed = 1
                            continue
                # now it's time to pay back the loan:
                if self.l[i] > 0:
                    self.C[i] -= self.l[i]
                    if self.C[i] < 0:
                        # we use E to pay back also the loan if not enough:
                        self.E[i] += self.C[i]
                        self.C[i] = 0
                        if self.E[i] < 0:
                            # if E<0, the <0 value means the amount not possible to cover:
                            self.failed = 1
                            self.bad_debt[ self.lenders[i] ] -= self.E[i]
                            self.C[self.lenders[i]] += self.l[i] + self.E[i]
                            self.s[self.lenders[i]] += self.l[i] + self.E[i]
                            self.log.debug("repayments", f"#{i} uses E to pay loan and fails, "
                                                         f"#{self.lenders[i]} "
                                                         f"bad_debt={self.log.format_number(-self.E[i])}")
                            continue
                        else:
                            # all the loan is paid:
                            self.C[self.lenders[i]] += self.l[i]
                            self.s[self.lenders[i]] += self.l[i]
                            self.log.debug("repayments", f"#{i} pays loan, #{self.lenders[i]}"
                                                         f" C+={self.log.format_number(self.l[i])}")

                    interest_to_payback = self.interest_rate[i] * self.l[i]
                    self.C[i] -= interest_to_payback
                    self.E[i] -= interest_to_payback
                    if self.C[i] < 0:
                        # we use E to pay back also the loan if not enough:
                        self.E[i] += self.C[i]
                        self.C[i] = 0
                        if self.E[i] < 0:
                            # if E<0, the <0 value means the amount not possible to cover:
                            self.failed = 0
                            amount_paid = interest_to_payback + self.E[i]
                            self.C[self.lenders[i]] += amount_paid
                            self.E[self.lenders[i]] += amount_paid
                            self.log.debug("repayments", f"#{i} uses E to pay interests and fails,"
                                                         f" C+={self.log.format_number(amount_paid)} and "
                                                         f"E+={self.log.format_number(amount_paid)}")
                            continue
                        else:
                            # all the interest is paid:
                            self.C[self.lenders[i]] += interest_to_payback
                            self.E[self.lenders[i]] += interest_to_payback
                            self.log.debug("repayments", f"#{i} pays interests, #{self.lenders[i]} "
                                                         f"C+={self.log.format_number(self.l[i])} and "
                                                         f"E+={self.log.format_number(self.l[i])}")

                        self.log.debug("repayments", f"#{i} cancels loan and has still {self.C[i]}")

    def init_step(self, t):
        self.t = t
        self.bad_debt = np.zeros(self.config.N)
        
    def run(self):
        self.init()
        self.log.debug_banks()
        for t in range(self.config.T):
            self.init_step(t)
            self.do_shock1()
            self.stats.compute_var_d1()
            self.setup_links()
            self.stats.compute_graph()
            self.stats.compute_potential_lenders()
            self.do_interest_rate()
            self.stats.compute_psi()
            self.stats.compute_assets()
            self.stats.compute_interest_rate()
            self.stats.compute_capacity()
            self.stats.compute_prob_bankruptcy()
            self.log.debug_banks()
            num_of_rationed, total_rationed = self.do_loans()
            self.stats.compute_lenders_and_borrowers()
            self.stats.compute_rationing(num_of_rationed, total_rationed)
            self.do_shock2()
            self.stats.compute_var_d2()
            self.log.debug_banks()
            self.do_repayments()
            self.stats.compute_liquidity()
            self.stats.compute_deposits()
            self.stats.compute_reserves()
            self.stats.compute_bad_debt()
            self.stats.compute_equity()
            self.log.debug_banks()
            self.log.next()
            #for i in range(self.config.N):
            #    if self.capacity[i]>0 and (self.banks_d[i]-self.capacity[i])<=0:
            #        print(f"{i} c={self.capacity[i]} d={self.banks_d[i]}")
        return self.finish()
    
    
    def run_interactive(self):
        """
            Interactively run the model
        """
        parser = argparse.ArgumentParser()
        parser.description = "<config=value> to set up Config options. use '?' to see values"
        parser.add_argument('--log', default='ERROR', help='Log level messages (ERROR,DEBUG,INFO...)')
        parser.add_argument('--logfile', default=None, help='File to send logs to')
        parser.add_argument('--save', default=None, help='Saves the output of this execution')
        parser.add_argument('--plot_format', type=str, default='none',
                            help='Generate plots with the specified format (svg,png,pdf,gif,agr)')
        parser.add_argument('--output_format', type=str, default='gdt',
                            help='File extension for data (gdt,txt,csv,both)')
        parser.add_argument('--output', type=str, default=None,
                            help='Directory where to store the results')
        parser.add_argument('--no_replace', action='store_true',
                            default=not self.config.allow_replacement_of_bankrupted,
                            help='No replace banks when they go bankrupted')
        args, other_possible_config_args = parser.parse_known_args()
        self.config.allow_replacement_of_bankrupted = not args.no_replace
        self.config.define_values_from_args(other_possible_config_args)
        self.log.define_log(args.log, args.logfile)
        self.stats.define_output_format(args.output_format)
        self.stats.define_output_file(args.save)
        self.stats.define_plot_format(args.plot_format)
        self.log.interactive = True
        self.run()

    @staticmethod
    def running_as_notebook():
        try:
            __IPYTHON__
            return get_ipython().__class__.__name__ != 'SpyderShell'
        except NameError:
            return False


model = Model()
if Model.running_as_notebook():
    model.stats.output_directory = '/content'
    model.stats.output_format = 'csv'
    model.run()
elif __name__ == '__main__':
    model.run_interactive()