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
    T: int = 1000  # time (1000)
    N: int = 50 # number of banks (50)

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
    #TODO 0.6 perfect simmetrical shock, 0.5 es eagerly negative shock    
    omega: float = 0.55 # omega ω   

    # banks initial parameters
    # L + C + R = D + E
    # but R = 0.02*D and C_i0= 5-0.18=4.82
    C_i0: float = 5  
    L_i0: float = 5
    # R_i0=0.18
    D_i0: float = 9     # deposits
    E_i0: float = 1     # equity
    r_i0: float = 0.02  # initial rate
    
    # maximum interest rate that can be applied to a loan (to avoid infinite rates when bankruptcy probability is 0 or psi=1)
    max_interest_rate: float = 1

    #TODO
    # if false when a bank dies it's not replaced:
    allow_replacement_of_bankrupted : bool = True
    allow_use_of_L_to_pay_rationing : bool = True

    def __init__(self, T:int=None, N:int=None, seed:int=None):
        if T:
            self.T = T
        if N:
            self.N = N
        if seed:
            self.seed = seed

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

    def __init__(self, T:int=None, N:int=None, seed:int=None):
        self.config = Config(T, N, seed)
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

    def set_bank(self, i, attr, value):
        matrix = getattr(self, attr)
        if value is not None:
            matrix[i] = value

    def bank_str(self, i, attr):
        matrix = getattr(self, attr)
        return Log.format_number(matrix[i])

    def init(self):
        self.t = 0
        self.C = np.zeros(self.config.N, dtype=float)
        self.bad_debt = np.zeros(self.config.N, dtype=float)
        self.failed = np.zeros(self.config.N, dtype=int)
        self.D = np.zeros(self.config.N, dtype=float)
        self.varD1 = np.zeros(self.config.N, dtype=float)
        self.varD2 = np.zeros(self.config.N, dtype=float)
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
            self.init_bank(i)
        self.stats.init()

    def finish(self):
        return self.stats.finish()

    def setup_links(self):
        self.lenderchange.setup_links(save_graph=False)

    def do_shock2(self, shock_values=None):
        # only when d>0 we will obtain a second shock:
        if shock_values is None:
            rand_values = np.random.rand(self.config.N)
            shock_values = self.config.mu + self.config.omega * rand_values
            newD = np.where(self.d > 0, self.D * shock_values, self.D)
            self.varD2 = newD - self.D
        else:
            self.varD2 = shock_values
            newD = self.D + self.varD2
        newR = self.config.reserves * newD
        varR = newR - self.R
        self.D = newD
        self.log.debug("shock2", f"ΔD={self.log.format_number(self.varD2)}")

        self.C += self.varD2 - varR
        self.R = newR


        # if C<0, then d=|C| -> NEW demand of loan (no lenders, because not satisfied) separated from .d
        # from shock1:
        self.d2 = np.where( self.C<0, abs(self.C), 0)
        # others, lenders:
        self.s2 = np.where( self.C>=0, self.C, 0)
        # and C<0 is impossible: those are zero now:
        self.C[ self.C<0 ] = 0
        self.R[ self.C<0 ] = 0


        # for i in range(self.config.N):
        #     if self.varD2[i]!=0:
        #         # only when it was a borrower we obtain a shock:
        #         newD = self.D[i]*shock_values[i]
        #         self.varD2[i] = newD - self.D[i]
        #         newR = self.config.reserves * self.D[i]
        #         varR = self.R[i] - newR
        #         self.D[i] = newD
        #         self.R[i] = newR
        #
        #         self.C[i] += self.varD2[i] - varR
        #         if self.varD2[i]>0:
        #             self.d[i] = self.varD2[i]
        #         else:
        #             self.d[i] = 0
        #     else:
        #         self.varD2[i] = 0
        self.log.debug("shock2", f"ΔD={self.log.format_number(self.varD2)}")

    def do_shock1(self, shock_values=None):
        if shock_values is None:
            rand_values = np.random.rand(self.config.N)
            shock_values = self.config.mu + self.config.omega * rand_values
            newD = self.D * shock_values
            self.varD1 = newD - self.D
        else:
            self.varD1 = shock_values
            newD = self.D + self.varD1
        newR = self.config.reserves * newD
        varR = newR - self.R
        self.D = newD
        self.log.debug("shock1", f"ΔD={self.log.format_number(self.varD1)}")

        self.C += self.varD1 - varR
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
        #TODO we can estimate prob bankruptcy for each borrower counting only its possible lenders
        #     more heterogeneous 

        # 1. probability of bankruptcy: for borrowers, we can estimate it as E/max(E of borrowers), for lenders is nan:
        borrowers_E = self.E[self.d>0]
        max_e_borrowers = np.nanmax(borrowers_E) if len(borrowers_E) > 0 and not np.isnan(borrowers_E).all() else 1.0
        self.prob_bankruptcy = np.where(self.d > 0, self.E / max_e_borrowers, np.nan)
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
        # max_leverage = 0
        # for i in range(self.config.N):
        #     if self.d[i]>0:
        #         self.leverage[i] = self.d[i] / self.E[i]
        #         if self.leverage[i] > max_leverage:
        #             max_leverage=self.leverage[i]
        #     else:
        #         self.leverage[i] = np.nan
        # 3. haircut:
        max_leverage = np.nanmax(self.leverage) if len(self.leverage) > 0 and not np.isnan(self.leverage).all() else 1.0
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
        lenders_E = self.E[self.s>0]
        if lenders_E.size > 0 and not np.isnan(lenders_E).all():
            max_e_lenders = np.nanmax(lenders_E)
            #self.psi[:] = 0 
            self.psi = np.where(self.s > 0, self.E / max_e_lenders, np.nan)
        else:
            self.psi[:] = 0
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
            if self.d[i] > 0 and self.lenders[i] != -1:
                psi = self.psi[self.lenders[i]]
                denominator = self.prob_bankruptcy[i] * (1 - psi)
                if denominator == 0 or np.isnan(denominator):
                    self.interest_rate[i] = self.config.max_interest_rate
                else:
                    self.interest_rate[i] = self.config.m / denominator
            else:
                self.interest_rate[i] = np.nan


    def do_loans(self):
        num_of_rationed = 0
        total_rationed = 0
        total_demanded = 0
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
                elif self.lenders[i] != -1 and self.s[self.lenders[i]] >= self.capacity[i]:
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
                elif self.lenders[i] != -1 and self.s[self.lenders[i]] < self.capacity[i]:
                    # borrower:
                    string_result += (f"s<c ({self.log.format_number(self.s[ self.lenders[i]])}>"
                                      f"{self.log.format_number(self.capacity[i])}) ")
                    self.l[i] = self.s[self.lenders[i]]
                    self.rationing[i] = self.d[i] - self.s[self.lenders[i]]
                    # lender:
                    self.C[self.lenders[i]] = 0
                    self.s[self.lenders[i]] = 0
                else:
                    self.l[i] = np.nan
                    self.rationing[i] = self.d[i]

                if self.rationing[i]>0:
                    total_rationed += self.rationing[i]
                    num_of_rationed += 1
                    string_result += f" rationed={self.log.format_number(self.rationing[i])},"
                if self.lenders[i] != -1:
                    self.log.debug("loans ", f"{string_result}l={self.capacity[i]},"
                                            f"lender.C={self.log.format_number(self.C[self.lenders[i]])},"
                                            f"lender.s={self.log.format_number(self.s[self.lenders[i]])}")
                else:
                    self.log.debug("loans ", f"{string_result}l=nan, lender=no lender")

        return num_of_rationed, total_rationed


    def check_if_bank_fails(self, bank, reason):
        if self.E[bank] < 0 or self.L[bank] < 0:
            self.failed[bank] = 1
            if self.l[bank] > 0 and self.lenders[bank] != -1:
                # firesaling process with ro=0 (no cost of liquidation). If there is also money to pay also
                # the interests, it is used as "excess" to pay them:
                amount_of_loan = self.l[bank]
                recovered_with_L = self.L[bank] if self.L[bank] > 0 else 0
                if amount_of_loan - recovered_with_L > 0:
                    bad_debt = amount_of_loan - recovered_with_L
                    interests = 0
                    paid_loan = recovered_with_L
                else:
                    bad_debt = 0
                    paid_loan = amount_of_loan
                    interests = min( self.interest_rate[bank] * amount_of_loan, recovered_with_L-amount_of_loan )
                self.log.debug("repayments", f"#{bank} uses L to pay {reason} and fails, "
                                             f"#{self.lenders[bank]} "
                                             f"bad_debt={self.log.format_number(bad_debt)},"
                                             f"interests={self.log.format_number(interests)},"
                                             f"paid={self.log.format_number(paid_loan)},")
                self.C[self.lenders[bank]] += paid_loan
                self.C[self.lenders[bank]] += interests
                self.E[self.lenders[bank]] += interests
                self.s[self.lenders[bank]] -= amount_of_loan
                self.bad_debt[self.lenders[bank]] += abs(bad_debt)
            return 1
        return 0

    def do_repayments(self):
        profits_paid = 0
        for i in range(self.config.N):
            # borrower it's rationed after shock1 ----------------------------------------------------------------------
            # - it didn't arrived to obtain enough in a loan o has no lender
            # - we pass the rationing to C (where we should have something... or it will fail directly)
            # - if C is not enough, we use L to cover the debt (and it affects to E in the same amount)
            if self.rationing[i] > 0:
                rationing_we_have = self.rationing[i]
                self.rationing[i] = 0
                self.C[i] -= rationing_we_have
                if self.C[i] >= 0:  # if C>0 we have been saved
                    self.log.debug("repayments", f"#{i} cancels "
                                              f"rationing={self.log.format_number(rationing_we_have)} and still has C="
                                              f"{self.log.format_number(self.C[i])}")
                else:
                    if self.config.allow_use_of_L_to_pay_rationing:
                        self.E[i] += self.C[i]
                        self.L[i] += self.C[i]
                        self.C[i] = 0
                        if self.check_if_bank_fails(i, "rationing"):
                            continue
                    else:
                        if self.check_if_bank_fails(i, "rationing"):
                            continue

            # if bank needs money after shock2 : d>0, we use C also, or L ----------------------------------------------
            if self.d2[i] > 0:
                amount_we_need = self.d2[i]
                self.d2[i] = 0
                self.C[i] -= amount_we_need
                if self.C[i] >= 0:
                    self.log.debug("repayments", f"#{i} cancels "
                                                 f"d={self.log.format_number(amount_we_need)} and still has "
                                                 f"{self.log.format_number(self.C[i])}")
                else: 
                    self.E[i] += self.C[i]
                    self.L[i] += self.C[i]
                    self.C[i] = 0
                    if self.check_if_bank_fails(i, "demand loan shock2"):
                        continue

            # now it's time to pay back the loan if there was one ------------------------------------------------------
            if self.l[i] > 0:
                amount_of_loan = self.l[i]
                self.C[i] -= amount_of_loan
                if self.C[i] < 0:
                    # we use L to pay back also the loan if not enough:
                    self.E[i] += self.C[i]
                    self.L[i] += self.C[i]
                    self.C[i] = 0
                    self.l[i] = 0
                    if self.check_if_bank_fails(i, "pay loan"):
                        continue

                # and the interests of the loan:
                interest_to_payback = self.interest_rate[i] * amount_of_loan
                self.C[i] -= interest_to_payback
                self.E[i] -= interest_to_payback
                if self.C[i] < 0:
                    # we use E to pay back also the loan if not enough:
                    self.L[i] += self.C[i]
                    self.E[i] += self.C[i]
                    self.C[i] = 0
                    if self.check_if_bank_fails(i, "pay interests"):
                        # interests_to_payback is greater than the L we had to use and partially paid:
                        really_paid = interest_to_payback - self.L[i]
                        self.C[self.lenders[i]] += really_paid
                        self.E[self.lenders[i]] += really_paid
                        profits_paid += really_paid
                        self.log.debug("repayments", f"#{i} pays partially loan+interests to #{self.lenders[i]} "
                                                     f"loan={self.log.format_number(amount_of_loan)} and "
                                                     f"interests={self.log.format_number(really_paid)} and fails")
                        continue
                    else:
                        # all the interest is paid to the lender:
                        self.C[self.lenders[i]] += interest_to_payback
                        self.E[self.lenders[i]] += interest_to_payback
                        profits_paid += interest_to_payback
                        self.log.debug("repayments", f"#{i} pays loan+interests to #{self.lenders[i]} "
                                                     f"loan={self.log.format_number(amount_of_loan)} and "
                                                     f"interests={self.log.format_number(interest_to_payback)}")
                self.check_if_bank_fails(i, "pay interests")
        return profits_paid


    def init_bank(self, i):
        self.C[i] = self.config.C_i0
        self.E[i] = self.config.E_i0
        self.D[i] = self.config.D_i0
        self.L[i] = self.config.L_i0
        self.R[i] = self.config.r_i0 * self.D[i]
        self.C[i] = self.C[i] - self.R[i]
        self.failed[i] = 0

    def replace_failed_banks(self):
        # surviving = self.failed != 1
        # num_surviving = np.sum(surviving)
        # if num_surviving == 0:
        #     return
        
        # def mode(data):
        #     unique_vals = np.unique(data)
        #     if len(unique_vals) == 0:
        #         return 0.0
        #     values, counts = np.unique(data, return_counts=True)
        #     return values[np.argmax(counts)]
        
        # mode_c = mode(self.C[surviving])
        # mode_d = mode(self.D[surviving])
        # mode_e = mode(self.E[surviving])
        
        for i in range(self.config.N):
            if self.failed[i] == 1:
                self.init_bank(i)


    def init_step(self, t):
        self.t = t
        self.l = np.zeros(self.config.N)
        self.bad_debt = np.zeros(self.config.N)
        
    def run(self):
        self.init()
        self.log.debug_banks()
        for t in range(self.config.T):
            self.init_step(t)
            self.do_shock1()
            self.stats.compute_d1()
            self.stats.compute_potential_lenders()
            self.setup_links()
            self.stats.compute_graph()
            self.do_interest_rate()
            self.stats.compute_psi()
            self.stats.compute_assets()
            self.stats.compute_ir()
            self.stats.compute_capacity()
            self.stats.compute_prob_bankruptcy()
            self.log.debug_banks()
            num_of_rationed, total_rationed = self.do_loans()
            self.stats.compute_rationing(num_of_rationed, total_rationed)
            self.stats.compute_num_loans()
            self.stats.compute_ir_avg()
            self.do_shock2()
            self.stats.compute_d2()
            self.log.debug_banks()
            profits_paid = self.do_repayments()
            self.stats.compute_profits(profits_paid)
            self.stats.compute_bankruptcies()
            self.replace_failed_banks()
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
        self.stats.define_output_directory(args.output)
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