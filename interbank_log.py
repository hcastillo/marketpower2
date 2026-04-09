#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Generates a simulation of an interbank network
# Usage: interbank.py --help
#
#
# author: hector@bith.net
# date:   04/2023, 09/2025, 03/2026
import os
import sys
import logging
import numpy as np
from progress.bar import Bar

class Log:
    """
    The class acts as a logger and helpers to represent the data and evol from the Model.
    """
    logger = logging.getLogger('interbank')
    modules = []
    model = None
    logLevel = 'DEBUG'
    progress_bar = None
    interactive = False

    def __init__(self, its_model):
        self.model = its_model

    def do_progress_bar(self, message, maximum):
        if self.interactive:
            self.progress_bar = Bar(message, max=maximum)

    def next(self):
        if self.progress_bar and self.interactive:
            self.progress_bar.next()

    @staticmethod
    def format_number(number):
        if isinstance(number, list) or isinstance(number, tuple) or isinstance(number, np.ndarray):
            return '[{}]'.format(','.join(Log.format_number(n) for n in number))
        else:
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
            if self.model.d[i]>0:
                result +=(f" d={format_value(i,"d")}")
                #          f"{'lender=' + str(self.model.lenders[i]) if self.model.lenders[i] >= 0 else ''}")
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
                if self.model.bad_debt[i]>=0:
                    result += f" ψ={format_value(i,"bad_debt")}"

            if self.model.failed[i]:
                result += f" ***failed***"
            return result.strip()
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

    def define_log(self, log: str, logfile: str = ''):
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



