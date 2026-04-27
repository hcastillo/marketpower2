import sys
import unittest

import numpy as np

import interbank

class InterbankTest(unittest.TestCase):
    shocks = []
    lenders_list = []
    model = None

    def configureTest(self, N: int = None, T: int = None, seed=None, model_name=''):
        self.model = interbank.Model(T,N, seed=seed)
        self.model.log.define_log(log='DEBUG', model_name=model_name)
        self.model.log.interactive = False
        self.model.init()

    def setShock1(self, bank:int = None, shock: float = None, shocks: np.ndarray = None):
        if shocks is None:
            shocks = np.zeros(self.model.config.N)
            shocks[bank] = shock
        self.model.do_shock1(shocks)

    def setShock2(self, bank:int = None, shock: float = None, shocks: np.ndarray = None):
        if shocks is None:
            shocks = np.zeros(self.model.config.N)
            shocks[bank] = shock
        self.model.do_shock2(shocks)

    def setBank(self, bank: int, C: float = 0, L: float = 0, D: float = 0,
                E: float = 0, R: float = None, lender: int = -1, d:float = None, varD1: float = None, l: float = None,
                rationing: float = None, s: float = None, failed: bool = None, interest_rate: float = None,
                bad_debt: float = None):
        if R is None:
            if D is None:
                self.model.log.error("setBank", "#{bank} D must be set if you don't set R")
                sys.exit(-1)
            else:
                R = D * self.model.config.reserves

        s = 0 if s is None else s
        l = 0 if l is None else l
        rationing = 0 if rationing is None else rationing
        failed = False if failed is None else failed
        bad_debt = 0 if bad_debt is None else bad_debt

        if round(L + C + R + s, 8) != round(D + E + l + rationing, 8):
            self.model.log.error("setBank", f"#{bank} L+C+R+s must be equal to D+E+l+rationing: "
                                            f"L={L} C={C} R={R} s={s} ({L+C+R+s}) != ({D+E+l+rationing}) "
                                            f"D={D} E={E} l={l} rationing={rationing}")
            sys.exit(-1)
        self.model.set_bank(bank, "L", L)
        self.model.set_bank(bank, "C", C)
        self.model.set_bank(bank, "R", R)
        self.model.set_bank(bank, "s", s)
        self.model.set_bank(bank, "D", D)
        self.model.set_bank(bank, "E", E)
        self.model.set_bank(bank, "d", d)
        self.model.set_bank(bank, "interest_rate", interest_rate)
        self.model.set_bank(bank, "varD1", varD1)
        self.model.set_bank(bank, "rationing", rationing)
        self.model.set_bank(bank, "l", l)
        self.model.set_bank(bank, "lenders", lender)
        self.model.set_bank(bank, "bad_debt", bad_debt)
        self.model.set_bank(bank, "failed", failed)

    def assertBank(self, bank: int, C: float = None, L: float = None, R: float = None, D: float = None,
                   E: float = None, l: float = None, s: float = None, rationing: float = None,
                   bad_debt: float = None, failed: bool = False, lender: int = None):
        if L is not None:
            self.assertEqual(round(self.model.bank(bank,"L"),8), L)
        if E is not None:
            self.assertEqual(round(self.model.bank(bank,"E"),8), E)
        if C is not None:
            self.assertEqual(round(self.model.bank(bank,"C"),8), C)
        if R is not None:
            self.assertEqual(round(self.model.bank(bank,"R"),8), R)
        if D is not None:
            self.assertEqual(round(self.model.bank(bank, "D"),8), D)
        if l is not None:
            self.assertEqual(round(self.model.bank(bank, "l"),8), l)
        if s is not None:
            self.assertEqual(round(self.model.bank(bank, "s"),8), s)
        if rationing is not None:
            self.assertEqual(round(self.model.bank(bank, "rationing"),8), rationing)
        if bad_debt is not None:
            self.assertEqual(round(self.model.bank(bank, "bad_debt"),8), bad_debt)
        if failed is not None:
            self.assertEqual(round(self.model.bank(bank, "failed"),8), failed)
        if lender is not None:
            self.assertEqual(round(self.model.bank(bank, "lenders"),8), lender)
