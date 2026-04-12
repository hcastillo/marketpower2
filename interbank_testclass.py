import unittest
import interbank

class InterbankTest(unittest.TestCase):
    shocks = []
    lenders_list = []
    model = None

    def configureTest(self, N: int = None, T: int = None, seed=None):
        self.model = interbank.Model(T,N, seed=seed)
        self.model.log.define_log(log='DEBUG')
        self.model.log.interactive = False
        self.model.init()

    def setBank(self, bank: int, C: float = None, L: float = None, D: float = None, E: float = None, lender: int = None):
        D = D if (D is not None) else self.model.config.D_i0
        L = L if L is not None else self.model.config.L_i0
        E = E if E is not None else self.model.config.E_i0
        R = D*self.model.config.reserves
        C = (C if C is not None else self.model.config.C_i0) - R
        if C < 0:
            C = 0
            L -= R
        if L + C + R != D + E:
            E = L + C + R - D
            if E < 0:
                E = 0
            self.model.log.debug("******",
                                 f"{bank}  L+C must be equal to D+E => E modified to {E:.3f}")
        self.model.set_bank(bank, "C", C)
        self.model.set_bank(bank, "L", L)
        self.model.set_bank(bank, "D", D)
        self.model.set_bank(bank, "E", C)
        self.model.set_bank(bank, "R", R)
        if lender is not None:
            self.model.set_bank(bank, "lenders", lender)

    def assertBank(self, bank: int, C: float = None, L: float = None, R: float = None, D: float = None,
                   E: float = None, l: float = None, s: float = None, rationing: float = None,
                   bad_debt: float = None, failed: bool = False, lender: int = None):
        if L:
            self.assertEqual(self.model.bank(bank,"L"), L)
        if E:
            self.assertEqual(self.model.bank(bank,"E"), E)
        if C:
            self.assertEqual(self.model.bank(bank,"C"), C)
        if R:
            self.assertEqual(self.model.bank(bank,"R"), R)
        if D:
            self.assertEqual(self.model.bank(bank, "D"), D)
        if l:
            self.assertEqual(self.model.bank(bank, "l"), l)
        if s:
            self.assertEqual(self.model.bank(bank, "s"), s)
        if rationing:
            self.assertEqual(self.model.bank(bank, "rationing"), rationing)
        if bad_debt:
            self.assertEqual(self.model.bank(bank, "bad_debt"), bad_debt)
        if failed:
            self.assertEqual(self.model.bank(bank, "failed"), failed)
        if lender:
            self.assertEqual(self.model.bank(bank, "lender"), lender)
