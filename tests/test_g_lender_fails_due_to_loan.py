
import unittest
import interbank_testclass

class ValuesAfterExecutionTestCase(interbank_testclass.InterbankTest):

    # 2 banks:
    # - bank0 lender
    # - bank1 borrower and not returns the loan and makes fail bank0

    def setUp(self):
        self.configureTest(N=2, T=2, seed=12)
        self.model.config.reserves = 0.1
        self.model.config.seed = 5

    def test_lender_fails(self):
        self.setBank(0, C=8-0.09, L=2, D=9, E=1, R=0.09, d=0)
        self.setBank(1, C=6-0.1, L=5, R=0.1, D=10, E=1, lender=0)
        self.setShock1(bank=1, shock=-10)
        self.model.log.debug_banks()
        self.model.do_interest_rate()
        self.model.capacity[1] = 6
        self.model.do_loans()
        self.model.log.debug_banks()
        self.setShock2(bank=1, shock=-10)
        self.model.log.debug_banks()
        self.assertBank(0, C=1.1, L=2, loaned=6, failed=False)
        self.assertBank(1, C=0, L=5, l=6, failed=False)
        self.model.do_repayments()
        self.model.log.debug_banks()
        self.assertBank(0, bad_debt=6, failed=True)
        self.model.log.debug_banks()
        


if __name__ == '__main__':
    unittest.main()
