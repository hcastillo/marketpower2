
import unittest
import interbank_testclass

class ValuesAfterExecutionTestCase(interbank_testclass.InterbankTest):

    def test_negative_shock__of2(self):
        self.configureTest(N=2, T=2, seed=12, model_name='negative')
        self.setBank(0, C=0, L=5, D=3, E=1, d=-1.06, rationing=1.06, interest_rate=0.1,
                        s=0, lender=1)
        self.setBank(1, C=1, L=5, D=4, E=3.08, s=1)
        self.model.log.debug_bank(0)
        self.setShock2(0,-2)
        self.model.log.debug_bank(0)
        self.assertBank(0, C=0)
        self.model.do_repayments()
        self.model.log.debug_bank(0)
        self.assertBank(0, C=0, R=0.02, D=1, rationing=0, E=-0.06, failed=1)

    def test_positive_shock_of2(self):
        self.configureTest(N=2, T=2, seed=12, model_name='pos_2')
        self.model.log.debug("positive_shock_of2","start")
        self.setBank(0, C=0, L=5, D=3, E=1, rationing=1.06, interest_rate=0.1,
                        s=0, lender=1)
        self.setBank(1, C=1, L=5, D=4, E=3.08, s=1)
        self.model.log.debug_bank(0)
        self.setShock2(0,2)
        self.model.do_repayments()
        self.model.log.debug_bank(0)
        self.assertBank(0, C=0.9, R=0.1, D=5, rationing=0)

    def test_positive_shock_of1(self):
        self.configureTest(N=2, T=2, seed=12, model_name='pos_1')
        self.model.log.debug("positive_shock_of1","start")
        self.setBank(0, C=0, L=5, D=3, E=1, rationing=1.06, interest_rate=0.1,
                        s=0, lender=1)
        self.setBank(1, C=1, L=5, D=4, E=3.08, s=1)
        self.model.log.debug_bank(0)
        self.setShock2(0,1)
        self.model.do_repayments()
        self.model.log.debug_bank(0)
        print(self.model.failed)
        self.assertBank(0, C=0, R=0.08, D=4, rationing=0, failed=False)


if __name__ == '__main__':
    unittest.main()
