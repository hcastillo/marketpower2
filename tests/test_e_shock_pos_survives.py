
import unittest
import interbank_testclass

class ValuesAfterExecutionTestCase(interbank_testclass.InterbankTest):

    def test_negative_shock__of2(self):
        self.configureTest(N=2, T=2, seed=12)
        self.setBank(0, C=0, L=5, D=3, E=1, d=-1.06, rationing=1.06, interest_rate=0.1,
                        s=0, lender=1)
        self.setBank(1, C=1, L=5, D=4, E=3.08, s=1)
        self.model.log.debug_bank(0,"start")
        self.setShock2(0,1)
        self.assertBank(0, D=4, C=0.98, L=5, R=0.08, rationing=1.06, failed=False)
        self.model.log.debug_bank(0,"shock")
        self.model.do_repayments()
        self.model.log.debug_bank(0,"repay")
        self.assertBank(0, C=0, R=0.08, L=4.92, E=0.92, D=4, rationing=0, failed=False)

if __name__ == '__main__':
    unittest.main()
