
import unittest
import interbank_testclass

class ValuesAfterExecutionTestCase(interbank_testclass.InterbankTest):

    def setUp(self):
        self.configureTest(N=2, T=2, seed=12)

    def test_values_after_execution(self):
        self.setBank(0, C=0, L=5, D=3, E=1, l=1, rationing=0.06, interest_rate=0.1,
                        d=1, s=0, lender=1)
        self.setBank(1, C=1, L=5, D=4, E=3.08, s=1)
        self.model.log.debug_bank(0)
        self.setShock2(0,1.06)
        self.model.log.debug_bank(0)
        self.assertBank(0, C=1.0388, R=0.0812, D=4.06, E=1)
        self.model.do_repayments()
        self.model.log.debug_bank(0)
        self.assertBank(0, C=0, R=0.0812, D=4.06, l=0, L=4.8788, E=0.7788)


if __name__ == '__main__':
    unittest.main()
