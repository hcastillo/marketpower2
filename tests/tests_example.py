
import unittest
import interbank_testclass

class ValuesAfterExecutionTestCase(interbank_testclass.InterbankTest):

    def setUp(self):
        self.configureTest(N=3, T=2, seed=12)
        self.setBank(0, C=20, L=10, D=10, E=10)
        self.setBank(1, C=20, L=10, D=10, E=10)
        self.setBank(2, C=20, L=10, D=10, E=10)

    def test_values_after_execution(self):
        self.model.do_repayments()
        self.assertBank(0, C=19.8)

    def test_values_after_execution1(self):
        self.model.do_shock1()
        self.assertBank(0, C=139.8)

if __name__ == '__main__':
    unittest.main()
