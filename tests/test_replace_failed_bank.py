
import unittest
import interbank_testclass

class ValuesAfterExecutionTestCase(interbank_testclass.InterbankTest):

    # 3 banks:
    # - bank0 fails
    # - bank1 C=5, L=5, D=5, E=5
    # - bank2 C=5, L=1, D=5, E=1
    # - bank3 C=1, L=5, D=5, E=1
    #
    # - bank0 reintroduced with C=5, D=5, E=1 and L=1-resrves

    def setUp(self):
        self.configureTest(N=4, T=2, seed=12)
        self.model.config.reserves = 0.1

    def test_values_after_execution(self):
        self.setBank(0, failed=True)
        self.setBank(1, C=5, L=4.5, D=5, E=5)
        self.setBank(2, C=5, L=0.5, D=5, E=1)
        self.setBank(3, C=1, L=4.5, D=5, E=1)
        self.model.log.debug_banks()
        self.model.replace_failed_banks()
        self.model.log.debug_banks()
        self.assertBank(0, C=5, D=5, E=1, L=0.5)


if __name__ == '__main__':
    unittest.main()
