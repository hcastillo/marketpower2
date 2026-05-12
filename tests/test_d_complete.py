
import unittest
import interbank_testclass

class ValuesAfterExecutionTestCase(interbank_testclass.InterbankTest):

    def setUp(self):
        self.configureTest(N=3, T=3, seed=12)

    def test_values_after_execution(self):
        # t=0 --------------------------------------
        #     no bank needs money (they cover all with their own cash or long-term assets)
        self.model.init_step(0)
        self.setBank(0, L=5, C=4.82, R=0.18, D=9, E=1, lender=None)
        self.setBank(1, L=5, C=4.82, R=0.18, D=9, E=1, lender=0)
        self.setBank(2, L=5, C=4.82, R=0.18, D=9, E=1, lender=0)
        self.setShock1(shocks=[4,-2,-4])
        self.assertBank(0, C=8.74, R=0.26, s=8.74)
        self.assertBank(1, C=2.86, R=0.14, d=0)
        self.assertBank(2, C=0.9, R=0.1, d=0)
        self.model.log.debug_banks()
        self.setShock2(shocks=[0,2,-1])
        self.model.do_repayments()        
        self.model.log.debug_banks()
        self.assertBank(0, C=8.74, R=0.26,D=13, s=8.74)
        self.assertBank(1, C=4.82, R=0.18,D=9, d=0)
        self.assertBank(2, C=0, L=4.92, R=0.08, D=4,d=0)  # no lender has given to #2 a loan to avoid selling L
        # t=1 --------------------------------------
        #     #1 and #2 need money from #0        
        self.model.init_step(1)
        self.setShock1(shocks=[2,-5,-2])
        self.model.log.debug_banks()
        self.assertBank(0, C=10.7, R=0.3,D=15, s=10.7)
        self.assertBank(1, C=0, R=0.08, D=4, d=0.08, L=5)
        self.assertBank(2, C=0, R=0.04, D=2, d=1.96, L=4.92)  # no lender has given to #2 a loan to avoid selling L
        #self.model.setup_links()
        self.model.do_interest_rate()
        self.model.log.debug_banks()
        self.setShock2(shocks=[0,1,-1.9])
        self.model.log.debug_banks()
        self.assertBank(0, C=10.7, R=0.3,D=15, s=10.7)
        self.assertBank(1, C=0.98, R=0.1, D=5, d=0.08, L=5)
        self.assertBank(2, C=0, R=0.002, D=0.1, d2=1.862, L=4.92)  
        self.model.do_repayments()
        self.model.log.debug_banks()
        self.assertBank(2, E=-0.942, failed=True)
        self.model.replace_failed_banks()
        self.model.log.debug_banks()
        self.setBank(2, L=5, C=4.82, R=0.18, D=9, E=1, lender=0)
        # t=2 --------------------------------------
        # finally #2 obtains a loan from #0 and fails again, with bad_debt
        #         #1 obtains no loan and rationed
        self.model.init_step(2)
        self.setShock1(shocks=[2,-2,-5])
        self.model.log.debug_banks()
        self.assertBank(0, C=12.66, s=12.66)
        self.assertBank(1, C=0, d=0.98)
        self.assertBank(2, C=0, d=0.08) 
        self.model.do_interest_rate()
        self.model.do_loans()
        self.assertBank(1, l=0.0, rationing=0.98)
        self.assertBank(2, l=0.07346939, rationing=0.00653061) 
        self.model.log.debug_banks()
        self.setShock2(shocks=[0,1,-1.9])
        self.model.log.debug_banks()
        self.model.do_repayments()
        self.model.log.debug_banks()
        self.assertBank(2, E=-0.86853061, failed=True)
        

if __name__ == '__main__':
    unittest.main()
