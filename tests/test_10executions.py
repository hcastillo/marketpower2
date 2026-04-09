
import unittest
import interbank


class ValuesAfterExecutionTestCase(unittest.TestCase):

    CONFIG = 'C_i0=30 D_i0=135 E_i0=15 L_i0=120 N=50 T=50 alfa=0.1 allow_replacement_of_bankrupted=True beta=5 chi=0.015 mu=0.7 omega=0.55 phi=0.25 psi=0.3 r_i0=0.02 reintroduce_with_median=False reserves=0.02 rho=0.3 xi=0.3'
    SEED = 39393

    def setUp(self):
        self.model = interbank.Model()
        self.model.configure_json(self.CONFIG)
        self.model.configure(N=5, T=10, seed=self.SEED)
        self.model.log.define_log('DEBUG')
        self.model.log.debug_banks()
        self.model.simulate_full()
        self.model.log.debug_banks()

    def test_values_after_execution(self):
        self.assertEqual(self.model.banks[1].C,  3.9897946301528373)
        self.assertEqual(self.model.banks[3].D, 139.4668368264057)
        self.assertEqual(self.model.banks[4].E, 15)
        self.assertEqual(self.model.banks[0].A, 187.7118464413857)
        self.assertEqual(self.model.banks[0].R, 3.4542369288277133)
        self.assertEqual(self.model.banks[1].A, 113.19682677073224)
        self.assertEqual(self.model.banks[2].A, 137.06256246093136)
        self.assertEqual(self.model.banks[3].A, 160.72630303066524)
        self.assertEqual(self.model.banks[4].A, 183.35259626622235)
        self.assertEqual(self.model.banks[4].R, 3.367051925324447)
        self.assertEqual(self.model.banks[4].R, self.model.banks[4].D*self.model.config.reserves)


if __name__ == '__main__':
    unittest.main()
