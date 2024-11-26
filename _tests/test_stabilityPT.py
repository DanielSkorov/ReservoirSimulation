import sys

sys.path.append('../_src/')

# import logging

# logger = logging.getLogger('stab')
# logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(process)d:%(name)s:%(levelname)s:\n\t%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

import unittest

import numpy as np

from eos import (
  pr78,
)

from stability import (
  stabilityPT,
)


class stabPT(unittest.TestCase):

  def test_case_1_qnss(self):
    P = np.float64(2e6)
    T = np.float64(40. + 273.15)
    yi = np.array([.15, .85])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr)
    is_stable, kv0, _ = stab.run(P, T, yi, method='qnss')
    self.assertTrue(is_stable)
    pass

  def test_case_2_qnss(self):
    P = np.float64(6e6)
    T = np.float64(10. + 273.15)
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr)
    is_stable, kv0, _ = stab.run(P, T, yi, method='qnss')
    self.assertFalse(is_stable)
    pass

  def test_case_3_qnss(self):
    P = np.float64(101325.)
    T = np.float64(20. + 273.15)
    yi = np.array([.1, .6, .3])
    Pci = np.array([4.600155e6, 3.2890095e6, 22.04832e6])
    Tci = np.array([190.6, 507.5, 647.3])
    wi = np.array([.008, .27504, .344])
    mwi = np.array([0.016043, 0.086, 0.018015])
    vsi = np.array([0., 0., 0.])
    dij = np.array([.0253, 0.4907, 0.48])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    stab = stabilityPT(pr, level=1)
    is_stable, kv0, _ = stab.run(P, T, yi, method='qnss')
    self.assertFalse(is_stable)
    pass


if __name__ == '__main__':
  unittest.main()

