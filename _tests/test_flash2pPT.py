import sys

sys.path.append('../_src/')

# import logging

# logger = logging.getLogger('flash')
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

from flash import (
  flash2pPT,
)


class flash(unittest.TestCase):

  def test_case_1_qnss(self):
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
    tol = 1e-5
    flash = flash2pPT(pr, flashmethod='qnss', stabmethod='qnss', tol=tol, maxiter=6)
    res = flash.run(P, T, yi)
    self.assertTrue((res.gnorm < tol) & (res.success))
    pass

  def test_case_1_ss(self):
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
    tol = 1e-5
    flash = flash2pPT(pr, flashmethod='ss', stabmethod='ss', tol=tol, maxiter=7)
    res = flash.run(P, T, yi)
    self.assertTrue((res.gnorm < tol) & (res.success))
    pass


if __name__ == '__main__':
  unittest.main()
