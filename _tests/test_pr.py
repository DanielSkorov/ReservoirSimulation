import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('eos')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

import unittest

import numpy as np

np.set_printoptions(linewidth=np.inf)

from eos import (
  pr78,
)


class pr(unittest.TestCase):

  def test_01(self):
    P = 2e6
    T = 40. + 273.15
    yi = np.array([0.15, 0.85])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([0.225, 0.008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    lnfi, Z = pr.getPT_lnfi_Z(P, T, yi)
    Z_ = 0.95664027
    lnfi_ = np.array([12.52680951, 14.30933176])
    self.assertTrue(np.isclose(Z, Z_) & np.allclose(lnfi, lnfi_))
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
