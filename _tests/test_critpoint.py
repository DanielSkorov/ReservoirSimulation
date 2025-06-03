import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('bound')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
  '%(process)d:%(name)s:%(levelname)s:\n\t%(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

import unittest

import numpy as np
np.set_printoptions(linewidth=np.inf)

from eos import (
  pr78,
)

from boundary import (
  getVT_PcTc,
)

class critpoint(unittest.TestCase):

  def test_01(self):
    yi = np.array([0.014, 0.943, 0.027, 0.0074, 0.0049, 0.001, 0.0027])
    Pci = np.array([33.5, 45.4, 48.2, 41.9, 37.5, 33.3, 32.46]) * 101325.
    Tci = np.array([126.2, 190.6, 305.4, 369.8, 425.2, 469.6, 507.5])
    wi = np.array([0.04, 0.008, 0.098, 0.152, 0.193, 0.251, 0.27504])
    mwi = np.array([28.013, 16.043, 30.07, 44.097, 58.124, 72.151, 86.]) / 1e3
    vsi = np.zeros_like(yi)
    dij = np.array([
      0.025,
      0.010, 0.0,
      0.090, 0.0, 0.0,
      0.095, 0.0, 0.0, 0.0,
      0.110, 0.0, 0.0, 0.0, 0.0,
      0.110, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    Pc, Tc = getVT_PcTc(yi, pr, maxiter=5)
    dPc = np.abs(Pc / 1e6 - 5.861138)
    dTc = np.abs(Tc - 202.6193)
    self.assertTrue((dPc < 1e-3) & (dTc < 1e-3))
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
