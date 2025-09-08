import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('lab')
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

from lab import (
  ccePT,
)

from constants import (
  R,
)

from matplotlib import (
  pyplot as plt,
)


class cce(unittest.TestCase):

  def test_01(self):
    PP = np.array([19070., 16860., 15280., 13740., 13100., 12570., 11760.,
                   10020., 8010., 5980., 5010., 4230.]) * 1e3
    T = 13.8 + 273.15
    yi = np.array([0.078407, 0.862017, 0.033332, 0.013233, 0.006012, 0.004636,
                   0.001919, 0.000432, 1.2e-5])
    Pci = np.array([32.789, 45.4, 48.2, 41.9, 37.022, 33.008, 20.794, 16.727,
                    11.698]) * 101325.
    Tci = np.array([122.141, 190.6, 305.4, 369.8, 419.746, 482.781, 547.033,
                    674.011, 833.580])
    wi = np.array([0.02946, 0.008, 0.098, 0.152, 0.18767, 0.25456, 0.31931,
                   0.32841, 0.43005])
    vsi = np.array([-0.15670, -0.18633, -0.15018, -0.11403, -0.07789,
                    -0.02655, 0.01541, 0.10440, 0.22653])
    mwi = np.array([27.385, 16.043, 30.070, 44.097, 58.124, 77.671, 103.418,
                    185.177, 298.021]) / 1e3
    dij = np.array([
      0.00030,
      0.00035, 0.00566,
      0.00043, 0.01884, 0.00296,
      0.00121, 0.02884, 0.00874, 0.00087,
      0.00150, 0.04275, 0.01812, 0.00824, 0.00211,
      0.00176, 0.06061, 0.03113, 0.02758, 0.01329, 0.00564,
      0.00418, 0.15956, 0.10787, 0.07426, 0.04277, 0.02351, -0.00126,
      0.00630, 0.19705, 0.13549, 0.08406, 0.05104, 0.02627, -0.01284,-0.01087,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    cce = ccePT(pr, flashkwargs=dict(method='qnss-newton', useprev=True,
                                     stabkwargs=dict(method='qnss-newton')))
    res = cce.run(PP, T, yi)
    pass

  def test_02(self):
    PP = np.array([29120., 26360., 23820., 21940., 19850., 17340., 16920.,
                   16280., 15820., 15180., 14190., 13210., 11590., 10910.,
                   9350., 7200., 5120., 3750., 2290., 900.]) * 1e3
    T = 68. + 273.15
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np.array([0.012, 0.1, 0.152, 0.2, 0.414])
    vsi = np.array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    dij = np.array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    cce = ccePT(pr, flashkwargs=dict(method='qnss-newton', useprev=True,
                                     stabkwargs=dict(method='qnss-newton')))
    res = cce.run(PP, T, yi)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
