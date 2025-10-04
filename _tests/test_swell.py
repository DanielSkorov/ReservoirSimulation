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
  swellPT,
  SimpleSeparator,
)


class swell(unittest.TestCase):

  def test_01(self):
    P0 = 22e6
    T = 92.2 + 273.15
    yi = np.array([0.0001, 0.4227, 0.1166, 0.1006, 0.0266, 0.1909, 0.1025,
                   0.0400])
    xi = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
    Fj = np.array([0., 0.1, 0.18, 0.31, 0.5, 0.7])
    Pci = np.array([72.8, 45.31, 52.85, 39.81, 33.35, 27.27, 17.63,
                    10.34]) * 101325.
    Tci = np.array([304.2, 190.1, 305.1, 391.3, 465.4, 594.5, 739.7, 886.2])
    wi = np.array([0.225, 0.0082, 0.13, 0.1666, 0.2401, 0.3708, 0.6151,
                   1.0501])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    mwi = np.array([44.01, 16.136, 33.585, 49.87, 72.151, 123.16, 225.88,
                    515.65]) / 1e3
    dij = np.array([
      0.103,
      0.130, 0.00153,
      0.135, 0.01114, 0.00446,
      0.125, 0.02076, 0.01118, 0.00154,
      0.150, 0.03847, 0.02512, 0.00862, 0.00290,
      0.150, 0.07040, 0.05245, 0.02728, 0.01610, 0.00542,
      0.103, 0.10800, 0.08640, 0.05402, 0.03812, 0.02049, 0.00495,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    sepg = SimpleSeparator(pr)
    sepo = SimpleSeparator(pr)
    sw = swellPT(pr, sepg, sepo)
    res = sw.run(P0, T, yi, xi, Fj)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
