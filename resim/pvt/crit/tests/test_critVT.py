import sys
import logging
import unittest

from numpy import (
  array as np_array,
)

from resim.pvt.eos import (
  pr78,
)

from resim.pvt.crit import (
  crit,
)


logger = logging.getLogger('crit')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class critVT(unittest.TestCase):

  def test_01(self):
    yi = np_array([0.014, 0.943, 0.027, 0.0074, 0.0049, 0.001, 0.0027])
    Pci = np_array([33.5, 45.4, 48.2, 41.9, 37.5, 33.3, 32.46]) * 101325.
    Tci = np_array([126.2, 190.6, 305.4, 369.8, 425.2, 469.6, 507.5])
    wi = np_array([0.04, 0.008, 0.098, 0.152, 0.193, 0.251, 0.27504])
    mwi = np_array([28.013, 16.043, 30.07, 44.097, 58.124, 72.151, 86.]) / 1e3
    dij = np_array([
      0.025,
      0.010, 0.,
      0.090, 0., 0.,
      0.095, 0., 0., 0.,
      0.110, 0., 0., 0., 0.,
      0.110, 0., 0., 0., 0., 0.,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, form='VT')
    state = crit.runVT(pr, yi)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
