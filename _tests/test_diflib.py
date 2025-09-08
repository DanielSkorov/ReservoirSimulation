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
  dlPT,
)

from constants import (
  R,
)

from matplotlib import (
  pyplot as plt,
)


class cvd(unittest.TestCase):

  def test_01(self):
    PP = np.array([18064.3, 16202.7, 14479.0, 12755.3, 11031.6, 9307.9,
                   7584.2, 5860.5, 4136.9, 2413.2, 1096.3, 101.3]) * 1e3
    T = 104.4 + 273.15
    yi = np.array([0.0091, 0.0016, 0.3647, 0.0967, 0.0695, 0.0144, 0.0393,
                   0.0144, 0.0141, 0.0433, 0.1320, 0.0757, 0.0510, 0.0315,
                   0.0427])
    Pci = np.array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np.array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np.array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0.])
    mwi = np.array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np.array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    dl = dlPT(pr, flashkwargs=dict(method='qnss-newton', useprev=True,
                                   stabkwargs=dict(method='qnss-newton')),
              psatkwargs=dict(method='newton-b',
                              stabkwargs=dict(method='qnss-newton')))
    res = dl.run(PP, T, yi, 18e6)
    pass

if __name__ == '__main__':
  unittest.main(verbosity=0)
