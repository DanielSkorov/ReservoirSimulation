import sys

sys.path.append('../_src/')

# import logging

# logger = logging.getLogger('bound')
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

from boundary import (
  PsatPT,
)

class stabPT(unittest.TestCase):

  def test_case_1_qnss(self):
    T = np.float64(68. + 273.15)
    P0 = np.float64(20e6)
    yi = np.array([.15, .85])
    Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975187616520054]) * 1e5
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022016206046])
    mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np.array([0.012, 0.1, 0.152, 0.2, 0.4140120986563907])
    vsi = np.array([-0.10170260557203252, -0.07663378625969472, -0.04989848152935413, -0.021862381679215244, 0.09093859308144847])
    dij = np.array([
      0.11001720977588050,
      0.02452055777238352, -0.02955930182910269,
      -0.2823143295220143, -0.41699521361585425, -0.04403831090822906,
      0.11866741268970402, 0.073285797638933340, 0.068150538448136390, -0.3019779998672122,
    ])
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    Psat = PsatPT(pr, method='qnss')
    res = Psat.run(T, yi, P0=P0)
    self.assertTrue(res.success)
    pass


if __name__ == '__main__':
  unittest.main()
