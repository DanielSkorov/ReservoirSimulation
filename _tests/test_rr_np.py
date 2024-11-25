import sys

sys.path.append('../_src/')

# import logging

# logger = logging.getLogger('rr')
# logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(process)d:%(name)s:%(levelname)s:\n\t%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

import unittest

import numpy as np
np.set_printoptions(linewidth=np.inf)

from rr import solveNp


class rr_solveNp(unittest.TestCase):

  def check_solution(self, Fj, kvji, yi, tol):
    Aji = 1. - kvji
    ti = 1. - Fj.dot(Aji)
    gj = Aji.dot(yi / ti)
    solved = np.linalg.norm(gj) < tol
    nfwindow = np.all(ti > 0.)
    return solved & nfwindow

  def test_case_1(self):
    yi = np.array(
      [0.204322076984, 0.070970999150, 0.267194323384, 0.296291964579, 0.067046080882, 0.062489248292, 0.031685306730]
    )
    kvji = np.array([
      [1.23466988745, 0.89727701141, 2.29525708098, 1.58954899888, 0.23349348597, 0.02038108640, 1.40715641002],
      [1.52713341421, 0.02456487977, 1.46348240453, 1.16090546194, 0.24166289908, 0.14815282572, 14.3128010831],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 6
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_2(self):
    yi = np.array([0.3, 0.4, 0.3])
    kvji = np.array([
      [2.64675, 1.16642, 1.25099E-03],
      [1.83256, 1.64847, 1.08723E-02],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 6
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_3(self):
    yi = np.array(
      [0.132266176697, 0.205357472415, 0.170087543100, 0.186151796211, 0.111333894738, 0.034955417168, 0.159847699672]
    )
    kvji = np.array([
      [26.3059904941, 1.91580344867, 1.42153325608, 3.21966622946, 0.22093634359, 0.01039336513, 19.4239894458],
      [66.7435876079, 1.26478653025, 0.94711004430, 3.94954222664, 0.35954341233, 0.09327536295, 12.0162990083],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 5
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_4(self):
    yi = np.array(
      [0.896646630194, 0.046757914522, 0.000021572890, 0.000026632729, 0.016499094171, 0.025646758089, 0.014401397406]
    )
    kvji = np.array([
      [1.64571122126, 1.91627717926, 0.71408616431, 0.28582415424, 0.04917567928, 0.00326226927, 0.00000570946],
      [1.61947897153, 2.65352105653, 0.68719907526, 0.18483049029, 0.01228448216, 0.00023212526, 0.00000003964],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 6
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_5(self):
    yi = np.array([0.08860, 0.81514, 0.09626])
    kvji = np.array([
      [0.112359551, 13.72549020, 3.389830508],
      [1.011235955, 0.980392157, 0.847457627],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 8
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_6(self):
    yi = np.array(
      [0.836206, 0.0115731, 0.0290914, 0.0324648, 0.0524046, 0.0258683, 0.0123914]
    )
    kvji = np.array([
      [1.46330454, 1.782453544, 0.953866131, 0.560800539, 0.142670434, 0.01174238, 0.000150252],
      [1.513154299, 2.490033379, 0.861916482, 0.323730849, 0.034794391, 0.000547609, 5.54587E-07],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 4
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_7(self):
    yi = np.array(
      [0.96, 0.011756, 0.004076, 0.00334, 0.001324, 0.004816, 0.006324, 0.003292, 0.002112, 0.001104, 0.001856]
    )
    kvji = np.array([
      [1.5678420840, 3.1505528290, 0.8312143829, 0.4373864613, 0.2335674319, 0.0705088530, 0.0204347990, 0.0039222104, 0.0004477899, 0.0000270002, 0.0000000592],
      [1.5582362780, 2.1752897740, 0.7919216342, 0.5232473203, 0.3492086459, 0.1630735937, 0.0742862374, 0.0263779361, 0.0067511045, 0.0011477841, 0.0000237639]
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 7
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_8(self):
    yi = np.array(
      [0.854161, 0.000701, 0.023798, 0.005884, 0.004336, 0.000526, 0.004803, 0.002307, 0.003139, 0.004847, 0.026886, 0.023928, 0.018579, 0.014146, 0.008586, 0.003373]
    )
    kvji = np.array([
      [1.248111, 1.566922, 1.338555, 0.86434, 0.722684, 0.629019, 0.596295, 0.522358, 0.501303, 0.413242, 0.295002, 0.170873, 0.08982, 0.042797, 0.018732, 0.006573],
      [1.2793605, 3.5983187, 2.3058527, 0.8619853, 0.4989022, 0.3439748, 0.2907231, 0.1968906, 0.173585, 0.1085509, 0.0428112, 0.0094336, 0.0016732, 0.0002458, 3.4493E-05, 4.7093E-06],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 4
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_9(self):
    yi = np.array([0.5, 0.2227, 0.1402, 0.1016, 0.0355])
    kvji = np.array([
      [23.75308598, 0.410182283, 0.009451899, 5.20178E-05, 5.04359E-09],
      [28.57470741, 1.5525E-10, 7.7405E-18, 8.0401E-40, 3.8652E-75],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 5
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_10(self):
    yi = np.array([0.5, 0.2227, 0.1402, 0.1016, 0.0355])
    kvji = np.array([
      [23.24202564, 0.401865268, 0.00928203, 5.13213E-05, 5.01517E-09],
      [28.55840446, 1.55221E-10, 7.73713E-18, 8.03102E-40, 3.85651E-75],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 5
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_11(self):
    yi = np.array([0.5, 0.15, 0.1, 0.1, 0.15])
    kvji = np.array([
      [4.354749468, 5.514004477, 0.056259268, 0.000465736, 3.83337E-05],
      [9.545344719, 8.93018E-06, 4.53738E-18, 1.15826E-35, 1.07956E-49],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 5
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_12(self):
    yi = np.array(
      [0.06, 0.802688, 0.009702, 0.024388, 0.027216, 0.043932, 0.021686, 0.010388]
    )
    kvji = np.array([
      [3.208411, 1.460005, 1.73068, 0.951327, 0.572598, 0.152184, 0.013802, 0.000211],
      [2.259859, 1.531517, 2.892645, 0.814588, 0.243232, 0.016798, 0.000118, 4.27E-08],
      [644.0243063, 0.001829876, 1.51452E-05, 8.05299E-10, 5.65494E-17, 3.81673E-34, 7.23797E-56, 6.58807E-68],
    ])
    Fj0 = np.array([0.25, 0.25, 0.25])
    tol = np.float64(1e-6)
    Niter = 6
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass

  def test_case_13(self):
    yi = np.array(
      [0.0825, 0.7425, 0.051433, 0.017833, 0.014613, 0.005793, 0.02107, 0.027668, 1.44E-02, 9.24E-03, 0.00483, 0.00812]
    )
    kvji = np.array([
      [1.931794, 1.423945, 2.634586, 0.83815, 0.466729, 0.263498, 0.088209, 0.028575, 0.006425, 9.00E-04, 6.96E-05, 2.14E-07],
      [3.081192, 1.42312, 1.729091, 0.833745, 0.617263, 0.460452, 0.263964, 0.149291, 0.07112, 0.026702, 0.007351, 0.000352],
      [1018.249407, 0.001355733, 2.069E-06, 4.15121E-09, 4.95509E-12, 5.43433E-15, 5.89003E-23, 2.46836E-31, 6.01409E-45, 1.36417E-62, 4.87765E-84, 4.5625E-132],
    ])
    Fj0 = np.array([0.25, 0.25, 0.25])
    tol = np.float64(1e-6)
    Niter = 6
    Fj = solveNp(kvji, yi, Fj0, tol, Niter)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass


if __name__ == '__main__':
  unittest.main()
