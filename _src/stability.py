import logging

from functools import (
  partial,
)

import numpy as np

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
  EOSPTType,
)


logger = logging.getLogger('stab')


class stabilityPT(object):
  """Stability test based on the Gibbs energy analysis.

  Checks the tangent-plane distance (TPD) in local minima of
  the Gibbs energy function.

  Arguments
  ---------
  eos : EOSPTType
    An initialized instance of a PT-based equation of state.

  tol : numpy.float64
    Tolerance. When the norm of an equilibrium equations vector
    reduces below the tolerance, the system of non-linear equations
    is considered solved. The default is `tol = 1e-6`.

  Niter : int
    Maximum number of solver iterations. The default is `Niter = 50`.

  eps : numpy.float64
    The system will be considered unstable when `TPD < -eps`.
    The default value is `eps = 1e-4`.

  level : int
    Regulates a set of initial k-values obtained by the method
    `eos.getPT_kvguess(P, T, yi, level)`. The default is 0, which means
    that the most simple approache are used to generate initial k-values
    for the stability test.

  use_prev : bool
    Allows to preseve previous calculation results and to use them
    as the first initial guess for the next one if the solution is
    non-trivial.
  """
  def __init__(
    self,
    eos: EOSPTType,
    tol: ScalarType = np.float64(1e-6),
    Niter: int = 50,
    eps: ScalarType = np.float64(1e-4),
    level: int = 0,
    use_prev: bool = False
  ) -> None:
    self.eos = eos
    self.eps = -eps
    self.tol = tol
    self.Niter = Niter
    self.level = level
    self.use_prev = use_prev
    self.kvi_prev: None | VectorType = None
    pass

  def run(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    method: str = 'qnss',
  ) -> tuple[bool, MatrixType | None, ScalarType]:
    """Performs the stability test for given pressure, temperature and
    composition.

    Arguments
    ---------
      P : numpy.float64
          Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

      method : str
        The method that will be used to find local minima of the Gibbs
        energy function. The default is `'qnss'`.

    Returns a tuple of a boolean (True if a system is stable, False
    otherwise), an array of k-values at a local minima that can be
    used as an initial guess for flash calculations and the
    compressibility factor for given pressure, temperature, and
    composition.
    """
    if method == 'qnss':
      return self._run_qnss(P, T, yi)
    elif method == 'bfgs':
      return self._run_bfgs(P, T, yi)
    else:
      raise ValueError(f'The unknown method: {method}.')

  def _run_bfgs(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[bool, MatrixType | None, ScalarType]:
    raise NotImplementedError(
      'The BFGS-method for stability testing is not implemented yet.'
    )
    return True, None

  def _run_qnss(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[bool, MatrixType | None, ScalarType]:
    """QNSS-method for stability testing

    Performs the quasi-Newton successive substitution (QNSS) method to
    find local minima of the Gibbs energy function from different
    initial guesses. Calculates the TPD value at found local minima.
    For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns a tuple of a boolean (True if a system is stable, False
    otherwise), an array of k-values at a local minima that can be
    used as an initial guess for flash calculations and the
    compressibility factor for given pressure, temperature, and
    composition.
    """
    logger.debug(
      'Stability Test (QNSS-method)'
      '\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
      P, T, yi,
    )
    kvji = self.eos.getPT_kvguess(P, T, yi, self.level)
    if self.use_prev and self.preserved:
      kvji = np.vstack([self.kvi_prev, kvji])
    lnphii, Z = self.eos.getPT_lnphii_Z(P, T, yi)
    hi = lnphii + np.log(yi)
    plnphii = partial(self.eos.getPT_lnphii, P=P, T=T)
    kvik: VectorType
    for j, kvik in enumerate(kvji):
      k = 0
      ni = kvik * yi
      gi = np.log(ni) + plnphii(yi=ni/ni.sum()) - hi
      gnorm = np.linalg.norm(gi)
      logger.debug('The kv-loop iteration number = %s', j)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tTPD = %s\n\tlmbd = %s',
        0, kvik, gi, -np.log(ni.sum()), 1.,
      )
      lmbd = 1.
      while (gnorm > self.tol) & (k < self.Niter):
        dlnkvi = -lmbd * gi
        max_dlnkvi = np.abs(dlnkvi).max()
        if max_dlnkvi > 6.:
          relax = 6. / max_dlnkvi
          lmbd *= relax
          dlnkvi *= relax
        k += 1
        tkm1 = dlnkvi.dot(gi)
        kvik *= np.exp(dlnkvi)
        ni = kvik * yi
        gi = np.log(ni) + plnphii(yi=ni/ni.sum()) - hi
        gnorm = np.linalg.norm(gi)
        logger.debug(
          'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tTPD = %s\n\tlmbd = %s',
          k, kvik, gi, -np.log(ni.sum()), lmbd,
        )
        if (gnorm < self.tol):
          break
        lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
        if lmbd > 30.:
          lmbd = 30.
      ni = kvik * yi
      TPD = -np.log(ni.sum())
      if (TPD < self.eps) & (k < self.Niter):
        if self.use_prev:
          self.kvi_prev = kvi
          self.preserved = True
        logger.debug('TPD = %s\n\tThe system is unstable.\n', TPD)
        n = ni.sum()
        xi = ni / n
        # eta = (1. - self.eps) / n
        # kv0_flash = np.vstack([xi / yi, yi / xi]) * eta
        kv0_flash = np.vstack([xi / yi, yi / xi])
        return False, kv0_flash, Z
    else:
      logger.debug('TPD = %s\n\tThe system is stable.\n', TPD)
      return True, None, Z

