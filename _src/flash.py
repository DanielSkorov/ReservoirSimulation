import logging

from functools import (
  partial,
)

import numpy as np

from stability import (
  stabilityPT,
)

from rr import (
  solve2p_FGH,
)

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
  EosptType,
)


logger = logging.getLogger('flash')


class flash2pPT(object):
  """Two-phase flash calculations

  Performs two-phase flash calculations for isobaric-isothermal systems.

  Arguments
  ---------
  eos : EosptType
    An initialized instance of a PT-based equation of state.

  tol : numpy.float64
    Tolerance. When the norm of an equilibrium equations vector
    reduces below the tolerance, the system of non-linear equations
    is considered solved. The default is `tol = 1e-5`.

  Niter : int
    Maximum number of solver iterations. The default is `Niter = 50`.

  skip_stab : bool
    If `True` the algorithm will not perform the stability test, and
    initial guesses of k-values will be calculated by the method of
    an eos instance. The default is `False`.
  """
  def __init__(
    self,
    eos: EosptType,
    tol: ScalarType = np.float64(1e-5),
    Niter: int = 50,
    skip_stab: bool = False,
  ) -> None:
    self.eos = eos
    self.tol = tol
    self.Niter = Niter
    self.skip_stab = skip_stab
    self.stability = stabilityPT(eos)
    pass

  def run(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    method: str = 'qnss',
  ) -> tuple[VectorType, MatrixType, VectorType]:
    """Performs flash calculations for given pressure, temperature and
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

    Returns a tuple of a phase mole fractions array, an array of
    component mole fractions in each phase, and an array of phase
    compressibility factors.

    Raises `ValueError` if the solution of non-linear equations was not
    found for initial guesses of k-values.
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
  ) -> tuple[VectorType, MatrixType, VectorType]:
    raise NotImplementedError(
      'The BFGS-method for stability testing is not implemented yet.'
    )
    return (
      np.array([0., 0.]),
      np.zeros(shape=(self.eos.Nc, self.eos.Nc), dtype=yi.dtype),
      np.array([0., 0.]),
    )

  def _run_qnss(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, MatrixType, VectorType]:
    """QNSS-method for flash calculations

    Performs the quasi-Newton successive substitution (QNSS) method to
    find an equilibrium state by solving a system of non-linear equations.
    For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns a tuple of a phase mole fractions array, an array of
    component mole fractions in each phase, and an array of phase
    compressibility factors.

    Raises `ValueError` if the solution of non-linear equations was not
    found for initial guesses of k-values.
    """
    logger.debug(
      'Flash Calculation (QNSS-method)'
      '\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
      P, T, yi,
    )
    if self.skip_stab:
      kvji = self.eos.get_kvguess(P, T)
    else:
      stab, kvji, Z = self.stability._run_qnss(P, T, yi)
      if stab:
        return (
          np.array([1., 0.]),
          np.vstack([yi, np.zeros_like(yi)]),
          np.array([Z, 0.]),
        )
    plnphii = partial(self.eos.get_lnphii_Z, P=P, T=T)
    kvik: VectorType
    for i, kvik in enumerate(kvji):
      logger.debug('The kv-loop iteration number = %s', i)
      k = 0
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = plnphii(yi=yli)
      lnphivi, Zv = plnphii(yi=yvi)
      gi = np.log(kvik) + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tFv = %s\n\tlmbd = %s',
        0, kvik, gi, Fv, 1.,
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
        Fv = solve2p_FGH(kvik, yi)
        yli = yi / ((kvik - 1.) * Fv + 1.)
        yvi = yli * kvik
        lnphili, Zl = plnphii(yi=yli)
        lnphivi, Zv = plnphii(yi=yvi)
        gi = np.log(kvik) + lnphivi - lnphili
        gnorm = np.linalg.norm(gi)
        logger.debug(
          'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tFv = %s\n\tlmbd = %s',
          k, kvik, gi, Fv, lmbd,
        )
        if (gnorm < self.tol):
          break
        lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
        if lmbd > 30.:
          lmbd = 30.
      if k < self.Niter:
        rhol = yli.dot(self.eos.mwi) / Zl
        rhov = yvi.dot(self.eos.mwi) / Zv
        if rhov < rhol:
          return (
            np.array([Fv, 1. - Fv]),
            np.vstack([yvi, yli]),
            np.array([Zv, Zl]),
          )
        else:
          return (
            np.array([1. - Fv, Fv]),
            np.vstack([yli, yvi]),
            np.array([Zl, Zv]),
          )
    raise ValueError(
      "The solution of the equilibrium was not found. "
      "Try to increase the number of iterations or choose an another method."
    )


