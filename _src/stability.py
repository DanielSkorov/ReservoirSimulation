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


class StabResult(dict):
  """Container for stability test outputs with pretty-printing.

  Attributes
  ----------

  stable: bool
    A Boolean-flag indicating if a one-phase state is stable.

  TPD: float
    Tangent-plane distance at a local minima of a potential energy
    function.

  Z: float
    Compressibility factor of tested mixture at given pressure
    and temperature.

  yti: ndarray, shape (Nc,) | None
    Trial-phase composition if it was found. `Nc` is the number of
    components.

  kvji: tuple[ndarray] | None
    Initial guesses of k-values of `Nc` components for further
    flash calculations if the trial phase composition was found.

  gnorm: float
    Norm of a vector of equilibrium equations.

  success: bool
    Whether or not the procedure exited successfully.
  """
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self):
    with np.printoptions(linewidth=np.inf):
      s = (f"The one-phase state is stable:\n{self.stable}\n"
           f"Tangent-plane distance:\n{self.TPD}\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


class stabilityPT(object):
  """Stability test based on the Gibbs energy analysis.

  Checks the tangent-plane distance (TPD) at the local minima of
  the Gibbs energy function.

  Arguments
  ---------
  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

      - `getPT_kvguess(P, T, yi, level) -> tuple[ndarray]`
        This method is used to generate initial guesses of k-values.

      - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
        This `getPT_lnphii_Z` method must accept pressure,
        temperature and composition of a phase and returns a tuple of
        logarithms of the fugacity coefficients of components and the
        phase compressibility factor.

    If the solution method would be one of `'newton'` or `'ss-newton'`
    then it also must have:

      - `getPT_lnphii_Z_dnj(P, T, yi) -> tuple[ndarray, float, ndarray]`
        This method should return a tuple of logarithms of the fugacity
        coefficients, the mixture compressibility factor, and partial
        derivatives of logarithms of the fugacity coefficients with
        respect to components mole numbers which are an `ndarray` of
        shape `(Nc, Nc)`.

  method: str
    Type of solver. Should be one of:

      - `'ss'` (Successive Substitution method),
      - `'qnss'` (Quasi-Newton Successive Substitution),
      - `'bfgs'` (Currently raises `NotImplementedError`),
      - `'newton'` (Currently raises `NotImplementedError`),
      - `'ss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  eps: float
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-6`.

  maxiter: int
    Maximum number of solver iterations. Default is `50`.

  level: int
    Regulates a set of initial k-values obtained by the method
    `eos.getPT_kvguess(P, T, yi, level)`. Default is `0`, which means
    that the most simple approach is used to generate initial k-values
    for the stability test.

  useprev: bool
    Allows to preseve previous calculation results (if the solution is
    non-trivial) and to use them as the first initial guess in next run.
    Default is `False`.

  Methods
  -------
  run(P, T, yi) -> StabResult
    This method performs stability test procedure for given pressure
    `P: float` in [Pa], temperature `T: float` in [K], composition
    `yi: ndarray` of `Nc` components, and returns stability test
    results as an instance of `StabResult`.
  """
  def __init__(
    self,
    eos: EOSPTType,
    method: str = 'ss',
    eps: ScalarType = -1e-4,
    tol: ScalarType = 1e-6,
    maxiter: int = 50,
    level: int = 0,
    useprev: bool = False,
  ) -> None:
    self.eos = eos
    self.level = level
    self.useprev = useprev
    self.prevkvji: None | tuple[VectorType] = None
    self.preserved: bool = False
    if method == 'ss':
      self.stabsolver = partial(_stabPT_ss, eos=eos, eps=eps, tol=tol,
                                maxiter=maxiter)
    elif method == 'qnss':
      self.stabsolver = partial(_stabPT_qnss, eos=eos, eps=eps, tol=tol,
                                maxiter=maxiter)
    elif method == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for the stability test is not implemented yet.'
      )
    elif method == 'newton':
      raise NotImplementedError(
        "The Newton's method for the stability test is not implemented yet."
      )
    elif method == 'ss-newton':
      raise NotImplementedError(
        'The SS-Newton method for the stability test is not implemented yet.'
      )
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(self, P: ScalarType, T: ScalarType, yi: VectorType) -> StabResult:
    """Performs the stability test for given pressure, temperature and
    composition.

    Arguments
    ---------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    Stability test results as an instance of `StabResult`. Important
    properties are: `stab` a Boolean-flag indicating if a one-phase
    state is stable, `TPD` the tangent-plane distance at a local minima
    of the Gibbs energy function, `success` a Boolean flag indicating
    if the calculation completed successfully.
    """
    kvji0 = self.eos.getPT_kvguess(P, T, yi, self.level)
    if self.useprev and self.preserved:
      kvji0 = *self.prevkvji, *kvji0
    stab = self.stabsolver(P, T, yi, kvji0)
    if stab.success and self.useprev:
      self.prevkvji = stab.kvji
      self.preserved = True
    return stab


def _stabPT_ss(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  eps: ScalarType = -1e-4,
  tol: ScalarType = 1e-6,
  maxiter: int = 50,
) -> StabResult:
  """Successive Substitution method for stability testing.

  Arguments
  ---------
  P: float
    Pressure of a mixture [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[ndarray]
    Initial guesses for k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

      - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
        This `getPT_lnphii_Z` method must accept pressure,
        temperature and composition of a phase and returns a tuple of
        logarithms of the fugacity coefficients of components and the
        phase compressibility factor.

  eps: float
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-6`.

  maxiter: int
    Maximum number of solver iterations. Default is `50`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  properties are: `stab` a Boolean-flag indicating if a one-phase
  state is stable, `TPD` the tangent-plane distance at a local minima
  of the Gibbs energy function, `success` a Boolean flag indicating
  if the calculation completed successfully.
  """
  logger.debug(
    'Stability Test (SS-method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  hi = lnphiyi + np.log(yi)
  plnphii = partial(eos.getPT_lnphii_Z, P=P, T=T)
  j: int
  kvi0: VectorType
  for j, kvi0 in enumerate(kvji0):
    k: int = 0
    kvik = kvi0.flatten()
    ni = kvik * yi
    lnphixi, Zx = plnphii(yi=ni/ni.sum())
    gi = np.log(ni) + lnphixi - hi
    gnorm = np.linalg.norm(gi)
    logger.debug('The kv-loop iteration number = %s', j)
    TPD = -np.log(ni.sum())
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tTPD = %s',
      0, kvik, gi, TPD,
    )
    while (gnorm > tol) & (k < maxiter):
      k += 1
      kvik *= np.exp(-gi)
      ni = kvik * yi
      lnphixi, Zx = plnphii(yi=ni/ni.sum())
      gi = np.log(ni) + lnphixi - hi
      gnorm = np.linalg.norm(gi)
      TPD = -np.log(ni.sum())
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tTPD = %s',
        k, kvik, gi, TPD,
      )
    if (TPD < eps) & (k < maxiter) & (np.isfinite(kvik).all()):
      logger.debug('TPD = %s\n\tThe system is unstable.\n', TPD)
      n = ni.sum()
      xi = ni / n
      kvji = xi / yi, yi / xi
      return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                        Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                        success=True)
  else:
    logger.debug('TPD = %s\n\tThe system is stable.\n', TPD)
    return StabResult(stable=True, yti=None, kvji=None, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)


def _stabPT_qnss(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  eps: ScalarType = -1e-4,
  tol: ScalarType = 1e-6,
  maxiter: int = 50,
) -> StabResult:
  """QNSS-method for stability testing.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find local minimums of the Gibbs energy function from different
  initial guesses. Checks the TPD-value at a found local minima.
  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Arguments
  ---------
  P: float
    Pressure of a mixture [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[ndarray]
    Initial guesses for k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

      - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
        This `getPT_lnphii_Z` method must accept pressure,
        temperature and composition of a phase and returns a tuple of
        logarithms of the fugacity coefficients of components and the
        phase compressibility factor.

  eps: float
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-6`.

  maxiter: int
    Maximum number of solver iterations. Default is `50`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  properties are: `stab` a Boolean-flag indicating if a one-phase
  state is stable, `TPD` the tangent-plane distance at a local minima
  of the Gibbs energy function, `success` a Boolean flag indicating
  if the calculation completed successfully.
  """
  logger.debug(
    'Stability Test (QNSS-method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  hi = lnphiyi + np.log(yi)
  plnphii = partial(eos.getPT_lnphii_Z, P=P, T=T)
  j: int
  kvi0: VectorType
  for j, kvi0 in enumerate(kvji0):
    k: int = 0
    kvik = kvi0.flatten()
    ni = kvik * yi
    lnphixi, Zx = plnphii(yi=ni/ni.sum())
    gi = np.log(ni) + lnphixi - hi
    gnorm = np.linalg.norm(gi)
    logger.debug('The kv-loop iteration number = %s', j)
    TPD = -np.log(ni.sum())
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tTPD = %s\n\tlmbd = %s',
      0, kvik, gi, TPD, 1.,
    )
    lmbd: ScalarType = 1.
    while (gnorm > tol) & (k < maxiter):
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
      lnphixi, Zx = plnphii(yi=ni/ni.sum())
      gi = np.log(ni) + lnphixi - hi
      gnorm = np.linalg.norm(gi)
      TPD = -np.log(ni.sum())
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tTPD = %s\n\tlmbd = %s',
        k, kvik, gi, TPD, lmbd,
      )
      if (gnorm < tol):
        break
      lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if (TPD < eps) & (k < maxiter) & (np.isfinite(kvik).all()):
      logger.debug('TPD = %s\n\tThe system is unstable.\n', TPD)
      n = ni.sum()
      xi = ni / n
      kvji = xi / yi, yi / xi
      return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                        Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                        success=True)
  else:
    logger.debug('TPD = %s\n\tThe system is stable.\n', TPD)
    return StabResult(stable=True, yti=None, kvji=None, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)

