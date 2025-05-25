import logging

from functools import (
  partial,
)

import numpy as np

from typing import Callable

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
    A boolean flag indicating if a one-phase state is stable.

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
  def __getattr__(self, name: str) -> object:
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self) -> str:
    with np.printoptions(linewidth=np.inf):
      s = (f"The one-phase state is stable:\n{self.stable}\n"
           f"Tangent-plane distance:\n{self.TPD}\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


class stabilityPT(object):
  """Stability test based on the Gibbs energy analysis.

  Checks the tangent-plane distance (TPD) at the local minima of
  the Gibbs energy function.

  Parameters
  ----------
  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P, T, yi, level) -> tuple[ndarray]`
      This method is used to generate initial guesses of k-values.

    - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
      This `getPT_lnphii_Z` method must accept the pressure,
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
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Newton's method with preceding successive
      substitution iterations for initial guess improvement),
    - `'qnss-newton'` (Newton's method with preceding quasi-newton
      successive substitution iterations for initial guess improvement).

    Default is `'ss'`.

  level: int
    Regulates a set of initial k-values obtained by the method
    `eos.getPT_kvguess(P, T, yi, level)`. Default is `0`, which means
    that the most simple approach is used to generate initial k-values
    for the stability test.

  useprev: bool
    Allows to preseve previous calculation results (if the solution is
    non-trivial) and to use them as the first initial guess in next run.
    Default is `False`.

  kwargs: dict
    Other arguments for a stability test solver. It may contain such
    arguments as `tol`, `maxiter` or others, depending on the selected
    stability test solver.

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
    level: int = 0,
    useprev: bool = False,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.level = level
    self.useprev = useprev
    self.prevkvji: None | tuple[VectorType] = None
    self.preserved = False
    if method == 'ss':
      self.stabsolver = partial(_stabPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.stabsolver = partial(_stabPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.stabsolver = partial(_stabPT_newt, eos=eos, **kwargs)
    elif method == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for the stability test is not implemented yet.'
      )
    elif method == 'ss-newton':
      self.stabsolver = partial(_stabPT_ssnewt, eos=eos, **kwargs)
    elif method == 'qnss-newton':
      self.stabsolver = partial(_stabPT_qnssnewt, eos=eos, **kwargs)
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(self, P: ScalarType, T: ScalarType, yi: VectorType) -> StabResult:
    """Performs the stability test for given pressure, temperature and
    composition.

    Parameters
    ----------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    Stability test results as an instance of `StabResult`. Important
    properties are: `stab` a boolean flag indicating if a one-phase
    state is stable, `TPD` the tangent-plane distance at a local minima
    of the Gibbs energy function, `success` a boolean flag indicating
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
  """Successive Substitution (SS) method to perform the stability test
  using a PT-based equation of state.

  Parameters
  ----------
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
      This `getPT_lnphii_Z` method must accept the pressure,
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
  properties are: `stab` a boolean flag indicating if a one-phase
  state is stable, `TPD` the tangent-plane distance at a local minima
  of the Gibbs energy function, `success` a boolean flag indicating
  if the calculation completed successfully.
  """
  logger.debug(
    'Stability Test (SS-method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  hi = lnphiyi + np.log(yi)
  for j, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', j)
    k = 0
    kvik = kvi0.flatten()
    ni = kvik * yi
    lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
    gi = np.log(ni) + lnphixi - hi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s', k, kvik, gnorm,
    )
    while (gnorm > tol) & (k < maxiter):
      k += 1
      kvik *= np.exp(-gi)
      ni = kvik * yi
      lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
      gi = np.log(ni) + lnphixi - hi
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s', k, kvik, gnorm,
      )
    if (k < maxiter) & (np.isfinite(kvik).all()):
      TPD = -np.log(ni.sum())
      if (TPD < eps):
        logger.info(
          'The system is unstable at P = %s Pa, T = %s K, yi = %s:\n\t'
          'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
          P, T, yi, kvik, TPD, gnorm, k,
        )
        n = ni.sum()
        xi = ni / n
        kvji = xi / yi, yi / xi
        return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                          TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                          lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info(
      'The system is stable at P = %s Pa, T = %s K, yi = %s.\n\t'
      'The last kv-loop iteration results:\n\t'
      'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
      P, T, yi, kvik, TPD, gnorm, k,
    )
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
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
  """QNSS-method to perform the stability test using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find local minimums of the Gibbs energy function from different
  initial guesses. Checks the TPD-value at a found local minima.
  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
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
      This `getPT_lnphii_Z` method must accept the pressure,
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
  properties are: `stab` a boolean flag indicating if a one-phase
  state is stable, `TPD` the tangent-plane distance at a local minima
  of the Gibbs energy function, `success` a boolean flag indicating
  if the calculation completed successfully.
  """
  logger.debug(
    'Stability Test (QNSS-method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  hi = lnphiyi + np.log(yi)
  for j, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', j)
    k = 0
    kvik = kvi0.flatten()
    ni = kvik * yi
    lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
    gi = np.log(ni) + lnphixi - hi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tlmbd = %s',
      k, kvik, gnorm, 1.,
    )
    lmbd = 1.
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
      lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
      gi = np.log(ni) + lnphixi - hi
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tlmbd = %s',
        k, kvik, gnorm, lmbd,
      )
      if (gnorm < tol):
        break
      lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if (k < maxiter) & (np.isfinite(kvik).all()):
      TPD = -np.log(ni.sum())
      if (TPD < eps):
        logger.info(
          'The system is unstable at P = %s Pa, T = %s K, yi = %s:\n\t'
          'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
          P, T, yi, kvik, TPD, gnorm, k,
        )
        n = ni.sum()
        xi = ni / n
        kvji = xi / yi, yi / xi
        return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                          TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                          lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info(
      'The system is stable at P = %s Pa, T = %s K, yi = %s.\n\t'
      'The last kv-loop iteration results:\n\t'
      'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
      P, T, yi, kvik, TPD, gnorm, k,
    )
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)


def _stabPT_newt(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  eps: ScalarType = -1e-4,
  tol: ScalarType = 1e-6,
  maxiter: int = 20,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
) -> StabResult:
  """Newton's method for the stability test using a PT-based
  equation of state.

  Parameters
  ----------
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
      This `getPT_lnphii_Z` method must accept the pressure,
      temperature and composition of a phase and returns a tuple of
      logarithms of the fugacity coefficients of components and the
      phase compressibility factor.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      This `getPT_lnphii_Z_dnj` method must accept the pressure,
      temperature, phase composition and its mole number and returns a
      tuple of logarithms of the fugacity coefficients of components,
      the phase compressibility factor and the partial derivatives of
      logarithms of the fugacity coefficients of components with
      respect to component's mole numbers.

  eps: float
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: float
    Terminate the Newton's method successfully if the norm of the
    vector of equilibrium equations is less than `tol`. Default is
    `1e-6`.

  maxiter: int
    Maximum number of the Newton's method iterations. Default is `20`.

  linsolver: Callable[[ndarray, ndarray], ndarray]
    Function that accepts matrix A and vector b and finds vector x,
    which is the solution of the system of linear equations Ax = b.
    Default is `numpy.linalg.solve`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  properties are: `stab` a boolean flag indicating if a one-phase
  state is stable, `TPD` the tangent-plane distance at a local minima
  of the Gibbs energy function, `success` a boolean flag indicating
  if the calculation completed successfully.
  """
  logger.debug(
    "Stability Test (Newton's method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s",
    P, T, yi,
  )
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  hi = lnphiyi + np.log(yi)
  for j, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', j)
    k = 0
    kvik = kvi0.flatten()
    ni = kvik * yi
    sqrtni = np.sqrt(ni)
    alphaik = 2. * sqrtni
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
    gi = sqrtni * (np.log(ni) + lnphixi - hi)
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s',
      k, ni/yi, gnorm,
    )
    while (gnorm > tol) & (k < maxiter):
      H = np.diagflat(.5 * gi + 1.) + (sqrtni[:,None] * sqrtni) * dlnphixidnj
      dalphai = linsolver(H, -gi)
      k += 1
      alphaik += dalphai
      sqrtni = alphaik * .5
      ni = sqrtni * sqrtni
      n = ni.sum()
      xi = ni / n
      lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
      gi = sqrtni * (np.log(ni) + lnphixi - hi)
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s',
        k, ni/yi, gnorm,
      )
    if (gnorm < tol) & (np.isfinite(kvik).all()):
      TPD = -np.log(ni.sum())
      if (TPD < eps):
        logger.info(
          'The system is unstable at P = %s Pa, T = %s K, yi = %s:\n\t'
          'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
          P, T, yi, ni/yi, TPD, gnorm, k,
        )
        n = ni.sum()
        xi = ni / n
        kvji = xi / yi, yi / xi
        return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                          TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                          lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info(
      'The system is stable at P = %s Pa, T = %s K, yi = %s.\n\t'
      'The last kv-loop iteration results:\n\t'
      'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
      P, T, yi, ni/yi, TPD, gnorm, k,
    )
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)


def _stabPT_ssnewt(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  eps: ScalarType = -1e-4,
  tol: ScalarType = 1e-6,
  maxiter: int = 20,
  tol_ss: ScalarType = 1e-2,
  maxiter_ss: int = 10,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
) -> StabResult:
  """Newton's method for the stability test using a PT-based
  equation of state with preceding successive substitution iterations
  for initial guess improvement.

  Parameters
  ----------
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
      This `getPT_lnphii_Z` method must accept the pressure,
      temperature and composition of a phase and returns a tuple of
      logarithms of the fugacity coefficients of components and the
      phase compressibility factor.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      This `getPT_lnphii_Z_dnj` method must accept the pressure,
      temperature, phase composition and its mole number and returns a
      tuple of logarithms of the fugacity coefficients of components,
      the phase compressibility factor and the partial derivatives of
      logarithms of the fugacity coefficients of components with
      respect to component's mole numbers.

  eps: float
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: float
    Terminate the Newton's method successfully if the norm of the
    vector of equilibrium equations is less than `tol`. Default is
    `1e-6`.

  maxiter: int
    Maximum number of the Newton's method iterations. Default is `20`.

  tol_ss: float
    Switch to the Newton's method if the norm of the vector of
    equilibrium equations is less than `tol_ss`. Default is `1e-2`.

  maxiter_ss: int
    Maximum number of the successive substitution iterations.
    Default is `10`.

  linsolver: Callable[[ndarray, ndarray], ndarray]
    Function that accepts matrix A and vector b and finds vector x,
    which is the solution of the system of linear equations Ax = b.
    Default is `numpy.linalg.solve`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  properties are: `stab` a boolean flag indicating if a one-phase
  state is stable, `TPD` the tangent-plane distance at a local minima
  of the Gibbs energy function, `success` a boolean flag indicating
  if the calculation completed successfully.
  """
  logger.debug(
    'Stability Test (SS + Newton method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  hi = lnphiyi + np.log(yi)
  for j, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', j)
    k = 0
    kvik = kvi0.flatten()
    ni = kvik * yi
    lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
    gi = np.log(ni) + lnphixi - hi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration (SS) #%s:\n\tkvi = %s\n\tgnorm = %s', k, kvik, gnorm,
    )
    while (gnorm > tol_ss) & (k < maxiter_ss):
      k += 1
      kvik *= np.exp(-gi)
      ni = kvik * yi
      lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
      gi = np.log(ni) + lnphixi - hi
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration (SS) #%s:\n\tkvi = %s\n\tgnorm = %s', k, kvik, gnorm,
      )
    if np.isfinite(kvik).all():
      if (gnorm < tol):
        TPD = -np.log(ni.sum())
        if (TPD < eps):
          logger.info(
            'The system is unstable at P = %s Pa, T = %s K, yi = %s:\n\t'
            'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
            P, T, yi, kvik, TPD, gnorm, k,
          )
          n = ni.sum()
          xi = ni / n
          kvji = xi / yi, yi / xi
          return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                            TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                            lnphiyti=lnphixi, success=True)
      else:
        k = 0
        sqrtni = np.sqrt(ni)
        alphaik = 2. * sqrtni
        n = ni.sum()
        xi = ni / n
        lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
        gi = sqrtni * (np.log(ni) + lnphixi - hi)
        gnorm = np.linalg.norm(gi)
        logger.debug(
          'Iteration (Newton) #%s:\n\tkvi = %s\n\tgnorm = %s',
          k, ni/yi, gnorm,
        )
        while (gnorm > tol) & (k < maxiter):
          H = (np.diagflat(.5 * gi + 1.)
               + (sqrtni[:,None] * sqrtni) * dlnphixidnj)
          dalphai = linsolver(H, -gi)
          k += 1
          alphaik += dalphai
          sqrtni = alphaik * .5
          ni = sqrtni * sqrtni
          n = ni.sum()
          xi = ni / n
          lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
          gi = sqrtni * (np.log(ni) + lnphixi - hi)
          gnorm = np.linalg.norm(gi)
          logger.debug(
            'Iteration (Newton) #%s:\n\tkvi = %s\n\tgnorm = %s',
            k, ni/yi, gnorm,
          )
        if (k < maxiter) & (np.isfinite(alphaik).all()):
          TPD = -np.log(ni.sum())
          if (TPD < eps):
            logger.info(
              'The system is unstable at P = %s Pa, T = %s K, yi = %s:\n\t'
              'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
              P, T, yi, ni/yi, TPD, gnorm, k,
            )
            n = ni.sum()
            xi = ni / n
            kvji = xi / yi, yi / xi
            return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                              TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                              lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info(
      'The system is stable at P = %s Pa, T = %s K, yi = %s.\n\t'
      'The last kv-loop iteration results:\n\t'
      'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
      P, T, yi, ni/yi, TPD, gnorm, k,
    )
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)


def _stabPT_qnssnewt(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  eps: ScalarType = -1e-4,
  tol: ScalarType = 1e-6,
  maxiter: int = 20,
  tol_qnss: ScalarType = 1e-2,
  maxiter_qnss: int = 10,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
) -> StabResult:
  """Newton's method for the stability test using a PT-based
  equation of state with preceding quasi-newton successive substitution
  iterations for initial guess improvement.

  Before the Newton's method, the algorithm performs the Quasi-Newton
  Successive Substitution (QNSS) iterations to improve initial guesses
  of k-values. For the details of the QNSS-method see
  10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
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
      This `getPT_lnphii_Z` method must accept the pressure,
      temperature and composition of a phase and returns a tuple of
      logarithms of the fugacity coefficients of components and the
      phase compressibility factor.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      This `getPT_lnphii_Z_dnj` method must accept the pressure,
      temperature, phase composition and its mole number and returns a
      tuple of logarithms of the fugacity coefficients of components,
      the phase compressibility factor and the partial derivatives of
      logarithms of the fugacity coefficients of components with
      respect to component's mole numbers.

  eps: float
    System will be considered unstable when `TPD < eps`.
    Default is `-1e-4`.

  tol: float
    Terminate the Newton's method successfully if the norm of the
    vector of equilibrium equations is less than `tol`. Default is
    `1e-6`.

  maxiter: int
    Maximum number of the Newton's method iterations. Default is `20`.

  tol_qnss: float
    Switch to the Newton's method if the norm of the vector of
    equilibrium equations is less than `tol_qnss`. Default is `1e-2`.

  maxiter_qnss: int
    Maximum number of the quasi-newton successive substitution
    iterations. Default is `10`.

  linsolver: Callable[[ndarray, ndarray], ndarray]
    Function that accepts matrix A and vector b and finds vector x,
    which is the solution of the system of linear equations Ax = b.
    Default is `numpy.linalg.solve`.

  Returns
  -------
  Stability test results as an instance of `StabResult`. Important
  properties are: `stab` a boolean flag indicating if a one-phase
  state is stable, `TPD` the tangent-plane distance at a local minima
  of the Gibbs energy function, `success` a boolean flag indicating
  if the calculation completed successfully.
  """
  logger.debug(
    'Stability Test (QNSS + Newton)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  lnphiyi, Z = eos.getPT_lnphii_Z(P, T, yi)
  hi = lnphiyi + np.log(yi)
  for j, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', j)
    k = 0
    kvik = kvi0.flatten()
    ni = kvik * yi
    lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
    gi = np.log(ni) + lnphixi - hi
    gnorm = np.linalg.norm(gi)
    lmbd = 1.
    logger.debug(
      'Iteration (QNSS) #%s:\n\tkvi = %s\n\tgnorm = %s', k, kvik, gnorm,
    )
    while (gnorm > tol_qnss) & (k < maxiter_qnss):
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
      lnphixi, Zx = eos.getPT_lnphii_Z(P, T, ni/ni.sum())
      gi = np.log(ni) + lnphixi - hi
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration (QNSS) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tlmbd = %s',
        k, kvik, gnorm, lmbd,
      )
      if (gnorm < tol_qnss):
        break
      lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if np.isfinite(kvik).all():
      if (gnorm < tol):
        TPD = -np.log(ni.sum())
        if (TPD < eps):
          logger.info(
            'The system is unstable at P = %s Pa, T = %s K, yi = %s:\n\t'
            'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
            P, T, yi, kvik, TPD, gnorm, k,
          )
          n = ni.sum()
          xi = ni / n
          kvji = xi / yi, yi / xi
          return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                            TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                            lnphiyti=lnphixi, success=True)
      else:
        k = 0
        sqrtni = np.sqrt(ni)
        alphaik = 2. * sqrtni
        n = ni.sum()
        xi = ni / n
        lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
        gi = sqrtni * (np.log(ni) + lnphixi - hi)
        gnorm = np.linalg.norm(gi)
        logger.debug(
          'Iteration (Newton) #%s:\n\tkvi = %s\n\tgnorm = %s',
          k, ni/yi, gnorm,
        )
        while (gnorm > tol) & (k < maxiter):
          H = (np.diagflat(.5 * gi + 1.)
               + (sqrtni[:,None] * sqrtni) * dlnphixidnj)
          dalphai = linsolver(H, -gi)
          k += 1
          alphaik += dalphai
          sqrtni = alphaik * .5
          ni = sqrtni * sqrtni
          n = ni.sum()
          xi = ni / n
          lnphixi, Zx, dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, T, xi, n)
          gi = sqrtni * (np.log(ni) + lnphixi - hi)
          gnorm = np.linalg.norm(gi)
          logger.debug(
            'Iteration (Newton) #%s:\n\tkvi = %s\n\tgnorm = %s',
            k, ni/yi, gnorm,
          )
        if (k < maxiter) & (np.isfinite(alphaik).all()):
          TPD = -np.log(ni.sum())
          if (TPD < eps):
            logger.info(
              'The system is unstable at P = %s Pa, T = %s K, yi = %s:\n\t'
              'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
              P, T, yi, ni/yi, TPD, gnorm, k,
            )
            n = ni.sum()
            xi = ni / n
            kvji = xi / yi, yi / xi
            return StabResult(stable=False, yti=xi, kvji=kvji, gnorm=gnorm,
                              TPD=TPD, Z=Z, Zt=Zx, lnphiyi=lnphiyi,
                              lnphiyti=lnphixi, success=True)
  else:
    TPD = -np.log(ni.sum())
    logger.info(
      'The system is stable at P = %s Pa, T = %s K, yi = %s.\n\t'
      'The last kv-loop iteration results:\n\t'
      'kvi = %s\n\tTPD = %s\n\tgnorm = %s\n\tNiter = %s',
      P, T, yi, ni/yi, TPD, gnorm, k,
    )
    n = ni.sum()
    xi = ni / n
    kvji = xi / yi, yi / xi
    return StabResult(stable=True, yti=xi, kvji=kvji, gnorm=gnorm, TPD=TPD,
                      Z=Z, Zt=Zx, lnphiyi=lnphiyi, lnphiyti=lnphixi,
                      success=True)


