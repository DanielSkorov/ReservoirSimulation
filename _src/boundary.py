import logging

from functools import (
  partial,
)

import numpy as np

from utils import (
  mineig_rayquot,
)

from typing import Callable

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
  TensorType,
  EOSPTType,
  EOSVTType,
)

from stability import (
  StabResult,
  stabilityPT,
)


logger = logging.getLogger('bound')


def getVT_Tspinodal(
  V: ScalarType,
  yi: VectorType,
  eos: EOSVTType,
  T0: None | ScalarType = None,
  zeta0i : None | VectorType = None,
  multdT0: ScalarType = 1e-5,
  tol: ScalarType = 1e-5,
  maxiter: int = 25,
) -> tuple[ScalarType, VectorType]:
  """Calculates the spinodal temperature for a given volume
  and composition of a mixture.

  Parameters
  ----------
  V: float
    Volume of a mixture [m3].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  eos: EOSVTType
    An initialized instance of a VT-based equation of state.

  T0: None | float
    An initial guess of the spinodal temperature [K]. If it equals
    `None` a pseudocritical temperature will used.

  zeta0i: None | ndarray, shape (Nc,)
    An initial guess for the eigenvector corresponding to the lowest
    eigenvalue of the matrix of second partial derivatives of the
    Helmholtz energy function with respect to component mole
    numbers. If it equals `None` then `yi` will be used instead.

  multdT0: float
    A multiplier used to compute the temperature shift to estimate
    the partial derivative of the lowest eigenvalue with respect to
    temperature at the first iteration. Default is `1e-5`.

  tol: float
    Terminate successfully if the absolute value of the lowest
    eigenvalue is less then `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of iterations. Default is `25`.

  Returns
  -------
  A tuple of the spinodal temperature and the eigenvector.

  Raises
  ------
  The `ValueError` if the solution was not found.
  """
  logger.debug(
    'The spinodal temperature calculation procedure\n\tV = %s m3\n\tyi = %s',
    V, yi,
  )
  k = 0
  if T0 is None:
    Tk = 1.3 * yi.dot(eos.Tci)
  else:
    Tk = T0
  if zeta0i is None:
    zeta0i = yi
  Q = eos.getVT_lnfi_dnj(V, Tk, yi)[1]
  Nitrq, zetai, lmbdk = mineig_rayquot(Q, zeta0i)
  dT = multdT0 * Tk
  QdT = eos.getVT_lnfi_dnj(V, Tk + dT, yi)[1]
  lmbdkdT = np.linalg.eigvals(QdT).min()
  dlmbddT = (lmbdkdT - lmbdk) / dT
  dT = -lmbdk / dlmbddT
  logger.debug(
    'Iteration #%s:\n\tlmbd = %s\n\tzetai = %s\n\tT = %s\n\tdT = %s',
    k, lmbdk, zetai, Tk, dT,
  )
  k += 1
  while np.abs(lmbdk) > tol and k < maxiter:
    Tkp1 = Tk + dT
    Q = eos.getVT_lnfi_dnj(V, Tkp1, yi)[1]
    Nitrq, zetai, lmbdkp1 = mineig_rayquot(Q, zetai)
    dlmbddT = (lmbdkp1 - lmbdk) / dT
    dT = -lmbdkp1 / dlmbddT
    Tk = Tkp1
    lmbdk = lmbdkp1
    logger.debug(
      'Iteration #%s:\n\tlmbd = %s\n\tzetai = %s\n\tT = %s\n\tdT = %s',
      k, lmbdk, zetai, Tk, dT,
    )
    k += 1
  if np.abs(lmbdk) < tol:
    return Tk, zetai
  logger.warning(
    "The spinodal temperature was not found using %s:"
    "\n\tV = %s\n\tyi = %s\n\tT0 = %s"
    "\n\tzeta0i = %s\n\tmultdT0 = %s\n\ttol = %s\n\tmaxiter = %s",
    eos.name, V, yi, T0, zeta0i, multdT0, tol, maxiter,
  )
  raise ValueError(
    "The spinodal temperature solution procedure completed unsuccessfully.\n"
    "Try to increase the number of iterations or change the initial guess."
  )


def getVT_PcTc(
  yi: VectorType,
  eos: EOSVTType,
  v0: None | ScalarType = None,
  T0: None | ScalarType = None,
  kappa0: None | ScalarType = 3.5,
  multdV0: ScalarType = 1e-5,
  krange: tuple[ScalarType] = (1.1, 5.),
  kstep: ScalarType = .1,
  tol: ScalarType = 1e-5,
  maxiter: int = 25,
) -> tuple[ScalarType, ScalarType]:
  """Calculates the critical pressure and temperature of a mixture.

  Parameters
  ----------
  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  eos: EOSVTType
    An initialized instance of a VT-based equation of state.

  v0: float
    An initial guess of the critical molar volume of a mixture [m3/mol].
    Default is `None` which means that the gridding procedure or the
    `kappa0` will be used instead.

  kappa0: float | None
    An initial guess for the relation of the critical molar volume
    to the minimal possible volume provided by an equation of state.
    Default is `None` which means that the gridding procedure or the
    `v0` will be used instead.

  krange: tuple[float]
    A range of possible values of the relation of the molar volume
    to the minimal possible volume provided by an equation of state
    for the gridding procedure. The gridding procedure will be used
    if both initial guesses `v0` and `kappa0` are equal `None`.
    Default is `(1.1, 5.0)`.

  kstep: float
    A step which is used to perform the gridding procedure to find
    an initial guess of the relation of the molar volume to the minimal
    possible volume. Default is `0.1`.

   tol: float
    Terminate successfully if the absolute value of the primary
    variable change is less than `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of iterations. Default is `25`.

  Returns
  -------
  A tuple of the critical pressure and temperature.

  Raises
  ------
  The `ValueError` if the solution was not found.
  """
  logger.debug('The critical point calculation procedure\n\tyi = %s', yi)
  k = 0
  if T0 is None:
    T = 1.3 * yi.dot(eos.Tci)
  else:
    T = T0
  if v0 is None:
    if kappa0 is None:
      raise NotImplementedError(
        'The gridding procedure is not implemented yet.'
      )
    else:
      vmin = eos.getVT_vmin(T, yi)
      vk = kappa0 * vmin
  else:
    vk = v0
  T, zetaik = getVT_Tspinodal(vk, yi, eos, T, yi)
  vmin = eos.getVT_vmin(T, yi)
  kappak = vk / vmin
  Ck = (kappak - 1.)**2 * eos.getVT_d3F(vk, T, yi, zetaik)
  dv = multdV0 * vk
  vks = vk + dv
  kappaks = vks / vmin
  Cks = (kappaks - 1.)**2 * eos.getVT_d3F(vks, T, yi, zetaik)
  dCdkappa = (Cks - Ck) / (kappaks - kappak)
  dkappa = -Ck / dCdkappa
  logger.debug(
    'Iteration #%s:\n\tkappa = %s\n\tT = %s\n\tC = %s\n\tdkappa = %s',
    k, kappak, T, Ck, dkappa,
  )
  k += 1
  while np.abs(dkappa) > tol and k < maxiter:
    kappakp1 = kappak + dkappa
    vkp1 = kappakp1 * vmin
    if vkp1 < vmin:
      vkp1 = (vk + vmin) / 2.
      kappakp1 = vkp1 / vmin
      dkappa = kappakp1 - kappak
    T, zetaikp1 = getVT_Tspinodal(vkp1, yi, eos, T, zetaik)
    if zetaikp1[0] * zetaik[0] < 0.:
      zetaikp1 *= -1.
    Ckp1 = (kappakp1 - 1.)**2 * eos.getVT_d3F(vkp1, T, yi, zetaikp1)
    dCdkappa = (Ckp1 - Ck) / dkappa
    dkappa = -Ckp1 / dCdkappa
    vmin = eos.getVT_vmin(T, yi)
    vk = vkp1
    kappak = kappakp1
    Ck = Ckp1
    zetaik = zetaikp1
    logger.debug(
      'Iteration #%s:\n\tkappa = %s\n\tT = %s\n\tC = %s\n\tdkappa = %s',
      k, kappak, T, Ck, dkappa,
    )
    k += 1
  if np.abs(dkappa) < tol:
    Pc = eos.getVT_P(vk, T, yi)
    logger.info(
      'Critical point for yi = %s:\n\tPc = %s Pa\n\tTc = %s K'
      '\n\tdkappa = %s\n\tNiter = %s',
      yi, Pc, T, dkappa, k,
    )
    return Pc, T
  logger.warning(
    "The critical point was not found using %s:"
    "\n\tyi = %s\n\tv0 = %s m3/mol\n\tT0 = %s K\n\tkappa0 = %s\n\t"
    "multdV0 = %s\n\tkrange = %s\n\tkstep = %s\n\ttol = %s\n\tmaxiter = %s",
    eos.name, yi, v0, T0, kappa0, multdV0, krange, kstep, tol, maxiter,
  )
  raise ValueError(
    "The critical point solution procedure completed unsuccessfully.\n"
    "Try to increase the number of iterations or change the initial guess."
  )


class SatResult(dict):
  """Container for saturation point calculation outputs with
  pretty-printing.

  Attributes
  ----------
  P: float
    Saturation pressure [Pa].

  T: float
    Saturation temperature [K].

  yji: ndarray, shape (Np, Nc)
    Equilibrium composition of two phases for the saturation point.
    Two-dimensional array of real elements of size `(Np, Nc)`, where
    `Np` is the number of phases and `Nc` is the number of components.

  Zj: ndarray, shape (Np,)
    Compressibility factors of each phase. Array of real elements of
    size `(Np,)`, where `Np` is the number of phases.

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
      s = (f"Saturation pressure: {self.P} Pa\n"
           f"Saturation temperature: {self.T} K\n"
           f"Phase composition:\n{self.yji}\n"
           f"Phase compressibility factors:\n{self.Zj}\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


class PsatPT(object):
  """Saturation pressure calculation.

  Performs saturation pressure calculation using PT-based equations of
  state.

  Parameters
  ----------
  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P, T, yi, level) -> tuple[ndarray]`, where
      `P: float` is pressure [Pa], `T: float` is temperature [K],
      and `yi: ndarray`, shape `(Nc,)` is an array of components
      mole fractions, `Nc` is the number of components. This method
      is used to generate initial guesses of k-values.

    - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor.

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_lnphii_Z_dnj_dP(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  method: str
    Type of the solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Currently raises `NotImplementedError`),
    - `'qnss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval of the saturation pressure. This dictionary is used to
    specify arguments for the stability test procedure. Default is an
    empty dictionary.

  kwargs: dict
    Other arguments for a Psat-solver. It may contain such arguments
    as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others depending
    on the selected solver.

  Methods
  -------
  run(P, T, yi) -> SatResult
    This method performs the saturation pressure calculation for given
    the initial guess `P0: float` in [Pa], temperature `T: float` in
    [K] and composition `yi: ndarray` of `Nc`. This method returns
    saturation pressure calculation results as an instance of
    `SatResult`.

  search(P, T, yi) -> tuple[float, ndarray, float, float]
    This method performs the preliminary search to refine an initial
    guess of the saturation pressure and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the saturation pressure [Pa],
    - the initial guess for k-values as ndarray of shape `(Nc,)`,
    - the lower bound of the saturation pressure [Pa],
    - the upper bound of the saturation pressure [Pa].
  """
  def __init__(
    self,
    eos: EOSPTType,
    method: str = 'ss',
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.solver = partial(_PsatPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_PsatPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_PsatPT_newtA, eos=eos, **kwargs)
    elif method == 'newton-b':
      self.solver = partial(_PsatPT_newtB, eos=eos, **kwargs)
    elif method == 'newton-c':
      self.solver = partial(_PsatPT_newtC, eos=eos, **kwargs)
    elif method == 'ss-newton':
      raise NotImplementedError(
        'The SS-Newton method for the saturation pressure calculation is '
        'not implemented yet.'
      )
    elif method == 'qnss-newton':
      raise NotImplementedError(
        'The QNSS-Newton method for the saturation pressure calculation is '
        'not implemented yet.'
      )
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    upper: bool = True,
    step: ScalarType = 0.1,
    Pmin: ScalarType = 1.,
    Pmax: ScalarType = 1e8,
  ) -> SatResult:
    """Performs the saturation pressure calculation for known
    temperature and composition. To improve an initial guess, the
    preliminary search is performed.

    Parameters
    ----------
    P: float
      Initial guess of the saturation pressure [Pa].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondentherm serves as the dividing point between upper and
      lower phase boundaries. Default is `True`.

    step: float
      To specify the confidence interval for the saturation pressure
      calculation, the preliminary search is performed. This parameter
      regulates the step of this search in fraction units. For example,
      if it is necessary to find the upper bound of the confidence
      interval, then the next value of pressure will be calculated from
      the previous one using the formula: `Pnext = Pprev * (1. + step)`.
      Default is `0.1`.

    Pmin: float
      During the preliminary search, the pressure can not drop below
      the lower limit. Otherwise, the `ValueError` will be rised.
      Default is `1.` [Pa].

    Pmax: float
      During the preliminary search, the pressure can not exceed the
      upper limit. Otherwise, the `ValueError` will be rised.
      Default is `1e8` [Pa].

    Returns
    -------
    Saturation pressure calculation results as an instance of the
    `SatResult`. Important attributes are:

    - `P` the saturation pressure in [Pa],
    - `T` the saturation temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase,
    - `success` a boolean flag indicating if the calculation completed
      successfully.

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    P0, kvi0, Plow, Pupp = self.search(P, T, yi, upper, step, Pmin, Pmax)
    return self.solver(P0, T, yi, kvi0, Plow=Plow, Pupp=Pupp, upper=upper)

  def search(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    upper: bool = True,
    step: ScalarType = 0.1,
    Pmin: ScalarType = 1.,
    Pmax: ScalarType = 1e8,
  ) -> tuple[ScalarType, VectorType, ScalarType, ScalarType]:
    """Performs a preliminary search to refine the initial guess of
    the saturation pressure.

    Parameters
    ----------
    P: float
      The initial guess of the saturation pressure to be improved [Pa].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondentherm serves as the dividing point between upper and
      lower phase boundaries. Default is `True`.

    step: float
      To specify the confidence interval for the saturation pressure
      calculation, the preliminary search is performed. This parameter
      regulates the step of this search in fraction units. For example,
      if it is necessary to find the upper bound of the confidence
      interval, then the next value of pressure will be calculated from
      the previous one using the formula: `Pnext = Pprev * (1. + step)`.
      Default is `0.1`.

    Pmin: float
      During the preliminary search, the pressure can not drop below
      the lower limit. Otherwise, the `ValueError` will be rised.
      Default is `1.` [Pa].

    Pmax: float
      During the preliminary search, the pressure can not exceed the
      upper limit. Otherwise, the `ValueError` will be rised.
      Default is `1e8` [Pa].

    Returns
    -------
    A tuple of:

    - the improved initial guess for the saturation pressure [Pa],
    - the initial guess for k-values as ndarray of shape `(Nc,)`,
    - the lower bound of the saturation pressure [Pa],
    - the upper bound of the saturation pressure [Pa].

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    stab = self.stabsolver.run(P, T, yi)
    logger.debug(
      'For the initial guess P = %.1f Pa, the one-phase state is stable: %s',
      P, stab.stable,
    )
    if stab.stable and upper:
      logger.debug(
        'Finding the two-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Pupp = P
      c = 1. - step
      Plow = c * P
      stabmin = self.stabsolver.run(Plow, T, yi)
      logger.debug(
        'Plow = %.1f Pa, the one-phase state is stable: %s',
        Plow, stabmin.stable,
      )
      if stabmin.stable:
        Pupp = Plow
      while stabmin.stable and Plow > Pmin:
        Plow *= c
        stabmin = self.stabsolver.run(Plow, T, yi)
        logger.debug(
          'Plow = %.1f Pa, the one-phase state is stable: %s',
          Plow, stabmin.stable,
        )
        if stabmin.stable:
          Pupp = Plow
      if Plow < Pmin:
        raise ValueError(
          'The two-phase region was not identified. It could be because of\n'
          'its narrowness or absence. Try to change the initial guess for\n'
          'pressure or stability test parameters using the `stabkwargs`.\n'
          'It also might be helpful to reduce the value of the `step`.'
        )
      else:
        P = Plow
        stab = stabmin
    elif not stab.stable and upper:
      logger.debug(
        'Finding the one-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Plow = P
      c = 1. + step
      Pupp = c * P
      stabmax = self.stabsolver.run(Pupp, T, yi)
      logger.debug(
        'Pupp = %.1f Pa, the one-phase state is stable: %s',
        Pupp, stabmax.stable,
      )
      if not stabmax.stable:
        Plow = Pupp
      while not stabmax.stable and Pupp < Pmax:
        Pupp *= c
        stabmax = self.stabsolver.run(Pupp, T, yi)
        logger.debug(
          'Pupp = %.1f Pa, the one-phase state is stable: %s',
          Pupp, stabmax.stable,
        )
        if not stabmax.stable:
          Plow = Pupp
          stab = stabmax
      if Pupp > Pmax:
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for pressure and/or `Pmax` parameter.'
        )
      else:
        P = Plow
    elif stab.stable and not upper:
      logger.debug(
        'Finding the two-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Plow = P
      c = 1. + step
      Pupp = c * P
      stabmax = self.stabsolver.run(Pupp, T, yi)
      logger.debug(
        'Pupp = %.1f Pa, the one-phase state is stable: %s',
        Pupp, stabmax.stable,
      )
      while stabmax.stable and Pupp < Pmax:
        Pupp *= c
        stabmax = self.stabsolver.run(Pupp, T, yi)
        logger.debug(
          'Pupp = %.1f Pa, the one-phase state is stable: %s',
          Pupp, stabmax.stable,
        )
        if stabmax.stable:
          Plow = Pupp
      if Pupp > Pmax:
        raise ValueError(
          'The two-phase region was not identified. It could be because of\n'
          'its narrowness or absence. Try to change the initial guess for\n'
          'pressure or stability test parameters using the `stabkwargs`.\n'
          'It also might be helpful to reduce the value of the `step`.'
        )
      else:
        P = Pupp
        stab = stabmax
    else:
      logger.debug(
        'Finding the one-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Pupp = P
      c = 1. - step
      Plow = c * P
      stabmin = self.stabsolver.run(Plow, T, yi)
      logger.debug(
        'Plow = %.1f Pa, the one-phase state is stable: %s',
        Plow, stabmin.stable,
      )
      while not stabmin.stable and Plow > Pmin:
        Plow *= c
        stabmin = self.stabsolver.run(Plow, T, yi)
        logger.debug(
          'Plow = %.1f Pa, the one-phase state is stable: %s',
          Plow, stabmin.stable,
        )
        if not stabmin.stable:
          Pupp = Plow
          stab = stabmin
      if Plow < Pmin:
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for pressure and/or `Pmin` parameter.'
        )
      else:
        P = Pupp
    return P, stab.kvji[0], Plow, Pupp



def _PsatPT_solve_TPDeq_P(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  xi: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
  Plow0: ScalarType = 1.,
  Pupp0: ScalarType = 1e8,
  increasing: bool = True,
) -> tuple[ScalarType, VectorType, VectorType,
           ScalarType, ScalarType, ScalarType]:
  """Solves the TPD-equation using the PT-based equations of state for
  pressure at a constant temperature. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance, which determines
  the phase appearance or disappearance. A combination of the bisection
  method with Newton's method is used to solve the TPD-equation.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  Plow0: float
    The initial lower bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `1.0` [Pa].

  Pupp0: float
    The initial upper bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `1e8` [Pa].

  increasing: bool
    A flag that indicates if the TPD-equation vs pressure is the
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation pressure,
  - an array of the natural logarithms of fugacity coefficients of
    components in the trial phase,
  - the same for the initial composition of the mixture,
  - compressibility factors for both mixtures,
  - value of the tangent-plane distance.
  """
  k = 0
  Pk = P0
  Plow = Plow0
  Pupp = Pupp0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(f'Iter #{k}: Pk = {Pk / 1e6} MPa, {TPD = }')
  while k < maxiter and np.abs(TPD) > tol:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Plow = Pk
    else:
      Pupp = Pk
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    dP = - TPD / dTPDdP
    Pkp1 = Pk + dP
    if Pkp1 > Pupp or Pkp1 < Plow:
      Pkp1 = .5 * (Plow + Pupp)
      dP = Pkp1 - Pk
    if np.abs(dP) < 1e-8:
      break
    Pk = Pkp1
    k += 1
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(f'Iter #{k}: Pk = {Pk / 1e6} MPa, {TPD = }')
  return Pk, lnphixi, lnphiyi, Zx, Zy, TPD


def _PsatPT_ss(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Plow: ScalarType = 1.,
  Pupp: ScalarType = 1e8,
  upper: bool = True,
) -> SatResult:
  """Successive substitution (SS) method for the saturation pressure
  calculation using a PT-based equation of state.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Plow: float
    The lower bound for the TPD-equation solver. Default is `1.0` [Pa].

  Pupp: float
    The upper bound for the TPD-equation solver. Default is `1e8` [Pa].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info('Saturation pressure calculation using the SS-method.')
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Psat, Pa', 'gnorm', 'TPD',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %9.2e %10.2e'
  solverTPDeq = partial(_PsatPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Plow0=Plow, Pupp0=Pupp,
                        increasing=upper)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P0, T, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  TPD = xi.dot(gi - np.log(n))
  logger.debug(tmpl, k, *lnkik, Pk, gnorm, TPD)
  while (gnorm > tol or np.abs(TPD) > tol_tpd) and k < maxiter:
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, k, *lnkik, Pk, gnorm, TPD)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.isfinite(kik).all()
      and np.isfinite(Pk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation pressure for T = %.2f K is: %.1f Pa.', T, Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "The SS-method for saturation pressure calculation terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa",
      eos.name, P0, T, yi, Plow, Pupp,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _PsatPT_qnss(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Plow: ScalarType = 1.,
  Pupp: ScalarType = 1e8,
  upper: bool = True,
) -> SatResult:
  """QNSS-method for the saturation pressure calculation using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of nonlinear equations.
  For the details of the QNSS-method see: 10.1016/0378-3812(84)80013-8
  and 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Plow: float
    The lower bound for the TPD-equation solver. Default is `1.0` [Pa].

  Pupp: float
    The upper bound for the TPD-equation solver. Default is `1e8` [Pa].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info('Saturation pressure calculation using the QNSS-method.')
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Psat, Pa', 'gnorm', 'TPD',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %9.2e %10.2e'
  solverTPDeq = partial(_PsatPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Plow0=Plow, Pupp0=Pupp,
                        increasing=upper)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P0, T, yi, xi)
  gi = np.log(kik) + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  TPD = xi.dot(gi - np.log(n))
  lmbd = 1.
  logger.debug(tmpl, k, *lnkik, Pk, gnorm, TPD)
  while (gnorm > tol or np.abs(TPD) > tol_tpd) and k < maxiter:
    dlnki = -lmbd * gi
    max_dlnki = np.abs(dlnki).max()
    if max_dlnki > 6.:
      relax = 6. / max_dlnki
      lmbd *= relax
      dlnki *= relax
    k += 1
    tkm1 = dlnki.dot(gi)
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    logger.debug(tmpl, k, *lnkik, Pk, gnorm, TPD)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.isfinite(kik).all()
      and np.isfinite(Pk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation pressure for T = %.2f K is: %.1f Pa.', T, Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "The QNSS-method for saturation pressure calculation terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa",
      eos.name, P0, T, yi, Plow, Pupp,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _PsatPT_newt_improveP0(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  xi: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
  Plow0: ScalarType = 1.,
  Pupp0: ScalarType = 1e8,
  increasing: bool = True,
) -> tuple[ScalarType, VectorType, ScalarType, VectorType]:
  """Improves initial guess of the saturation pressure by solving
  the TPD-equation using the PT-based equations of state for pressure
  at a constant temperature. The TPD-equation is the equation of
  equality to zero of the tangent-plane distance, which determines
  the phase appearance or disappearance. A combination of the bisection
  method with Newton's method is used to solve the TPD-equation. The
  algorithm is the same as for the `_PsatPT_solve_TPDeq_P` but it
  differs in returned result.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  Plow0: float
    The initial lower bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `1.0` [Pa].

  Pupp0: float
    The initial upper bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `1e8` [Pa].

  increasing: bool
    A flag that indicates if the TPD-equation vs pressure is the
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation pressure,
  - an array of the natural logarithms of fugacity coefficients of
    components in the mixture,
  - compressibility factor of the mixture,
  - an array of partial derivatives of logarithms of the fugacity
    coefficients of components with respect to pressure.
  """
  k = 0
  Pk = P0
  Plow = Plow0
  Pupp = Pupp0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(f'Iter #{k}: Pk = {Pk / 1e6} MPa, {TPD = }')
  while k < maxiter and np.abs(TPD) > tol:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Plow = Pk
    else:
      Pupp = Pk
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    dP = - TPD / dTPDdP
    Pkp1 = Pk + dP
    if Pkp1 > Pupp or Pkp1 < Plow:
      Pkp1 = .5 * (Plow + Pupp)
      dP = Pkp1 - Pk
    if np.abs(dP) < 1e-8:
      break
    Pk = Pkp1
    k += 1
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(f'Iter #{k}: Pk = {Pk / 1e6} MPa, {TPD = }')
  return Pk, lnphiyi, Zy, dlnphiyidP


def _PsatPT_newtA(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  Plow: ScalarType = 1.,
  Pupp: ScalarType = 1e8,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation pressure by solving a system of
  nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the sum of mole numbers of components in
  the trial phase equals unity.

  The formulation of the system of nonlinear equations is based on the
  paper of M.L. Michelsen: 10.1016/0378-3812(80)80001-X.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dnj_dP(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Plow: float
    The pressure lower bound. Default is `1.0` [Pa].

  Pupp: float
    The pressure upper bound. Default is `1e8` [Pa].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm of the saturation pressure
    initial guess improvement. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation pressure initial guess
    improvement. Default is `8`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. This parameter is used by the algorithm of
    the saturation pressure initial guess improvement.
    Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info(
    "Saturation pressure calculation using Newton's method (A-form)."
  )
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s', 'Nit',
    *map(lambda s: 'lnkv' + s, map(str, range(Nc))), 'Psat, Pa', 'gnorm',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %9.2e'
  J = np.zeros(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphiyi, Zy, dlnphiyidP = _PsatPT_newt_improveP0(P0, T, yi, xi, eos,
                                                       tol_tpd, maxiter_tpd,
                                                       Plow, Pupp, upper)
  lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(Pk, T, xi,
                                                                   n)
  gi[:Nc] = lnkik + lnphixi - lnphiyi
  gi[-1] = n - 1.
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Pk, gnorm)
  while gnorm > tol and k < maxiter:
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = ni
    J[:Nc,-1] = Pk * (dlnphixidP - dlnphiyidP)
    try:
      dlnkilnP = linsolver(J, -gi)
    except:
      dlnkilnP = -gi
    k += 1
    lnkik += dlnkilnP[:-1]
    Pkp1 = Pk * np.exp(dlnkilnP[-1])
    if Pkp1 > Pupp:
      Pk = .5 * (Pk + Pupp)
    elif Pkp1 < Plow:
      Pk = .5 * (Plow + Pk)
    else:
      Pk = Pkp1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(Pk, T,
                                                                     xi, n)
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    gi[:Nc] = lnkik + lnphixi - lnphiyi
    gi[-1] = n - 1.
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, k, *lnkik, Pk, gnorm)
  if gnorm < tol and np.isfinite(kik).all() and np.isfinite(Pk):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation pressure for T = %.2f K is: %.1f Pa.', T, Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Newton's method (A-form) for saturation pressure calculation "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa",
      eos.name, P0, T, yi, Plow, Pupp,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _PsatPT_newtB(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  Plow: ScalarType = 1.,
  Pupp: ScalarType = 1e8,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation pressure by solving a system of
  nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the tangent-plane distance is zero at
  the saturation curve.

  The formulation of the system of nonlinear equations is based on the
  paper of L.X. Nghiem and Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dnj_dP(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Plow: float
    The pressure lower bound. Default is `1.0` [Pa].

  Pupp: float
    The pressure upper bound. Default is `1e8` [Pa].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm of the saturation pressure
    initial guess improvement. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation pressure initial guess
    improvement. Default is `8`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. This parameter is used by the algorithm of
    the saturation pressure initial guess improvement.
    Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info(
    "Saturation pressure calculation using Newton's method (B-form)."
  )
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s', 'Nit',
    *map(lambda s: 'lnkv' + s, map(str, range(Nc))), 'Psat, Pa', 'gnorm',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %9.2e'
  J = np.empty(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphiyi, Zy, dlnphiyidP = _PsatPT_newt_improveP0(P0, T, yi, xi, eos,
                                                       tol_tpd, maxiter_tpd,
                                                       Plow, Pupp, upper)
  lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(Pk, T, xi,
                                                                   n)
  gi[:Nc] = lnkik + lnphixi - lnphiyi
  hi = gi[:Nc] - np.log(n)
  gi[-1] = xi.dot(hi)
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Pk, gnorm)
  while gnorm > tol and k < maxiter:
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = xi * (hi - gi[-1])
    J[:Nc,-1] = Pk * (dlnphixidP - dlnphiyidP)
    J[-1,-1] = xi.dot(J[:Nc,-1])
    try:
      dlnkilnP = linsolver(J, -gi)
    except:
      dlnkilnP = -gi
    k += 1
    lnkik += dlnkilnP[:-1]
    Pkp1 = Pk * np.exp(dlnkilnP[-1])
    if Pkp1 > Pupp:
      Pk = .5 * (Pk + Pupp)
    elif Pkp1 < Plow:
      Pk = .5 * (Plow + Pk)
    else:
      Pk = Pkp1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(Pk, T,
                                                                     xi, n)
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    gi[:Nc] = lnkik + lnphixi - lnphiyi
    hi = gi[:Nc] - np.log(n)
    gi[-1] = xi.dot(hi)
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, k, *lnkik, Pk, gnorm)
  if gnorm < tol and np.isfinite(kik).all() and np.isfinite(Pk):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation pressure for T = %.2f K is: %.1f Pa.', T, Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Newton's method (B-form) for saturation pressure calculation "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa",
      eos.name, P0, T, yi, Plow, Pupp,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _PsatPT_newtC(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Plow: ScalarType = 1.,
  Pupp: ScalarType = 1e8,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation pressure by solving a system of
  nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the tangent-plane distance is zero at
  the saturation curve. The TPD-equation is solved in the inner
  while-loop.

  The formulation of the system of nonlinear equations is based on the
  paper of L.X. Nghiem and Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Plow: float
    The pressure lower bound. Default is `1.0` [Pa].

  Pupp: float
    The pressure upper bound. Default is `1e8` [Pa].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info(
    "Saturation pressure calculation using Newton's method (C-form)."
  )
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Psat, Pa', 'gnorm', 'TPD',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %9.2e %10.2e'
  solverTPDeq = partial(_PsatPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Plow0=Plow, Pupp0=Pupp,
                        increasing=upper)
  I = np.eye(eos.Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P0, T, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  TPD = ni.dot(gi)
  logger.debug(tmpl, k, *lnkik, Pk, gnorm, TPD)
  while (gnorm > tol or np.abs(TPD) > tol_tpd) and k < maxiter:
    dlnphixidnj = eos.getPT_lnphii_Z_dnj(Pk, T, xi, n)[2]
    J = I + ni * dlnphixidnj
    try:
      dlnki = linsolver(J, -gi)
    except:
      dlnki = -gi
    k += 1
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, k, *lnkik, Pk, gnorm, TPD)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.isfinite(kik).all()
      and np.isfinite(Pk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation pressure for T = %.2f K is: %.1f Pa.', T, Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Newton's method (C-form) for saturation pressure calculation "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp= %s Pa",
      eos.name, P0, T, yi, Plow, Pupp,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


class TsatPT(object):
  """Saturation temperature calculation.

  Performs saturation temperature calculation using PT-based equations
  of state.

  Parameters
  ----------
  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P, T, yi, level) -> tuple[ndarray]`, where
      `P: float` is pressure [Pa], `T: float` is temperature [K],
      and `yi: ndarray`, shape `(Nc,)` is an array of components
      mole fractions, `Nc` is the number of components. This method
      is used to generate initial guesses of k-values.

    - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor.

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_lnphii_Z_dnj_dT(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  method: str
    Type of the solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Currently raises `NotImplementedError`),
    - `'newton'` (Currently raises `NotImplementedError`),
    - `'ss-newton'` (Currently raises `NotImplementedError`),
    - `'qnss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval of the saturation temperature. This dictionary is used to
    specify arguments for the stability test procedure. Default is an
    empty dictionary.

  kwargs: dict
    Other arguments for a Tsat-solver. It may contain such arguments
    as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others depending
    on the selected solver.

  Methods
  -------
  run(P, T, yi) -> SatResult
    This method performs the saturation temperature calculation for
    given pressure `P: float` in [Pa], the initial guess `T0: float` in
    [K] and composition `yi: ndarray` of `Nc`. This method returns
    saturation temperature calculation results as an instance of
    `SatResult`.

  search(P, T, yi) -> tuple[float, ndarray, float, float]
    This method performs the preliminary search to refine an initial
    guess of the saturation temperature and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the saturation temperature [K],
    - the initial guess for k-values as ndarray of shape `(Nc,)`,
    - the lower bound of the saturation temperature [K],
    - the upper bound of the saturation temperature [K].
  """
  def __init__(
    self,
    eos: EOSPTType,
    method: str = 'ss',
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.solver = partial(_TsatPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_TsatPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_TsatPT_newtA, eos=eos, **kwargs)
    elif method == 'newton-b':
      self.solver = partial(_TsatPT_newtB, eos=eos, **kwargs)
    elif method == 'newton-c':
      self.solver = partial(_TsatPT_newtC, eos=eos, **kwargs)
    elif method == 'ss-newton':
      raise NotImplementedError(
        'The SS-Newton method for the saturation temperature calculation is '
        'not implemented yet.'
      )
    elif method == 'qnss-newton':
      raise NotImplementedError(
        'The QNSS-Newton method for the saturation temperature calculation '
        'is not implemented yet.'
      )
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    upper: bool = True,
    step: ScalarType = 0.1,
    Tmin: ScalarType = 173.15,
    Tmax: ScalarType = 973.15,
  ) -> SatResult:
    """Performs the saturation temperature calculation for known
    pressure and composition. To improve an initial guess, the
    preliminary search is performed.

    Parameters
    ----------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Initial guess of the saturation temperature [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondenbar serves as the dividing point between upper and
      lower phase boundaries. Default is `True`.

    step: float
      To specify the confidence interval for the saturation temperature
      calculation, the preliminary search is performed. This parameter
      regulates the step of this search in fraction units. For example,
      if it is necessary to find the upper bound of the confidence
      interval, then the next value of temperature will be calculated
      from the previous one using the formula:
      `Tnext = Tprev * (1. + step)`. Default is `0.1`.

    Tmin: float
      During the preliminary search, the temperature can not drop below
      the lower limit. Otherwise, the `ValueError` will be rised.
      Default is `173.15` [K].

    Tmax: float
      During the preliminary search, the temperature can not exceed the
      upper limit. Otherwise, the `ValueError` will be rised.
      Default is `973.15` [K].

    Returns
    -------
    Saturation temperature calculation results as an instance of the
    `SatResult`. Important attributes are:

    - `P` the saturation pressure in [Pa],
    - `T` the saturation temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase,
    - `success` a boolean flag indicating if the calculation completed
      successfully.

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    T0, kvi0, Tlow, Tupp = self.search(P, T, yi, upper, step, Tmin, Tmax)
    return self.solver(P, T0, yi, kvi0, Tlow=Tlow, Tupp=Tupp, upper=upper)

  def search(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    upper: bool = True,
    step: ScalarType = 0.1,
    Tmin: ScalarType = 173.15,
    Tmax: ScalarType = 973.15,
  ) -> tuple[ScalarType, VectorType, ScalarType, ScalarType]:
    """Performs a preliminary search to refine the initial guess of
    the saturation temperature.

    Parameters
    ----------
    P: float
      Pressure of a mixture [Pa].

    T: float
      The initial guess of the saturation temperature to be improved
      [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondenbar serves as the dividing point between upper and
      lower phase boundaries. Default is `True`.

    step: float
      To specify the confidence interval for the saturation temperature
      calculation, the preliminary search is performed. This parameter
      regulates the step of this search in fraction units. For example,
      if it is necessary to find the upper bound of the confidence
      interval, then the next value of temperature will be calculated
      from the previous one using the formula:
      `Tnext = Tprev * (1. + step)`. Default is `0.1`.

    Tmin: float
      During the preliminary search, the temperature can not drop below
      the lower limit. Otherwise, the `ValueError` will be rised.
      Default is `173.15` [K].

    Tmax: float
      During the preliminary search, the temperature can not exceed the
      upper limit. Otherwise, the `ValueError` will be rised.
      Default is `973.15` [K].

    Returns
    -------
    A tuple of:

    - the improved initial guess for the saturation temperature [K],
    - the initial guess for k-values as ndarray of shape `(Nc,)`,
    - the lower bound of the saturation temperature [K],
    - the upper bound of the saturation temperature [K].

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    stab = self.stabsolver.run(P, T, yi)
    logger.debug(
      'For the initial guess T = %.2f K, the one-phase state is stable: %s',
      T, stab.stable,
    )
    if stab.stable and upper:
      logger.debug(
        'Finding the two-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Tupp = T
      c = 1. - step
      Tlow = c * T
      stabmin = self.stabsolver.run(P, Tlow, yi)
      logger.debug(
        'Tlow = %.2f K, the one-phase state is stable: %s',
        Tlow, stabmin.stable,
      )
      if stabmin.stable:
        Tupp = Tlow
      while stabmin.stable and Tlow > Tmin:
        Tlow *= c
        stabmin = self.stabsolver.run(P, Tlow, yi)
        logger.debug(
          'Tlow = %.2f K, the one-phase state is stable: %s',
          Tlow, stabmin.stable,
        )
        if stabmin.stable:
          Tupp = Tlow
      if Tlow < Tmin:
        raise ValueError(
          'The two-phase region was not identified. It could be because of\n'
          'its narrowness or absence. Try to change the initial guess for\n'
          'temperature or stability test parameters using the `stabkwargs`.\n'
          'It also might be helpful to reduce the value of the `step`.'
        )
      else:
        T = Tlow
        stab = stabmin
    elif not stab.stable and upper:
      logger.debug(
        'Finding the one-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Tlow = T
      c = 1. + step
      Tupp = c * T
      stabmax = self.stabsolver.run(P, Tupp, yi)
      logger.debug(
        'Tupp = %.2f K, the one-phase state is stable: %s',
        Tupp, stabmax.stable,
      )
      if not stabmax.stable:
        Tlow = Tupp
      while not stabmax.stable and Tupp < Tmax:
        Tupp *= c
        stabmax = self.stabsolver.run(P, Tupp, yi)
        logger.debug(
          'Tupp = %.2f K, the one-phase state is stable: %s',
          Tupp, stabmax.stable,
        )
        if not stabmax.stable:
          Tlow = Tupp
          stab = stabmax
      if Tupp > Tmax:
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for temperature and/or `Tmax` parameter.'
        )
      else:
        T = Tlow
    elif stab.stable and not upper:
      logger.debug(
        'Finding the two-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Tlow = T
      c = 1. + step
      Tupp = c * T
      stabmax = self.stabsolver.run(P, Tupp, yi)
      logger.debug(
        'Tupp = %.2f K, the one-phase state is stable: %s',
        Tupp, stabmax.stable,
      )
      while stabmax.stable and Tupp < Tmax:
        Tupp *= c
        stabmax = self.stabsolver.run(P, Tupp, yi)
        logger.debug(
          'Tupp = %.2f K, the one-phase state is stable: %s',
          Tupp, stabmax.stable,
        )
        if stabmax.stable:
          Tlow = Tupp
      if Tupp > Tmax:
        raise ValueError(
          'The two-phase region was not identified. It could be because of\n'
          'its narrowness or absence. Try to change the initial guess for\n'
          'temperature or stability test parameters using the `stabkwargs`\n'
          'It also might be helpful to reduce the value of the `step`.'
        )
      else:
        T = Tupp
        stab = stabmax
    else:
      logger.debug(
        'Finding the one-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Tupp = T
      c = 1. - step
      Tlow = c * T
      stabmin = self.stabsolver.run(P, Tlow, yi)
      logger.debug(
        'Tlow = %.2f K, the one-phase state is stable: %s',
        Tlow, stabmin.stable,
      )
      while not stabmin.stable and Tlow > Tmin:
        Tlow *= c
        stabmin = self.stabsolver.run(P, Tlow, yi)
        logger.debug(
          'Tlow = %.2f K, the one-phase state is stable: %s',
          Tlow, stabmin.stable,
        )
        if not stabmin.stable:
          Tupp = Tlow
          stab = stabmin
      if Tlow < Tmin:
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for temperature and/or `Tmin` parameter.'
        )
      else:
        T = Tupp
    return T, stab.kvji[0], Tlow, Tupp


def _TsatPT_solve_TPDeq_T(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  xi: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
  Tlow0: ScalarType = 173.15,
  Tupp0: ScalarType = 973.15,
  increasing: bool = True,
) -> tuple[ScalarType, VectorType, VectorType,
           ScalarType, ScalarType, ScalarType]:
  """Solves the TPD-equation using the PT-based equations of state for
  temperature at a constant pressure. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance, which determines
  the phase appearance or disappearance. A combination of the bisection
  method with Newton's method is used to solve the TPD-equation.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  Tlow0: float
    The initial lower bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `173.15` [K].

  Tupp0: float
    The initial upper bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `973.15` [K].

  increasing: bool
    A flag that indicates if the TPD-equation vs temperature is the
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation temperature,
  - an array of the natural logarithms of fugacity coefficients of
    components in the trial phase,
  - the same for the initial composition of the mixture,
  - compressibility factors for both mixtures.
  - value of the tangent-plane distance.
  """
  k = 0
  Tlow = Tlow0
  Tupp = Tupp0
  Tk = T0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
  lnphixi, Zx, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(f'Iter #{k}: Tk = {Tk-273.15} C, {TPD = }')
  while k < maxiter and np.abs(TPD) > tol:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Tlow = Tk
    else:
      Tupp = Tk
    dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
    dT = - TPD / dTPDdT
    Tkp1 = Tk + dT
    if Tkp1 > Tupp or Tkp1 < Tlow:
      Tkp1 = .5 * (Tlow + Tupp)
      dT = Tkp1 - Tk
    if np.abs(dT) < 1e-8:
      break
    Tk = Tkp1
    k += 1
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    lnphixi, Zx, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(f'Iter #{k}: Tk = {Tk-273.15} C, {TPD = }')
  return Tk, lnphixi, lnphiyi, Zx, Zy, TPD


def _TsatPT_ss(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Tlow: ScalarType = 173.15,
  Tupp: ScalarType = 973.15,
  upper: bool = True,
) -> SatResult:
  """Successive substitution (SS) method for the saturation temperature
  calculation using a PT-based equation of state.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Tlow: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info(
    "Saturation temperature calculation using the SS-method."
  )
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Tsat, K', 'gnorm', 'TPD',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %8.2f %9.2e %10.2e'
  solverTPDeq = partial(_TsatPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tlow0=Tlow, Tupp0=Tupp,
                        increasing=upper)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Tk, gnorm, TPD)
  while (gnorm > tol or np.abs(TPD) > tol_tpd) and k < maxiter:
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, k, *lnkik, Tk, gnorm, TPD)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.isfinite(kik).all()
      and np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "The SS-method for saturation pressure calculation terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K",
      eos.name, P, T0, yi, Tlow, Tupp,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _TsatPT_qnss(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Tlow: ScalarType = 173.15,
  Tupp: ScalarType = 973.15,
  upper: bool = True,
) -> SatResult:
  """Quasi-Newton Successive Substitution (QNSS) method for the
  saturation temperature calculation using a PT-based equation
  of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of nonlinear equations.
  For the details of the QNSS-method see: 10.1016/0378-3812(84)80013-8
  and 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Tlow: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info(
    "Saturation temperature calculation using the QNSS-method."
  )
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Tsat, K', 'gnorm', 'TPD',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %8.2f %9.2e %10.2e'
  solverTPDeq = partial(_TsatPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tlow0=Tlow, Tupp0=Tupp,
                        increasing=upper)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  TPD = ni.dot(gi)
  lmbd = 1.
  logger.debug(tmpl, k, *lnkik, Tk, gnorm, TPD)
  while (gnorm > tol or np.abs(TPD) > tol_tpd) and k < maxiter:
    dlnki = -lmbd * gi
    max_dlnki = np.abs(dlnki).max()
    if max_dlnki > 6.:
      relax = 6. / max_dlnki
      lmbd *= relax
      dlnki *= relax
    k += 1
    tkm1 = dlnki.dot(gi)
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    logger.debug(tmpl, k, *lnkik, Tk, gnorm, TPD)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.isfinite(kik).all()
      and np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "The QNSS-method saturation pressure calculation terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K",
      eos.name, P, T0, yi, Tlow, Tupp,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _TsatPT_newt_improveT0(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  xi: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
  Tlow0: ScalarType = 173.15,
  Tupp0: ScalarType = 973.15,
  increasing: bool = True,
) -> tuple[ScalarType, VectorType, ScalarType, VectorType]:
  """Improves initial guess of the saturation temperature by solving
  the TPD-equation using the PT-based equations of state for temperature
  at a constant pressure. The TPD-equation is the equation of equality
  to zero of the tangent-plane distance, which determines the phase
  appearance or disappearance. A combination of the bisection method
  with Newton's method is used to solve the TPD-equation. The algorithm
  is the same as for the `_TsatPT_solve_TPDeq_T` but it differs in
  returned result.

  Parameters
  ----------
  T0: float
    Initial guess of the saturation temperature [K].

  P: float
    Pressure of a mixture [Pa].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  Tlow0: float
    The initial lower bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `173.15` [K].

  Tupp0: float
    The initial upper bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `973.15` [K].

  increasing: bool
    A flag that indicates if the TPD-equation vs temperature is the
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation temperature,
  - an array of the natural logarithms of fugacity coefficients of
    components in the mixture,
  - compressibility factors of the mixture,
  - an array of partial derivatives of logarithms of the fugacity
    coefficients of components with respect to temperature.
  """
  k = 0
  Tlow = Tlow0
  Tupp = Tupp0
  Tk = T0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
  lnphixi, Zt, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(f'Iter #{k}: Tk = {Tk-273.15} C, {TPD = }')
  while k < maxiter and np.abs(TPD) > tol:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Tlow = Tk
    else:
      Tupp = Tk
    dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
    dT = - TPD / dTPDdT
    Tkp1 = Tk + dT
    if Tkp1 > Tupp or Tkp1 < Tlow:
      Tkp1 = .5 * (Tlow + Tupp)
      dT = Tkp1 - Tk
    if np.abs(dT) < 1e-8:
      break
    Tk = Tkp1
    k += 1
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    lnphixi, Zt, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(f'Iter #{k}: Tk = {Tk-273.15} C, {TPD = }')
  return Tk, lnphiyi, Zy, dlnphiyidT


def _TsatPT_newtA(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  Tlow: ScalarType = 1.,
  Tupp: ScalarType = 1e8,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation temperature by solving a system
  of nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the sum of mole numbers of components in
  the trial phase equals unity.

  The formulation of the system of nonlinear equations is based on the
  paper of M.L. Michelsen: 10.1016/0378-3812(80)80001-X.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dnj_dT(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Tlow: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm of the saturation
    temperature initial guess improvement. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation temperature initial guess
    improvement. Default is `8`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. This parameter is used by the algorithm of
    the saturation temperature initial guess improvement.
    Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info(
    "Saturation temperature calculation using Newton's method (A-form)."
  )
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s', 'Nit',
    *map(lambda s: 'lnkv' + s, map(str, range(Nc))), 'Tsat, K', 'gnorm',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %8.2f %9.2e'
  J = np.zeros(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphiyi, Zy, dlnphiyidT = _TsatPT_newt_improveT0(P, T0, yi, xi, eos,
                                                       tol_tpd, maxiter_tpd,
                                                       Tlow, Tupp, upper)
  lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk, xi,
                                                                   n)
  gi[:Nc] = lnkik + lnphixi - lnphiyi
  gi[-1] = n - 1.
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Tk, gnorm)
  while gnorm > tol and k < maxiter:
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = ni
    J[:Nc,-1] = Tk * (dlnphixidT - dlnphiyidT)
    try:
      dlnkilnT = linsolver(J, -gi)
    except:
      dlnkilnT = -gi
    k += 1
    lnkik += dlnkilnT[:-1]
    Tkp1 = Tk * np.exp(dlnkilnT[-1])
    if Tkp1 > Tupp:
      Tk = .5 * (Tk + Tupp)
    elif Tkp1 < Tlow:
      Tk = .5 * (Tlow + Tk)
    else:
      Tk = Tkp1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk,
                                                                     xi, n)
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    gi[:Nc] = lnkik + lnphixi - lnphiyi
    gi[-1] = n - 1.
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, k, *lnkik, Tk, gnorm)
  if gnorm < tol and np.isfinite(kik).all() and np.isfinite(Tk):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Newton's method (A-form) for saturation temperature calculation "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K",
      eos.name, P, T0, yi, Tlow, Tupp,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _TsatPT_newtB(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  Tlow: ScalarType = 1.,
  Tupp: ScalarType = 1e8,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation temperature by solving a system
  of nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the tangent-plane distance is zero at
  the saturation curve.

  The formulation of the system of nonlinear equations is based on the
  paper of L.X. Nghiem and Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dnj_dT(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Tlow: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm of the saturation
    temperature initial guess improvement. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation temperature initial guess
    improvement. Default is `8`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. This parameter is used by the algorithm of
    the saturation temperature initial guess improvement.
    Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info(
    "Saturation temperature calculation using Newton's method (A-form)."
  )
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s', 'Nit',
    *map(lambda s: 'lnkv' + s, map(str, range(Nc))), 'Tsat, K', 'gnorm',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %8.2f %9.2e'
  J = np.empty(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphiyi, Zy, dlnphiyidT = _TsatPT_newt_improveT0(P, T0, yi, xi, eos,
                                                       tol_tpd, maxiter_tpd,
                                                       Tlow, Tupp, upper)
  lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk, xi,
                                                                   n)
  gi[:Nc] = lnkik + lnphixi - lnphiyi
  hi = gi[:Nc] - np.log(n)
  gi[-1] = xi.dot(hi)
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Tk, gnorm)
  while gnorm > tol and k < maxiter:
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = xi * (hi - gi[-1])
    J[:Nc,-1] = Tk * (dlnphixidT - dlnphiyidT)
    J[-1,-1] = xi.dot(J[:Nc,-1])
    try:
      dlnkilnT = linsolver(J, -gi)
    except:
      dlnkilnT = -gi
    k += 1
    lnkik += dlnkilnT[:-1]
    Tkp1 = Tk * np.exp(dlnkilnT[-1])
    if Tkp1 > Tupp:
      Tk = .5 * (Tk + Tupp)
    elif Tkp1 < Tlow:
      Tk = .5 * (Tlow + Tk)
    else:
      Tk = Tkp1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk,
                                                                     xi, n)
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    gi[:Nc] = lnkik + lnphixi - lnphiyi
    hi = gi[:Nc] - np.log(n)
    gi[-1] = xi.dot(hi)
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, k, *lnkik, Tk, gnorm)
  if gnorm < tol and np.isfinite(kik).all() and np.isfinite(Tk):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Newton's method (B-form) for saturation temperature calculation "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K",
      eos.name, P, T0, yi, Tlow, Tupp,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _TsatPT_newtC(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Tlow: ScalarType = 1.,
  Tupp: ScalarType = 1e8,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation temperature by solving a system
  of nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the tangent-plane distance is zero at
  the saturation curve. The TPD-equation is solved in the inner
  while-loop.

  The formulation of the system of nonlinear equations is based on the
  paper of L.X. Nghiem and Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Tlow: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info(
    "Saturation temperature calculation using Newton's method (C-form)."
  )
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Tsat, K', 'gnorm', 'TPD',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %8.2f %9.2e %10.2e'
  solverTPDeq = partial(_TsatPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tlow0=Tlow, Tupp0=Tupp,
                        increasing=upper)
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Tk, gnorm, TPD)
  while (gnorm > tol or np.abs(TPD) > tol_tpd) and k < maxiter:
    dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, Tk, xi, n)[2]
    J = I + ni * dlnphixidnj
    try:
      dlnki = linsolver(J, -gi)
    except:
      dlnki = -gi
    k += 1
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, k, *lnkik, Tk, gnorm, TPD)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.isfinite(kik).all()
      and np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Newton's method (C-form) for saturation temperature calculation "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K",
      eos.name, P, T0, yi, Tlow, Tupp,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


class PmaxPT(PsatPT):
  """Cricondenbar calculation.

  Performs the cricondenbar point calculation using PT-based equations
  of state.

  Parameters
  ----------
  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P, T, yi, level) -> tuple[ndarray]`, where
      `P: float` is pressure [Pa], `T: float` is temperature [K],
      and `yi: ndarray`, shape `(Nc,)` is an array of components
      mole fractions, `Nc` is the number of components. This method
      is used to generate initial guesses of k-values.

    - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor.

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dT_d2T(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature.

    If the solution method would be `'newton'` then it also must have:

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  method: str
    Type of the solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'newton'` (Newton's method).

    Default is `'ss'`.

  step: float
    To specify the confidence interval for pressure of the cricondenbar
    point calculation, the preliminary search is performed. This
    parameter regulates the step of this search in fraction units.
    During the preliminary search, the next value of pressure will be
    calculated from the previous one using the formula:
    `Pnext = Pprev * (1. + step)`. Default is `0.1`.

  lowerlimit: float
    During the preliminary search, the pressure can not drop below
    the lower limit. Otherwise, the `ValueError` will be rised.
    Default is `1.` [Pa].

  upperlimit: float
    During the preliminary search, the pressure can not exceed the
    upper limit. Otherwise, the `ValueError` will be rised.
    Default is `1e8` [Pa].

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval for pressure of the cricondenbar point. This dictionary is
    used to specify arguments for the stability test procedure. Default
    is an empty dictionary.

  kwargs: dict
    Other arguments for a cricondebar-solver. It may contain such
    arguments as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others
    depending on the selected solver.

  Methods
  -------
  run(P, T, yi) -> SatResult
    This method performs the cricondenbar point calculation of the
    composition `yi: ndarray` with shape `(Nc,)` for given initial
    guesses of pressure `P: float` in [Pa] and temperature `T: float`
    in [K]. This method returns cricondenbar point calculation results
    as an instance of `SatResult`.

  search(P, T, yi) -> tuple[float, ndarray, float, float]
    This method performs the preliminary search to refine an initial
    guess of the cricondenbar pressure and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the cricondenbar pressure [Pa],
    - the initial guess for k-values as ndarray of shape `(Nc,)`,
    - the lower bound of the cricondenbar pressure [Pa],
    - the upper bound of the cricondenbar pressure [Pa].
  """
  def __init__(
    self,
    eos: EOSPTType,
    method: str = 'ss',
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.solver = partial(_PmaxPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_PmaxPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_PmaxPT_newtC, eos=eos, **kwargs)
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    step: ScalarType = 0.1,
    Pmin: ScalarType = 1.,
    Pmax: ScalarType = 1e8,
  ) -> SatResult:
    """Performs the cricondenbar point calculation for a mixture. To
    improve the initial guess of pressure, the preliminary search is
    performed.

    Parameters
    ----------
    P: float
      Initial guess of the cricondenbar pressure [Pa].

    T: float
      Initial guess of the cricondenbar temperature [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    step: float
      To specify the confidence interval for the cricondenbar pressure
      calculation, the preliminary search is performed. This parameter
      regulates the step of this search in fraction units. For example,
      if it is necessary to find the upper bound of the confidence
      interval, then the next value of pressure will be calculated from
      the previous one using the formula: `Pnext = Pprev * (1. + step)`.
      Default is `0.1`.

    Pmin: float
      During the preliminary search, the pressure can not drop below
      the lower limit. Otherwise, the `ValueError` will be rised.
      Default is `1.` [Pa].

    Pmax: float
      During the preliminary search, the pressure can not exceed the
      upper limit. Otherwise, the `ValueError` will be rised.
      Default is `1e8` [Pa].

    Returns
    -------
    Cricondenbar point calculation results as an instance of the
    `SatResult`. Important attributes are:

    - `P` the cricondenbar pressure in [Pa],
    - `T` the cricondenbar temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase,
    - `success` a boolean flag indicating if the calculation completed
      successfully.

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    P0, kvi0, Plow, Pupp = self.search(P, T, yi, True, step, Pmin, Pmax)
    return self.solver(P0, T, yi, kvi0, Plow=Plow, Pupp=Pupp)


def _PmaxPT_solve_TPDeq_P(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  xi: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
) -> ScalarType:
  """Solves the TPD-equation using the PT-based equations of state for
  pressure at a constant temperature. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance functon. Newton's
  method is used to solve the TPD-equation.

  Parameters
  ----------
  P0: float
    Initial guess for pressure [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  Returns
  -------
  The root (pressure) of the TPD-equation.
  """
  k = 0
  Pk = P0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(f'Iter #{k}: Pk = {Pk/1e6} MPa, {TPD = }')
  while k < maxiter and np.abs(TPD) > tol:
    # dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    dTPDdlnP = Pk * xi.dot(dlnphixidP - dlnphiyidP)
    k += 1
    # Pk -= TPD / dTPDdP
    Pk *= np.exp(-TPD / dTPDdlnP)
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(f'Iter #{k}: Pk = {Pk/1e6} MPa, {TPD = }')
  return Pk


def _PmaxPT_solve_dTPDdTeq_T(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  xi: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
) -> tuple[ScalarType, VectorType, VectorType, ScalarType, ScalarType]:
  """Solves the cricondenbar equation using the PT-based equations of
  state for temperature at a constant pressure. The cricondenbar equation
  is the equation of equality to zero of the partial derivative of the
  tangent-plane distance with respect to temperature. Newton's method is
  used to solve the cricondenbar equation.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess for temperature [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT_d2T(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  Returns
  -------
  A tuple of:
  - the root (temperature) of the cricondenbar equation,
  - an array of the natural logarithms of fugacity coefficients of
    components in the trial phase,
  - the same for the initial composition of the mixture,
  - compressibility factors for both mixtures.
  - value of the cricondenbar equation.
  """
  k = 0
  Tk = T0
  lnphiyi, Zy, dlnphiyidT, d2lnphiyidT2 = eos.getPT_lnphii_Z_dT_d2T(P, Tk, yi)
  lnphixi, Zx, dlnphixidT, d2lnphixidT2 = eos.getPT_lnphii_Z_dT_d2T(P, Tk, xi)
  eq = xi.dot(dlnphixidT - dlnphiyidT)
  # print(f'Iter #{k}: Tk = {Tk-273.15} C, {eq = }')
  while np.abs(eq) > tol and k < maxiter:
    deqdT = xi.dot(d2lnphixidT2 - d2lnphiyidT2)
    k += 1
    Tk -= eq / deqdT
    lnphiyi, Zy, dlnphiyidT, d2lnphiyidT2 = eos.getPT_lnphii_Z_dT_d2T(P, Tk,
                                                                      yi)
    lnphixi, Zx, dlnphixidT, d2lnphixidT2 = eos.getPT_lnphii_Z_dT_d2T(P, Tk,
                                                                      xi)
    eq = xi.dot(dlnphixidT - dlnphiyidT)
    # print(f'Iter #{k}: Tk = {Tk-273.15} C, {eq = }')
  return Tk, lnphixi, lnphiyi, Zx, Zy, eq


def _PmaxPT_ss(
  P0: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Plow: ScalarType = 1.,
  Pupp: ScalarType = 1e8,
) -> SatResult:
  """Successive substitution (SS) method for the cricondenbar
  calculation using a PT-based equation of state. To find the
  cricondenbar, the algorithm solves a system of non-linear equations:

  - `Nc` equations of equilibrium of components in the mixture and in
    the trial phase,
  - the TPD-equation that represents the condition where the
    tangent-plane distance equals zero (this equation is linearized with
    pressure),
  - the cricondenbar equation which is the equation of equality to zero
    of the partial derivative of the tangent-plane distance function
    with respect to temperature (this equation is linearized with
    temperature).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the cricondenbar pressure [Pa].

  T0: float
    Initial guess of the cricondenbar temperature [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dT_d2T(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation and the cricondenbar equation solvers
    successfully if the absolute value of the equation is less than
    `tol_tpd`. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondenbar equation solvers. Default is `8`.

  Plow: float
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1.0` [Pa].

  Pupp: float
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1e8` [Pa].

  Returns
  -------
  The cricondenbar point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondenbar point,
  - `T` the temperature in [K] of the cricondenbar point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info('Cricondenbar calculation using the SS-method.')
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Prs, Pa', 'Tmp, K', 'gnorm', 'TPD', 'dTPDdT'
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %8.2f %9.2e %10.2e %10.2e'
  solverTPDeq = partial(_PmaxPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCBAReq = partial(_PmaxPT_solve_dTPDdTeq_T, eos=eos, tol=tol_tpd,
                         maxiter=maxiter_tpd)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Pk, _, _, _, _, TPD = _PsatPT_solve_TPDeq_P(P0, T0, yi, xi, eos, tol_tpd,
                                              maxiter_tpd, Plow, Pupp, True)
  Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdT)
  while k < maxiter and (gnorm > tol or np.abs(TPD) > tol_tpd
                         or np.abs(dTPDdT) > tol_tpd):
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Pk = solverTPDeq(Pk, Tk, yi, xi)
    Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    TPD = xi.dot(gi - np.log(n))
    logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdT)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.abs(dTPDdT) < tol_tpd
      and np.isfinite(kik).all() and np.isfinite(Pk) and np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('The cricondenbar: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     success=True)
  else:
    logger.warning(
      "The SS-method for cricondenbar calculation terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T0 = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa",
      eos.name, P0, T0, yi, Plow, Pupp,
    )
    return SatResult(P=Pk, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _PmaxPT_qnss(
  P0: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Plow: ScalarType = 1.,
  Pupp: ScalarType = 1e8,
) -> SatResult:
  """Quasi-Newton Successive Substitution (SS) method for the
  cricondenbar calculation using a PT-based equation of state. To find
  the cricondenbar, the algorithm solves a system of non-linear
  equations:

  - `Nc` equations of equilibrium of components in the mixture and in
    the trial phase,
  - the TPD-equation that represents the condition where the
    tangent-plane distance equals zero (this equation is linearized with
    pressure),
  - the cricondenbar equation which is the equation of equality to zero
    of the partial derivative of the tangent-plane distance function
    with respect to temperature (this equation is linearized with
    temperature).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  For the details of the QNSS-method see: 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P0: float
    Initial guess of the cricondenbar pressure [Pa].

  T0: float
    Initial guess of the cricondenbar temperature [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dT_d2T(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation and the cricondenbar equation solvers
    successfully if the absolute value of the equation is less than
    `tol_tpd`. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondenbar equation solvers. Default is `8`.

  Plow: float
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1.0` [Pa].

  Pupp: float
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1e8` [Pa].

  Returns
  -------
  The cricondenbar point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondenbar point,
  - `T` the temperature in [K] of the cricondenbar point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info('Cricondenbar calculation using the QNSS-method.')
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Prs, Pa', 'Tmp, K', 'gnorm', 'TPD', 'dTPDdT'
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %8.2f %9.2e %10.2e %10.2e'
  solverTPDeq = partial(_PmaxPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCBAReq = partial(_PmaxPT_solve_dTPDdTeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Pk, _, _, _, _, TPD = _PsatPT_solve_TPDeq_P(P0, T0, yi, xi, eos, tol_tpd,
                                              maxiter_tpd, Plow, Pupp, True)
  Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  lmbd = 1.
  logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdT)
  while k < maxiter and (gnorm > tol or np.abs(TPD) > tol_tpd
                         or np.abs(dTPDdT) > tol_tpd):
    dlnki = -lmbd * gi
    max_dlnki = np.abs(dlnki).max()
    if max_dlnki > 6.:
      relax = 6. / max_dlnki
      lmbd *= relax
      dlnki *= relax
    k += 1
    tkm1 = dlnki.dot(gi)
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Pk = solverTPDeq(Pk, Tk, yi, xi)
    Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    TPD = xi.dot(gi - np.log(n))
    lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdT)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.abs(dTPDdT) < tol_tpd
      and np.isfinite(kik).all() and np.isfinite(Pk) and np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('The cricondenbar: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     success=True)
  else:
    logger.warning(
      "The QNSS-method for cricondenbar calculation terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T0 = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa",
      eos.name, P0, T0, yi, Plow, Pupp,
    )
    return SatResult(P=Pk, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _PmaxPT_newtC(
  P0: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Plow: ScalarType = 1.,
  Pupp: ScalarType = 1e8,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
) -> SatResult:
  """This function calculates the cricondenbar point by solving a system
  of nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirements that the tangent-plane distance and its
  partial derivative with respect to temperature are zero. The
  TPD- and cricondenbar equations are solved in the inner while-loop.
  The TPD-equation that represents the condition where the
  tangent-plane distance equals zero (this equation is linearized with
  pressure). The cricondenbar equation is the equation of equality to
  zero of the partial derivative of the tangent-plane distance function
  with respect to temperature (this equation is linearized with
  temperature).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the cricondenbar pressure [Pa].

  T0: float
    Initial guess of the cricondenbar temperature [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dT_d2T(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`,

    - `name: str`
      The EOS name (for proper logging),

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation and the cricondenbar equation solvers
    successfully if the absolute value of the equation is less than
    `tol_tpd`. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondenbar equation solvers. Default is `8`.

  Plow: float
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1.0` [Pa].

  Pupp: float
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1e8` [Pa].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  The cricondenbar point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondenbar point,
  - `T` the temperature in [K] of the cricondenbar point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info("Cricondenbar calculation using Newton's method (C-form).")
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Prs, Pa', 'Tmp, K', 'gnorm', 'TPD', 'dTPDdT'
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %8.2f %9.2e %10.2e %10.2e'
  solverTPDeq = partial(_PmaxPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCBAReq = partial(_PmaxPT_solve_dTPDdTeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  I = np.eye(eos.Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, _, _, _, _, TPD = _PsatPT_solve_TPDeq_P(P0, T0, yi, xi, eos, tol_tpd,
                                              maxiter_tpd, Plow, Pupp, True)
  Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdT)
  while k < maxiter and (gnorm > tol or np.abs(TPD) > tol_tpd
                         or np.abs(dTPDdT) > tol_tpd):
    dlnphixidnj = eos.getPT_lnphii_Z_dnj(Pk, Tk, xi, n)[2]
    J = I + ni * dlnphixidnj
    try:
      dlnki = linsolver(J, -gi)
    except:
      dlnki = -gi
    k += 1
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Pk = solverTPDeq(Pk, Tk, yi, xi)
    Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    TPD = xi.dot(gi - np.log(n))
    gnorm = np.linalg.norm(gi)
    logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdT)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.abs(dTPDdT) < tol_tpd
      and np.isfinite(kik).all() and np.isfinite(Pk) and np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('The cricondenbar: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     success=True)
  else:
    logger.warning(
      "Newton's method (C-form) for cricondenbar calculation "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T0 = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa",
      eos.name, P0, T0, yi, Plow, Pupp,
    )
    return SatResult(P=Pk, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


class TmaxPT(TsatPT):
  """Cricondentherm calculation.

  Performs the cricondentherm point calculation using PT-based equations
  of state.

  Parameters
  ----------
  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P, T, yi, level) -> tuple[ndarray]`, where
      `P: float` is pressure [Pa], `T: float` is temperature [K],
      and `yi: ndarray`, shape `(Nc,)` is an array of components
      mole fractions, `Nc` is the number of components. This method
      is used to generate initial guesses of k-values.

    - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor.

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dP_d2P(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure.

    If the solution method would be `'newton'` then it also must have:

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  method: str
    Type of the solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'newton'` (Newton's method).

    Default is `'ss'`.

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval for temperature of the cricondentherm point. This
    dictionary is used to specify arguments for the stability test
    procedure. Default is an empty dictionary.

  kwargs: dict
    Other arguments for a cricondentherm-solver. It may contain such
    arguments as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others
    depending on the selected solver.

  Methods
  -------
  run(P, T, yi) -> SatResult
    This method performs the cricondentherm point calculation of the
    composition `yi: ndarray` with shape `(Nc,)` for given initial
    guesses of pressure `P: float` in [Pa] and temperature `T: float`
    in [K]. This method returns cricondentherm point calculation results
    as an instance of `SatResult`.

  search(P, T, yi) -> tuple[float, ndarray, float, float]
    This method performs the preliminary search to refine an initial
    guess of the cricondentherm temperature and find lower and upper
    bounds. It returns a tuple of:
    - the improved initial guess for the cricondentherm temperature
      [K],
    - the initial guess for k-values as ndarray of shape `(Nc,)`,
    - the lower bound of the cricondentherm temperature [K],
    - the upper bound of the cricondentherm temperature [K].
  """
  def __init__(
    self,
    eos: EOSPTType,
    method: str = 'ss',
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.solver = partial(_TmaxPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_TmaxPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_TmaxPT_newtC, eos=eos, **kwargs)
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    step: ScalarType = 0.1,
    Tmin: ScalarType = 173.15,
    Tmax: ScalarType = 973.15,
  ) -> SatResult:
    """Performs the cricindentherm point calculation for a mixture. To
    improve an initial guess of temperature, the preliminary search is
    performed.

    Parameters
    ----------
    P: float
      Initial guess of the cricondentherm pressure [Pa].

    T: float
      Initial guess of the cricondentherm temperature [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    step: float
      To specify the confidence interval for temperature of the
      cricondentherm point calculation, the preliminary search is
      performed. This parameter regulates the step of this search in
      fraction units. During the preliminary search, the next value of
      temperature will be calculated from the previous one using the
      formula: `Tnext = Tprev * (1. + step)`. Default is `0.1`.

    Tmin: float
      During the preliminary search, the temperature can not drop below
      the lower limit. Otherwise, the `ValueError` will be rised.
      Default is `173.15` [Pa].

    Tmax: float
      During the preliminary search, the temperature can not exceed the
      upper limit. Otherwise, the `ValueError` will be rised.
      Default is `973.15` [Pa].

    Returns
    -------
    Cricondentherm point calculation results as an instance of the
    `SatResult`. Important attributes are:

    - `P` the cricondentherm pressure in [Pa],
    - `T` the cricondentherm temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase,
    - `success` a boolean flag indicating if the calculation completed
      successfully.

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    T0, kvi0, Tlow, Tupp = self.search(P, T, yi, True, step, Tmin, Tmax)
    return self.solver(P, T0, yi, kvi0, Tlow=Tlow, Tupp=Tupp)


def _TmaxPT_solve_TPDeq_T(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  xi: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
) -> ScalarType:
  """Solves the TPD-equation using the PT-based equations of state for
  temperature at a constant pressure. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance functon. Newton's
  method is used to solve the TPD-equation.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess for temperature [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        temperature.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  Returns
  -------
  The root (temperature) of the TPD-equation.
  """
  k = 0
  Tk = T0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
  lnphixi, Zx, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(f'Iter #{k}: Tk = {Tk-273.15} C, {TPD = }')
  while k < maxiter and np.abs(TPD) > tol:
    # dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
    dTPDdlnT = Tk * xi.dot(dlnphixidT - dlnphiyidT)
    k += 1
    # Tk -= TPD / dTPDdT
    Tk *= np.exp(-TPD / dTPDdlnT)
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    lnphixi, Zx, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(f'Iter #{k}: Tk = {Tk-273.15} C, {TPD = }')
  return Tk


def _TmaxPT_solve_dTPDdPeq_P(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  xi: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
) -> tuple[ScalarType, VectorType, VectorType, ScalarType, ScalarType]:
  """Solves the cricondentherm equation using the PT-based equations of
  state for pressure at a constant temperature. The cricondentherm
  equation is the equation of equality to zero of the partial derivative
  of the tangent-plane distance with respect to pressure. Newton's
  method is used to solve the cricondentherm equation.

  Parameters
  ----------
  P0: float
    Initial guess for pressure [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  yti: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP_d2P(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the relative
    pressure change is less than `tol`. Default is `1e-6`.

  Returns
  -------
  A tuple of:
  - the root (pressure) of the cricondentherm equation,
  - an array of the natural logarithms of fugacity coefficients of
    components in the trial phase,
  - the same for the initial composition of the mixture,
  - compressibility factors for both mixtures.
  - value of the cricondentherm equation.
  """
  k = 0
  Pk = P0
  lnphiyi, Zy, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_Z_dP_d2P(Pk, T, yi)
  lnphixi, Zx, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_Z_dP_d2P(Pk, T, xi)
  eq = xi.dot(dlnphixidP - dlnphiyidP)
  deqdP = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
  dP = -eq / deqdP
  # print(f'Iter #{k}: Pk = {Pk/1e6} MPa, {dP = } Pa')
  while np.abs(dP) / Pk > tol and k < maxiter:
    k += 1
    Pk += dP
    lnphiyi, Zy, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_Z_dP_d2P(Pk, T,
                                                                      yi)
    lnphixi, Zx, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_Z_dP_d2P(Pk, T,
                                                                      xi)
    eq = xi.dot(dlnphixidP - dlnphiyidP)
    deqdP = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
    dP = -eq / deqdP
    # print(f'Iter #{k}: Pk = {Pk/1e6} MPa, {dP = } Pa')
  return Pk, lnphixi, lnphiyi, Zx, Zy, eq


def _TmaxPT_ss(
  P0: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Tlow: ScalarType = 173.15,
  Tupp: ScalarType = 973.15,
) -> SatResult:
  """Successive substitution (SS) method for the cricondentherm
  calculation using a PT-based equation of state. To find the
  cricondentherm, the algorithm solves a system of non-linear equations:

  - `Nc` equations of equilibrium of components in the mixture and in
    the trial phase,
  - the TPD-equation that represents the condition where the
    tangent-plane distance equals zero (this equation is linearized with
    temperature),
  - the cricondentherm equation which is the equation of equality to
    zero of the partial derivative of the tangent-plane distance
    function with respect to pressure (this equation is linearized with
    pressure).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the cricondentherm pressure [Pa].

  T0: float
    Initial guess of the cricondentherm temperature [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dP_d2P(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Terminate the
    cricondentherm equation solver successfully if the absolute value
    of the relative pressure change is less than `tol_tpd`. Default
    is `1e-6`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondentherm equation solvers. Default is `8`.

  Tlow: float
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `173.15` [K].

  Tupp: float
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `973.15` [K].

  Returns
  -------
  The cricondentherm point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondentherm point,
  - `T` the temperature in [K] of the cricondentherm point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info("Cricondentherm calculation using the SS-method.")
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Prs, Pa', 'Tmp, K', 'gnorm', 'TPD', 'dTPDdP',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %8.2f %9.2e %10.2e %10.2e'
  solverTPDeq = partial(_TmaxPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCTHERMeq = partial(_TmaxPT_solve_dTPDdPeq_P, eos=eos, tol=tol_tpd,
                           maxiter=maxiter_tpd)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Tk, _, _, _, _, TPD = _TsatPT_solve_TPDeq_T(P0, T0, yi, xi, eos, tol,
                                              maxiter, Tlow, Tupp, True)
  Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(P0, Tk, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdP)
  while k < maxiter and (gnorm > tol or np.abs(TPD) > tol_tpd
                         or np.abs(dTPDdP) > tol_tpd):
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Tk = solverTPDeq(Pk, Tk, yi, xi)
    Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    TPD = xi.dot(gi - np.log(n))
    logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdP)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.abs(dTPDdP) < tol_tpd
      and np.isfinite(kik).all() and np.isfinite(Pk) and np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('The cricondentherm: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     success=True)
  else:
    logger.warning(
      "The SS-method for cricondentherm calculation terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K",
      eos.name, P0, T0, yi, Tlow, Tupp,
    )
    return SatResult(P=Pk, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _TmaxPT_qnss(
  P0: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Tlow: ScalarType = 173.15,
  Tupp: ScalarType = 973.15,
) -> SatResult:
  """Quasi-Newton Successive substitution (QNSS) method for the
  cricondentherm calculation using a PT-based equation of state. To find
  the cricondentherm, the algorithm solves a system of non-linear
  equations:

  - `Nc` equations of equilibrium of components in the mixture and in
    the trial phase,
  - the TPD-equation that represents the condition where the
    tangent-plane distance equals zero (this equation is linearized with
    temperature),
  - the cricondentherm equation which is the equation of equality to
    zero of the partial derivative of the tangent-plane distance
    function with respect to pressure (this equation is linearized with
    pressure).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  For the details of the QNSS see: 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P0: float
    Initial guess of the cricondentherm pressure [Pa].

  T0: float
    Initial guess of the cricondentherm temperature [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dP_d2P(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Terminate the
    cricondentherm equation solver successfully if the absolute value
    of the relative pressure change is less than `tol_tpd`. Default
    is `1e-6`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondentherm equation solvers. Default is `8`.

  Tlow: float
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `173.15` [K].

  Tupp: float
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `973.15` [K].

  Returns
  -------
  The cricondentherm point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondentherm point,
  - `T` the temperature in [K] of the cricondentherm point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info("Cricondentherm calculation using the QNSS-method.")
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Prs, Pa', 'Tmp, K', 'gnorm', 'TPD', 'dTPDdP',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %8.2f %9.2e %10.2e %10.2e'
  solverTPDeq = partial(_TmaxPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCTHERMeq = partial(_TmaxPT_solve_dTPDdPeq_P, eos=eos, tol=tol_tpd,
                           maxiter=maxiter_tpd)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Tk, _, _, _, _, TPD = _TsatPT_solve_TPDeq_T(P0, T0, yi, xi, eos, tol,
                                              maxiter, Tlow, Tupp, True)
  Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(P0, Tk, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  lmbd = 1.
  logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdP)
  while k < maxiter and (gnorm > tol or np.abs(TPD) > tol_tpd
                         or np.abs(dTPDdP) > tol_tpd):
    dlnki = -lmbd * gi
    max_dlnki = np.abs(dlnki).max()
    if max_dlnki > 6.:
      relax = 6. / max_dlnki
      lmbd *= relax
      dlnki *= relax
    k += 1
    tkm1 = dlnki.dot(gi)
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Tk = solverTPDeq(Pk, Tk, yi, xi)
    Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    TPD = xi.dot(gi - np.log(n))
    lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdP)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.abs(dTPDdP) < tol_tpd
      and np.isfinite(kik).all() and np.isfinite(Pk) and np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('The cricondentherm: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     success=True)
  else:
    logger.warning(
      "The QNSS-method for cricondentherm calculation terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K",
      eos.name, P0, T0, yi, Tlow, Tupp,
    )
    return SatResult(P=Pk, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _TmaxPT_newtC(
  P0: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Tlow: ScalarType = 173.15,
  Tupp: ScalarType = 973.15,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
) -> SatResult:
  """This function calculates the cricondenthern point by solving a
  system of nonlinear equations using Newton's method. The system
  incorporates the condition of equal fugacity for all components in
  both phases, as well as the requirements that the tangent-plane
  distance and its partial derivative with respect to pressure are
  zero. The TPD- and cricondentherm equations are solved in the inner
  while-loop. The TPD-equation that represents the condition where the
  tangent-plane distance equals zero (this equation is linearized with
  temperature). The cricondentherm equation is the equation of equality
  to zero of the partial derivative of the tangent-plane distance
  function with respect to pressure (this equation is linearized with
  pressure).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the cricondentherm pressure [Pa].

  T0: float
    Initial guess of the cricondentherm temperature [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dP_d2P(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of logarithms of the fugacity coefficients of `Nc`
        components,
      - the compressibility factor of the mixture,
      - an array with shape `(Nc,)` of first partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure,
      - an array with shape `(Nc,)` of second partial derivatives of
        logarithms of the fugacity coefficients with respect to
        pressure.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging),

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Terminate the
    cricondentherm equation solver successfully if the absolute value
    of the relative pressure change is less than `tol_tpd`. Default
    is `1e-6`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondentherm equation solvers. Default is `8`.

  Tlow: float
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `173.15` [K].

  Tupp: float
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `973.15` [K].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  The cricondentherm point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondentherm point,
  - `T` the temperature in [K] of the cricondentherm point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.info("Cricondentherm calculation using Newton's method (C-form).")
  Nc = eos.Nc
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *map(lambda s: 'lnkv' + s, map(str, range(Nc))),
    'Prs, Pa', 'Tmp, K', 'gnorm', 'TPD', 'dTPDdP',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %11.1f %8.2f %9.2e %10.2e %10.2e'
  solverTPDeq = partial(_TmaxPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCTHERMeq = partial(_TmaxPT_solve_dTPDdPeq_P, eos=eos, tol=tol_tpd,
                           maxiter=maxiter_tpd)
  I = np.eye(eos.Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Tk, _, _, _, _, TPD = _TsatPT_solve_TPDeq_T(P0, T0, yi, xi, eos, tol,
                                              maxiter, Tlow, Tupp, True)
  Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(P0, Tk, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdP)
  while k < maxiter and (gnorm > tol or np.abs(TPD) > tol_tpd
                         or np.abs(dTPDdP) > tol_tpd):
    dlnphixidnj = eos.getPT_lnphii_Z_dnj(Pk, Tk, xi, n)[2]
    J = I + ni * dlnphixidnj
    try:
      dlnki = linsolver(J, -gi)
    except:
      dlnki = -gi
    lnkik += dlnki
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Tk = solverTPDeq(Pk, Tk, yi, xi)
    Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    TPD = xi.dot(gi - np.log(n))
    logger.debug(tmpl, k, *lnkik, Pk, Tk, gnorm, TPD, dTPDdP)
  if (gnorm < tol and np.abs(TPD) < tol_tpd and np.abs(dTPDdP) < tol_tpd
      and np.isfinite(kik).all() and np.isfinite(Pk) and np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info('The cricondentherm: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     success=True)
  else:
    logger.warning(
      "Newton's method (C-form) for cricondentherm calculation terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K",
      eos.name, P0, T0, yi, Tlow, Tupp,
    )
    return SatResult(P=Pk, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


class EnvelopeResult(dict):
  """Container for phase envelope calculation outputs with
  pretty-printing.

  Attributes
  ----------
  Pk: ndarray, shape (Ns,)
    This array includes pressures [Pa] of the phase envelope that
    consists of `Ns` states (points).

  Tk: ndarray, shape (Ns,)
    This array includes temperatures [K] of the phase envelope that
    consists of `Ns` states (points).

  ykji: ndarray, shape (Ns, Np, Nc)
    The phase state boundary compises `Ns` states (points), each
    representing the mole fractions of `Nc` components in `Np` phases
    that are in equilibrium along the `Np`-phase envelope. These phase
    compositions were calculated and stored in this three-dimensional
    array.

  Zkj: ndarray, shape (Ns, Np)
    This two-dimensional array represents compressibility factors of
    phases that are in equilibrium along the `Np`-phase envelope
    consisting of `Ns` states (points).

  Pc: float
    Critical point pressure [Pa].

  Tc: float
    Critical point temperature [K].

  Pcb: float
    Cricondenbar point pressure [Pa].

  Tcb: float
    Cricondenbar point temperature [Pa].

  Pct: float
    Cricondentherm point pressure [Pa].

  Tct: float
    Cricondentherm point temperature [Pa].

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
      s = (f"Critical point: {self.Pc} Pa, {self.Tc} K\n"
           f"Cricondenbar: {self.Pcb} Pa, {self.Tcb} K\n"
           f"Cricondentherm: {self.Pct} Pa, {self.Tct} K\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


class env2pPT(PsatPT):
  """Two-phase envelope construction using a PT-based equation of state.

  Parameters
  ----------
  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P, T, yi, level) -> tuple[ndarray]`, where
      `P: float` is pressure [Pa], `T: float` is temperature [K],
      and `yi: ndarray`, shape `(Nc,)` is an array of components
      mole fractions, `Nc` is the number of components. This method
      is used to generate initial guesses of k-values.

    - `getPT_lnphii_Z_dP_dT_dyj(P, T, yi) -> tuple[ndarray, float, ndarray,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K] and composition
      (ndarray of shape `(Nc,)`), this method must return a tuple that
      contains:

      - a vector of logarithms of the fugacity coefficients of
        components (ndarray of shape `(Nc,)`),
      - the phase compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to pressure (ndarray of shape `(Nc,)`),
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to temperature (ndarray of shape `(Nc,)`),
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole fractions (ndarray of shape
        `(Nc, Nc)`) without taking into account the mole fraction
        constraint.

    If the contstruction of the approximate phase envelope is selected
    then it must have instead:

    - `getPT_lnphii_Z_dP_dT(P, T, yi) -> tuple[ndarray, float, ndarray,
                                               ndarray]`
      For a given pressure [Pa], temperature [K] and composition
      (ndarray of shape `(Nc,)`), this method must return a tuple that
      contains:

      - a vector of logarithms of the fugacity coefficients of
        components (ndarray of shape `(Nc,)`),
      - the phase compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to pressure (ndarray of shape `(Nc,)`),
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to temperature (ndarray of shape `(Nc,)`).

    Also, this instance must have attributes:

    - `mwi: ndarray`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  method: str
    This parameter allows to select the algorithm for the phase envelope
    construction. It should be one of:

    - `'base'`
      The standard algorithm of the phase envelope construction is based
      on the paper of M.L. Michelsen (doi: 10.1016/0378-3812(80)80001-X).

    - `'approx'`
      The algorithm is based on the paper of M.L. Michelsen
      (doi: 10.1016/0378-3812(94)80104-5). All points of the approximate
      phase envelope are located inside and close to the actual two-phase
      region boundary. The accuracy of the internal phase lines is
      expected to be poorer than that of the phase boundary. Therefore,
      this option is most substantial for systems with many components.
      Currently raises `NotImplementedError`.

    - `'bead-spring'`
      The algorithm is based on the paper of I.K. Nikolaidis (doi:
      10.1002/aic.15064). Currently raises `NotImplementedError`.

    Default is `'base'`.

  Pmin: float
    The minimum pressure [Pa] for phase envelope construction.
    This limit is also used by the saturation point solver. Default is
    `1.0` [Pa].

  Pmax: float
    The maximum pressure [Pa] for phase envelope construction.
    This limit is also used by the saturation point solver. Default is
    `1e8` [Pa].

  Tmin: float
    The minimum temperature [K] for phase envelope construction.
    This limit is also used by the saturation point solver. Default is
    `173.15` [K].

  Tmax: float
    The maximum temperature [K] for phase envelope construction.
    This limit is also used by the saturation point solver. Default is
    `937.15` [K].

  stopunstab: bool
    The flag indicates whether it is necessary to stop the phase
    envelope construction if both trial and real phase compositions
    along it are found unstable by the stability test. Enabling this
    option may prevent drawing false saturation lines but takes extra
    CPU time to conduct stability test for each point on the phase
    envelope. Default is `False`.

  stabkwargs: dict
    To clarify the initial guess for the first saturation point, a
    preliminary search using the stability test may be employed.
    Additionally, the stability test can be utilized to examine the
    stability of both trial and real phase compositions along the
    phase envelope. This parameter controls the settings of the
    stability test solver needed to perform the above-mentioned tasks.
    Default is an empty dictionary.

  kwargs: dict
    Other arguments for the two-phase envelope solver. It may contain
    such arguments as `tol`, `maxiter`, `miniter` and `linsolver`.

  Methods
  -------
  run(P0, T0, yi, Fv) -> EnvelopeResult
    This method should be used to run the envelope construction program,
    for which the initial guess of the saturation pressure `P0: float`
    in [Pa], starting temperature `T0: float` in [K], mole fractions
    of `Nc` components `yi: ndarray` with shape `(Nc,)`, and phase
    mole fraction `Fv: float` must be given. It returns the phase
    envelope construction results as an instance of the
    `EnvelopeResult`.
  """
  def __init__(
    self,
    eos: EOSPTType,
    method: str = 'base',
    Pmin: ScalarType = 1.,
    Pmax: ScalarType = 1e8,
    Tmin: ScalarType = 173.15,
    Tmax: ScalarType = 973.15,
    stopunstab: bool = False,
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    lnPmin = np.log(Pmin)
    lnPmax = np.log(Pmax)
    lnTmin = np.log(Tmin)
    lnTmax = np.log(Tmax)
    self.lnTmin = lnTmin
    self.lnPmin = lnPmin
    self.stopunstab = stopunstab
    if method == 'base':
      self.solver = partial(_env2pPT, eos=eos, lnPmin=lnPmin, lnPmax=lnPmax,
                            lnTmin=lnTmin, lnTmax=lnTmax, **kwargs)
    elif method == 'approx':
      raise NotImplementedError(
        'The construction of the approximate phase envelope is not '
        'implemented yet.'
      )
      # self.solver = partial(_xenv2pPT, eos=eos, **kwargs)
    elif method == 'bead-spring':
      raise NotImplementedError(
        'The bead spring method for the phase envelope construction is not '
        'implemented yet.'
      )
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P0: ScalarType,
    T0: ScalarType,
    yi: VectorType,
    Fv: ScalarType,
    sidx0: int = -2,
    improve_P0: bool = False,
    step0: ScalarType = 0.005,
    dlnkvnorm: ScalarType = 0.5,
    dlnPnorm: ScalarType = 0.15,
    dlnTnorm: ScalarType = 0.015,
    rdamp: ScalarType = 0.75,
    maxstep: ScalarType = 1.1,
    maxosfac: ScalarType = 2.,
    cfmax: ScalarType = 0.182,
    maxrepeats: int = 8,
    mindsval: ScalarType = 1e-12,
    maxpoints: int = 200,
    searchkwargs: dict = {},
  ) -> EnvelopeResult:
    """This method should be used to calculate the entire phase envelope.

    Parameters
    ----------
    P0: float
      Initial guess of the saturation pressure [Pa]. It is recommended
      to not use pressure limits (`Pmin`, `Pmax`) as a starting point
      for the phase diagram construction.

    T0: float
      Initial guess of the saturation temperature [K]. It is recomended
      to not use temperature limits (`Tmin`, `Tmax`) as a starting point
      for the phase diagram construction.

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components in a mixture.

    Fv: float
      Phase mole fraction for which the envelope is needed.

    sidx0: int
      For iteration zero, this parameter indexes the specified variable
      in an array of basic variables. The array of basic variables
      includes:

      - `Nc` natural logarithms of k-values of components,
      - the natural logarithm of pressure,
      - the natural logarithm of temperature.

      The specified variable is considered known and fixed for the
      algorithm of saturation point determination. Therefore, changing
      of this index may improve the algorithm converegence for the
      zeroth iteration. To initiate calculations, a pressure
      specification was recommended by M.L. Michelsen in his paper
      (doi: 10.1016/0378-3812(80)80001-X). The general rule for
      specified variable selection was also given in this paper. It was
      recommended to select the specified variable based on the largest
      rate of change, which refers to the largest derivative of the
      basic variables with respect to the specified variable. Default
      is `-2`.

    improve_P0: bool
      The flag indicates whether it is necessary to perform the
      preliminary search using the stability test to clarify the
      initial guess of the saturation pressure and corresponding
      k-values. Default is `False` which refers to considering the
      given value of `P0` as an initial guess and calculating the
      initial guess of k-values using the method
      `getPT_kvguess(P0, T0, yi)` of an initialized instance of
      an equation of state.

    step0: float
      The step size (the difference between two subsequent values of a
      specified variable) for iteration zero. It should be small enough
      to consider the saturation point found at the zeroth iteration as
      a good initial guess for the next saturation point calculation.
      Default is `0.005`.

    dlnkvnorm: float
      This parameter allows to specify the normal (expected) variation
      of components k-values when passing from a calculated saturation
      point to the next one. Expected variations of basic variables
      controls the step size and the specified variable change according
      to the foolowing formulas:

      .. math::

        S_{k+1} = S_k + \\lambda \\left( S_k - S_{k-1} \\right) ,

      where :math:`S_{k+1}` is the specified variable for
      :math:`\\left( k+1 \\right)`-th iteration, :math:`S_k` is the
      specified variable for :math:`\\left( k \\right)`-th iteration,
      :math:`S_{k-1}` is the specified variable for
      :math:`\\left( k-1 \\right)`-th iteration, and :math:`\\lambda`
      is the step size:

      .. math::

        \\lambda = \\frac{1 + r_d}{r_d + \\max_i \\left( \\frac{
        \\left| {x_i}_k - {x_i}_{k-1} \\right| }{ \\xi_i } \\right) } ,

      where :math:`r_d` is the damping factor (parameter `rdamp`),
      :math:`{x_i}_k` is the :math:`\\left( i \\right)`-th element
      of the array of basic variables at :math:`\\left( k \\right)`-th
      iteration :math:`{x_i}_{k-1}` is the :math:`\\left( i \\right)`-th
      element of the array of basic variables at
      :math:`\\left( k-1 \\right)`-th iteration, :math:`\\xi_i` is the
      :math:`\\left( i \\right)`-th element of the array of expected
      variations of basic variables.

      If the basic variable change between two iterations is greater than
      the corresponding expected variable change, then the step size
      will be reduced, and the damping factor will control the rate
      of such reduction.

      Default is `0.5`.

    dlnPnorm: float
      The normal (expected) change of the natural logarithm of pressure
      between two subsequent points of the phase envelope. Default is
      `0.15`.

    dlnTnorm: float
      The normal (expected) change of the natural logarithm of
      temperature between two subsequent points of the phase envelope.
      Default is `0.015`.

    rdamp: float
      The damping factor used for the calculation of the step size.
      Default is `0.75`.

    maxstep: float
      The step size calculated according to the above formula cannot be
      greater than the value specified by the `maxstep` parameter.
      Default is `1.1`.

    maxosfac: float
      Defines the maximum oscillation factor :math:`\\eta` for the basic
      variables. If the calculated saturation point at any iteration
      :math:`\\left( k \\right)` (or its initial guess) violates the
      following condition:

      .. math::

        \\left| {x_i}_k - {x_i}_{k-1} \\right| < \\eta \\xi_i,
        \\; i = 1 \\, \\ldots \\, N_c + 2 ,

      then the step size :math:`\\lambda` will be reduced by half and
      the iteration will be repeated. Default is `2`.

    cfmax: float
      M.L. Michelsen in his paper (doi: 10.1016/0378-3812(80)80001-X)
      recommended to use the 3rd-order polynomial extrapolation
      to provide an initial guess for the saturation point calculation.
      However, according to the papers of Agger and Sorenses, 2017
      (doi: 10.1021/acs.iecr.7b04246) and Xu and Li, 2023 (doi:
      10.1016/j.geoen.2023.212058) estimates from extrapolation in the
      critical or semicritical regions with high-order polynomials may
      lead to substantial error. Therefore, they recommended to use
      the linear extrapolation when crossing the critical point. In
      their paper, Xu and Li introduced the critical-region factor
      which is calculated according to the following formula:

      .. math::

        cf = \\frac{\\max_i K_i}{\\min_i K_i} - 1 ,

      where :math:`K_i` is the k-value of :math:`\\left( i \\right)`-th
      component. They suggest to use the following condition:

      .. math::

        cf < 0.2

      to determine whether the current saturation point is inside the
      near-critical region. This condition can be expressed through
      natural logarithms of k-values:

      .. math::

        \\max_i \\ln K_i - \\min_i \\ln K_i < \\ln 1.2 .

      Default of the `cfmax` is :math:`\\ln 1.2 \\approx 0.182`.
      To disable this option, enter zero for this parameter.

    maxrepeats: int
      The saturation point calculation can be repeated several times
      with a reduced step size if convergence of equations was not
      achieved or the condition of maximum oscillation was violated.
      This parameter allows to specify the maximum number of repeats
      for any saturation point calculation. If the number of repeats
      exceeds the given bound, then the construction of the phase
      envelope will be stopped. Default is `8`.

    mindsval: float
      Stop the phase envelope construction if the change in the
      specified variable is less than `mindsval`. Default is `1e-12`.

    maxpoints: int
      The maximum number of points of the phase envelope. Default is
      `200`.

    searchkwargs: dict
      Advanced parameters for the preliminary search function. For the
      details, see the method `search` of the class `PsatPT`. Default
      is an empty dictionary.

    Returns
    -------
    The phase envelope construction results as an instance of the
    `EnvelopeResult`.

    Raises
    ------
    The `ValueError` if the solution was not found for the zeroth
    saturation point (for the given initial guesses).
    """
    logger.info('Constructing the phase envelope for Fv = %s.', Fv)
    Nc = self.eos.Nc
    if improve_P0:
      P0, kvi0, _, _ = self.search(P0, T0, yi, **searchkwargs)
      x0 = np.log(np.hstack([kvi0, P0, T0]))
    else:
      x0 = np.log(np.hstack([self.eos.getPT_kvguess(P0, T0, yi)[0], P0, T0]))
    logger.info(
      '%3s %3s %5s %7s %4s %9s %6s' + Nc * ' %8s' + ' %8s %7s',
      'Npnt', 'Ncut', 'Niter', 'Step', 'Sidx', 'Sval', 'CF',
      *map(lambda s: 'lnkv' + s, map(str, range(Nc))), 'lnP', 'lnT',
    )
    tmpl = '%4s %4s %5s %7.4f %4s %9.4f %6s' + Nc * ' %8.4f' + ' %8.4f %7.4f'
    sidx = sidx0
    sval = x0[sidx]
    x0, y0ji, Z0j, dgdx, Niter, suc = self.solver(x0, sidx, sval, yi, Fv)
    if not suc:
      raise ValueError(
        'The saturation pressure was not found for the specified starting\n'
        f'temperature {T0 = } K and the initial guess {P0 = } Pa. It may\n'
        'be beneficial to modify the initial guess and/or the starting\n'
        'temperature. Additionally, increasing the number of iterations\n'
        'could prove helpful. To locate the initial guess inside the\n'
        'two-phase region, the `improve_P0` flag can be activated. Changing\n'
        'the basic variable using `sidx0` may also improve convergence.'
      )
    logger.info(tmpl, 0, 0, Niter, 0., sidx, sval, False, *x0)
    dxnorm = np.hstack([np.full((Nc,), dlnkvnorm), dlnPnorm, dlnTnorm])
    dxmax = maxosfac * dxnorm
    mdgds = np.zeros(shape=(Nc + 2,))
    mdgds[-1] = 1.
    sidx = np.argmax(np.abs(np.linalg.solve(dgdx, mdgds)))
    xk, ykji, Zkj = self.curve(yi, Fv, x0, step0, sidx, dxnorm, rdamp,
                               maxstep, dxmax, cfmax, maxrepeats, mindsval,
                               maxpoints)
    if (x0[-2] > self.lnPmin and x0[-1] > self.lnTmin
        and xk.shape[0] < maxpoints):
      xl, ylji, Zlj = self.curve(yi, Fv, x0, -step0, sidx, dxnorm, rdamp,
                                 maxstep, dxmax, cfmax, maxrepeats, mindsval,
                                 maxpoints - xk.shape[0])
      if xl.shape[0] > 1:
        xk = np.vstack([np.flipud(xk), xl[1:]])
        ykji = np.concatenate([np.flipud(ykji), [y0ji], ylji])
        Zkj = np.vstack([np.flipud(Zkj), [Z0j], Zlj])
    elif xk.shape[0] > 1:
      ykji = np.concatenate([[y0ji], ykji])
      Zkj = np.vstack([[Z0j], Zkj])
    Pk = np.exp(xk[:, -2])
    Tk = np.exp(xk[:, -1])
    logger.info('The phase envelope for Fv = %s was completed.', Fv)
    # dP = np.diff(Pk, prepend=0.)
    # dT = np.diff(Tk, prepend=0.)
    # crit = 0
    # cbar = 0
    # ctrm = 0
    # for i in range(xk.shape[0] - 1):
    #   if xk[i, 0] * xk[i+1, 0] < 0.:
    #     cidx = i
    #   if dP[i] / dT[i] > 0. and dP[i+1] / dT[i+1] < 0.:
    #     cbar = i
    #   if dP[i] / dT[i] < 0. and dP[i+1] / dT[i+1] > 0.:
    #     ctrm = i
    #   if crit and cbar and ctrm:
    #     break
    # if crit:
    #   ...
    # else:
    #   logger.warning(
    #     'The critical point was not found using the %s EOS for the mixture:'
    #     'yi = \n\t%s', eos.name, yi,
    #   )
    return EnvelopeResult(Pk=Pk, Tk=Tk, ykji=ykji, Zkj=Zkj, success=True)

  def curve(
    self,
    yi: VectorType,
    Fv: ScalarType,
    x0: VectorType,
    step0: ScalarType,
    sidx: int,
    dxnorm: VectorType,
    rdamp: ScalarType,
    maxstep: ScalarType,
    dxmax: VectorType,
    cfmax: ScalarType,
    maxrepeats: int,
    mindsval: ScalarType,
    maxpoints: int,
  ) -> tuple[MatrixType, TensorType, MatrixType]:
    """This method should be used to calculate a part of the phase
    envelope for a fixed direction.

    Parameters
    ----------
    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components in a mixture.

    Fv: float
      Phase mole fraction for which the envelope is needed.

    x0: ndarray, shape (Nc + 2,)
      The first point of the phase envelope. The array must contain
      `Nc + 2` items:

      - `Nc` natural logarithms of k-values of components,
      - the natural logarithm of pressure,
      - the natural logarithm of temperature.

    step0: float
      The step size (the difference between two subsequent values of a
      specified variable) for iteration zero. This parameter can also be
      used to specify the direction in which the phase envelope must be
      calculated by the algorithm. In general, the change in the sign of
      the initial step size would lead to a different branch of the
      phase envelope.

    sidx: int
      This parameter indexes the specified variable in an array of basic
      variables for the next saturation point calculation.

    dxnorm: ndarray, shape (Nc + 2,)
      An array of the normal (expected) variations of the basic
      variables.

    rdamp: float
      The damping factor used for the calculation of the step size.

    maxstep: float
      The upper bound for the step size.

    dxmax: ndarray, shape(Nc + 2,)
      An array of the maximum variations of the basic variables.
      Saturation point calculation will be repeated with a reduced step
      size if the current solution deviates from the previous one by
      more than `dxmax`.

    cfmax: float
      The critical-region factor, which is used to determine whether the
      current saturation point is inside the near-critical region.

    maxrepeats: int
      This parameter allows to specify the maximum number of repeats
      for any saturation point calculation. If the number of repeats
      exceeds the given bound, then the construction of the phase
      envelope will be stopped.

    mindsval: float
      Stop the phase envelope construction if the change in the
      specified variable is less than `mindsval`.

    maxpoints: int
      The maximum number of points of the phase envelope.

    Returns
    -------
    A tuple of:
    - a matrix of the shape (Ns, Nc + 2) of calculated saturation
      points (`Ns` arrays of basic variables that correspond to the
      solution of equations),
    - a tensor of the shape (Ns, 2, Nc) of mole fractions of components
      in the real and trial phases along the phase envelope,
    - a matrix of the shape (Ns, 2) of compressibility factors of the
      real and trial phases along the phase envelope.
    """
    Nc = self.eos.Nc
    Ncp2 = Nc + 2
    tmpl = '%4s %4s %5s %7.4f %4s %9.4f %6s' + Nc * ' %8.4f' + ' %8.4f %7.4f'
    xk = np.zeros(shape=(maxpoints, Ncp2))
    pows = np.array([2, 3])
    ones = np.ones(shape=(4, 1))
    M = np.empty(shape=(4, 4))
    mdgds = np.zeros(shape=(Ncp2,))
    mdgds[-1] = 1.
    ykji = []
    Zkj = []
    k = 0
    xk[k] = x0
    step = np.abs(step0)
    r = 0
    cf = x0[:-2].max() - x0[:-2].min() < cfmax
    kmax = maxpoints - 1
    dsval = np.sign(step0) * step * x0[sidx]
    sval = x0[sidx] + dsval
    while k < kmax and xk[k, -1] >= self.lnTmin and xk[k, -2] >= self.lnPmin:
      if r > maxrepeats:
        logger.warning(
          'The maximum number of step repeats has been reached.'
        )
        break
      if np.abs(dsval) < mindsval:
        logger.warning(
          'The minimum change of the specified variable has been reached.',
        )
        break
      if k > 3 and not cf:
        svals = xk[k-3:k+1, sidx][:, None]
        np.concatenate([ones, svals, np.power(svals, pows)], axis=1, out=M)
        C = np.linalg.solve(M, xk[k-3:k+1])
        sval2 = sval * sval
        xi = np.array([1., sval, sval2, sval2 * sval]).dot(C)
        if (np.abs(xi - xk[k]) > dxmax).any():
          xi = xk[k] + ((xk[k] - xk[k-1]) / (xk[k, sidx] - xk[k-1, sidx])
                        * (sval - xk[k, sidx]))
      elif k > 1 and cf:
        xi = xk[k] + ((xk[k] - xk[k-1]) / (xk[k, sidx] - xk[k-1, sidx])
                      * (sval - xk[k, sidx]))
      else:
        xi = xk[k]
      xkp1, yji, Zj, dgdx, Niter, suc = self.solver(xi, sidx, sval, yi, Fv)
      if suc:
        if (np.abs(xkp1 - xk[k]) > dxmax).any():
          step *= .5
          if k > 0:
            dsval = step * (xk[k, sidx] - xk[k-1, sidx])
            sval = xk[k, sidx] + dsval
          else:
            dsval = np.sign(step0) * step * x0[sidx]
            sval = x0[sidx] + dsval
          r += 1
          continue
        logger.info(tmpl, k + 1, r, Niter, step, sidx, sval, cf, *xkp1)
        if self.stopunstab:
          P = np.exp(xk[k, -2])
          T = np.exp(xk[k, -1])
          stabx = self.stabsolver.run(P, T, yji[0])
          staby = self.stabsolver.run(P, T, yji[1])
          if not (stabx.stable and staby.stable):
            logger.warning('The unstable region has been detected.')
            break
        cf = xkp1[:-2].max() - xkp1[:-2].min() < cfmax
        if cf:
          sidx = np.argmax(np.abs(np.linalg.solve(dgdx, mdgds)[:-2]))
        else:
          sidx = np.argmax(np.abs(np.linalg.solve(dgdx, mdgds)))
        step = (1. + rdamp) / (rdamp + np.max(np.abs(xkp1 - xk[k]) / dxnorm))
        # if Niter <= 1:
        #   step = 1.75
        # elif Niter == 2:
        #   step = 1.25
        # elif Niter == 3:
        #   step = 1.15
        # elif Niter == 4:
        #   step = 0.75
        # else:
        #   step = 0.35
        if np.abs(step) > maxstep:
          step = maxstep
        dsval = step * (xkp1[sidx] - xk[k, sidx])
        sval = xkp1[sidx] + dsval
        k += 1
        xk[k] = xkp1
        ykji.append(yji)
        Zkj.append(Zj)
        r = 0
      else:
        step *= .5
        if k > 0:
          dsval = step * (xk[k, sidx] - xk[k-1, sidx])
          sval = xk[k, sidx] + dsval
        else:
          dsval = np.sign(step0) * step * x0[sidx]
          sval = x0[sidx] + dsval
        r += 1
    xk = xk[:k+1]
    ykji = np.array(ykji)
    Zkj = np.array(Zkj)
    return xk, ykji, Zkj


def _env2pPT(
  x0: VectorType,
  sidx: int,
  sval: ScalarType,
  yi: VectorType,
  Fv: ScalarType,
  eos: EOSPTType,
  tol: ScalarType = 1e-14,
  maxiter: int = 5,
  miniter: int = 1,
  lnPmin: ScalarType = 0.,
  lnPmax: ScalarType = 18.42,
  lnTmin: ScalarType = 5.154,
  lnTmax: ScalarType = 6.881,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
) -> tuple[VectorType, MatrixType, VectorType, MatrixType, int, bool]:
  Nc = eos.Nc
  logger.debug(
    'Solving the system of phase boundary equations for: '
    'Fv = %.3f, sidx = %s, sval = %.4f', Fv, sidx, sval,
  )
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%8s%10s%10s', 'Nit',
    *map(lambda s: 'lnkv' + s, map(str, range(Nc))), 'lnP', 'lnT', 'gnorm',
    'dxnorm',
  )
  tmpl = '%3s' + Nc * ' %8.4f' + ' %8.4f %7.4f %9.2e %9.2e'
  J = np.zeros(shape=(Nc + 2, Nc + 2))
  J[-1, sidx] = 1.
  g = np.empty(shape=(Nc + 2,))
  I = np.eye(Nc)
  k = 0
  xk = x0.flatten()
  xk[sidx] = sval
  ex = np.exp(xk)
  P = ex[-2]
  T = ex[-1]
  lnkvi = xk[:-2]
  kvi = ex[:-2]
  di = 1. + Fv * (kvi - 1.)
  yli = yi / di
  yvi = kvi * yli
  (lnphivi, Zv, dlnphividP,
   dlnphividT, dlnphividyvj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yvi)
  (lnphili, Zl, dlnphilidP,
   dlnphilidT, dlnphilidylj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yli)
  g[:Nc] = lnkvi + lnphivi - lnphili
  g[Nc] = np.sum(yvi - yli)
  g[Nc+1] = xk[sidx] - sval
  gnorm = np.linalg.norm(g)
  dylidlnkvi = -Fv * yvi / di
  dyvidlnkvi = yvi + kvi * dylidlnkvi
  J[:Nc,:Nc] = I + dlnphividyvj * dyvidlnkvi - dlnphilidylj * dylidlnkvi
  J[-2,:Nc] = yvi / di
  J[:Nc,-2] = P * (dlnphividP - dlnphilidP)
  J[:Nc,-1] = T * (dlnphividT - dlnphilidT)
  dx = linsolver(J, -g)
  dx2 = dx.dot(dx)
  notsolved = dx2 > tol
  logger.debug(tmpl, k, *xk, gnorm, dx2)
  while (notsolved or k < miniter) and k < maxiter:
    k += 1
    xkp1 = xk + dx
    if xkp1[-2] > lnPmax:
      xkp1[-2] = .5 * (xk[-2] + lnPmax)
    elif xkp1[-2] < lnPmin:
      xkp1[-2] = .5 * (xk[-2] + lnPmin)
    if xkp1[-1] > lnTmax:
      xkp1[-1] = .5 * (xk[-1] + lnTmax)
    elif xkp1[-1] < lnTmin:
      xkp1[-1] = .5 * (xk[-1] + lnTmin)
    xk = xkp1
    ex = np.exp(xk)
    P = ex[-2]
    T = ex[-1]
    lnkvi = xk[:-2]
    kvi = ex[:-2]
    di = 1. + Fv * (kvi - 1.)
    yli = yi / di
    yvi = kvi * yli
    (lnphivi, Zv, dlnphividP,
     dlnphividT, dlnphividyvj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yvi)
    (lnphili, Zl, dlnphilidP,
     dlnphilidT, dlnphilidylj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yli)
    g[:Nc] = lnkvi + lnphivi - lnphili
    g[Nc] = np.sum(yvi - yli)
    g[Nc+1] = xk[sidx] - sval
    gnorm = np.linalg.norm(g)
    J[:Nc,:Nc] = I + dlnphividyvj * dyvidlnkvi - dlnphilidylj * dylidlnkvi
    J[-2,:Nc] = yvi / di
    J[:Nc,-2] = P * (dlnphividP - dlnphilidP)
    J[:Nc,-1] = T * (dlnphividT - dlnphilidT)
    dx = linsolver(J, -g)
    dx2 = dx.dot(dx)
    notsolved = dx2 > tol
    logger.debug(tmpl, k, *xk, gnorm, dx2)
  suc = not notsolved and np.isfinite(ex).all() and np.isfinite(dx2)
  rhol = yli.dot(eos.mwi) / Zl
  rhov = yvi.dot(eos.mwi) / Zv
  if rhov < rhol:
    yji = np.vstack([yvi, yli])
    Zj = np.array([Zv, Zl])
  else:
    yji = np.vstack([yli, yvi])
    Zj = np.array([Zl, Zv])
  return xk, yji, Zj, J, k, suc



def _xenv2pPT(
  x0: VectorType,
  alpha: VectorType,
  lnkpi: VectorType,
  Fv: ScalarType,
  zi: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-14,
  maxiter: int = 5,
  miniter: int = 1,
  lnPmin: ScalarType = 0.,
  lnPmax: ScalarType = 18.42,
  lnTmin: ScalarType = 5.154,
  lnTmax: ScalarType = 6.881,
  inplace: bool = True,
) -> None | tuple[VectorType]:
  k = 0
  logger.debug(
    'Solving the system of equations of the approximate phase boundary for: '
    'Fv = %.3f, alpha = %.3f', Fv, alpha,
  )
  logger.debug('%3s%9s%8s%10s%10s', 'Nit', 'lnP', 'lnT', 'gnorm', 'dx2')
  tmpl = '%3s %8.4f %7.4f %9.2e %9.2e'
  dx = np.empty_like(x0)
  Yi = zi * np.exp(alpha * lnkpi)
  yi = Yi / Yi.sum()
  lnkvi = np.log(yi / zi)
  if inplace:
    xk = x0
  else:
    xk = x0.flatten()
  P = np.exp(xk[0])
  T = np.exp(xk[1])
  lnphiyi, Zy, dlnphiyidP, dlnphiyidT = eos.getPT_lnphii_Z_dP_dT(P, T, yi)
  lnphizi, Zz, dlnphizidP, dlnphizidT = eos.getPT_lnphii_Z_dP_dT(P, T, zi)
  di = lnkvi + lnphiyi - lnphizi
  g1 = yi.dot(di)
  g2 = zi.dot(di)
  gnorm = np.sqrt(g1 * g1 + g2 * g2)
  ddidP = dlnphiyidP - dlnphizidP
  ddidT = dlnphiyidT - dlnphizidT
  dg1dlnP = P * yi.dot(ddidP)
  dg2dlnP = P * zi.dot(ddidP)
  dg1dlnT = T * yi.dot(ddidT)
  dg2dlnT = T * zi.dot(ddidT)
  D = 1. / (dg1dlnP * dg2dlnT - dg1dlnT * dg2dlnP)
  dx[0] = D * (dg1dlnT * g2 - dg2dlnT * g1)
  dx[1] = D * (dg2dlnP * g1 - dg1dlnP * g2)
  dx2 = dx.dot(dx)
  ongoing = dx2 > tol
  logger.debug(tmpl, k, *xk, gnorm, dx2)
  while dx2 > tol and k < maxiter:
    xkp1 = xk + dx
    if xkp1[0] > lnPmax:
      xkp1[0] = .5 * (xk[0] + lnPmax)
    elif xkp1[0] < lnPmin:
      xkp1[0] = .5 * (xk[0] + lnPmin)
    if xkp1[1] > lnTmax:
      xkp1[1] = .5 * (xk[1] + lnTmax)
    elif xkp1[1] < lnTmin:
      xkp1[1] = .5 * (xk[1] + lnTmin)
    if inplace:
      xk[:] = xkp1
    else:
      xk = xkp1
    k += 1
    P = np.exp(xk[0])
    T = np.exp(xk[1])
    lnphiyi, Zy, dlnphiyidP, dlnphiyidT = eos.getPT_lnphii_Z_dP_dT(P, T, yi)
    lnphizi, Zz, dlnphizidP, dlnphizidT = eos.getPT_lnphii_Z_dP_dT(P, T, zi)
    di = lnkvi + lnphiyi - lnphizi
    g1 = yi.dot(di)
    g2 = zi.dot(di)
    gnorm = np.sqrt(g1 * g1 + g2 * g2)
    ddidP = dlnphiyidP - dlnphizidP
    ddidT = dlnphiyidT - dlnphizidT
    dg1dlnP = P * yi.dot(ddidP)
    dg2dlnP = P * zi.dot(ddidP)
    dg1dlnT = T * yi.dot(ddidT)
    dg2dlnT = T * zi.dot(ddidT)
    D = 1. / (dg1dlnP * dg2dlnT - dg1dlnT * dg2dlnP)
    dx[0] = D * (dg1dlnT * g2 - dg2dlnT * g1)
    dx[1] = D * (dg2dlnP * g1 - dg1dlnP * g2)
    dx2 = dx.dot(dx)
    ongoing = dx2 > tol
    logger.debug(tmpl, k, *xk, gnorm, dx2)
  if inplace:
    pass
  else:
    return xk, lnkvi, np.array([Zy, Zz])

