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
      'For the initial guess P = %s Pa, the one-phase state is stable: %s',
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
        'Plow = %s Pa, the one-phase state is stable: %s',
        Plow, stabmin.stable,
      )
      if stabmin.stable:
        Pupp = Plow
      while stabmin.stable and Plow > Pmin:
        Plow *= c
        stabmin = self.stabsolver.run(Plow, T, yi)
        logger.debug(
          'Plow = %s Pa, the one-phase state is stable: %s',
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
        'Pupp = %s Pa, the one-phase state is stable: %s',
        Pupp, stabmax.stable,
      )
      if not stabmax.stable:
        Plow = Pupp
      while not stabmax.stable and Pupp < Pmax:
        Pupp *= c
        stabmax = self.stabsolver.run(Pupp, T, yi)
        logger.debug(
          'Pupp = %s Pa, the one-phase state is stable: %s',
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
        'Pupp = %s Pa, the one-phase state is stable: %s',
        Pupp, stabmax.stable,
      )
      while stabmax.stable and Pupp < Pmax:
        Pupp *= c
        stabmax = self.stabsolver.run(Pupp, T, yi)
        logger.debug(
          'Pupp = %s Pa, the one-phase state is stable: %s',
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
        'Plow = %s Pa, the one-phase state is stable: %s',
        Plow, stabmin.stable,
      )
      while not stabmin.stable and Plow > Pmin:
        Plow *= c
        stabmin = self.stabsolver.run(Plow, T, yi)
        logger.debug(
          'Plow = %s Pa, the one-phase state is stable: %s',
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
  logger.debug(
    'Saturation pressure calculation using the SS-method:\n'
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa',
    P0, T, yi, Plow, Pupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tgnorm = %s\n\tTPD = %s',
    k, kik, Pk, gnorm, TPD,
  )
  while (gnorm > tol or np.abs(TPD) > tol_tpd) and k < maxiter:
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tgnorm = %s\n\tTPD = %s',
      k, kik, Pk, gnorm, TPD,
    )
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
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tTPD = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, TPD, k,
    )
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
  logger.debug(
    'Saturation pressure calculation using the QNSS-method:\n'
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa',
    P0, T, yi, Plow, Pupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tgnorm = %s\n\tTPD = %s',
    k, kik, Pk, gnorm, TPD,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tPk = %s\n\tgnorm = %s\n\tTPD = %s',
      k, kik, Pk, gnorm, TPD,
    )
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
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tTPD = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, TPD, k,
    )
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
  logger.debug(
    "Saturation pressure calculation using Newton's method (A-form):\n"
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa',
    P0, T, yi, Plow, Pupp,
  )
  Nc = eos.Nc
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tgnorm = %s',
    k, kik, Pk, gnorm,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tgnorm = %s',
      k, kik, Pk, gnorm,
    )
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
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, k,
    )
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
  logger.debug(
    "Saturation pressure calculation using Newton's method (B-form):\n"
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa',
    P0, T, yi, Plow, Pupp,
  )
  Nc = eos.Nc
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tgnorm = %s',
    k, kik, Pk, gnorm,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tgnorm = %s',
      k, kik, Pk, gnorm,
    )
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
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, k,
    )
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
  logger.debug(
    "Saturation pressure calculation using Newton's method (C-form):\n"
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa',
    P0, T, yi, Plow, Pupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tgnorm = %s\n\tTPD = %s',
    k, kik, Pk, gnorm, TPD,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tgnorm = %s\n\tTPD = %s',
      k, kik, Pk, gnorm, TPD,
    )
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
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tTPD = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, TPD, k,
    )
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
      'For the initial guess T = %s K, the one-phase state is stable: %s',
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
        'Tlow = %s K, the one-phase state is stable: %s',
        Tlow, stabmin.stable,
      )
      if stabmin.stable:
        Tupp = Tlow
      while stabmin.stable and Tlow > Tmin:
        Tlow *= c
        stabmin = self.stabsolver.run(P, Tlow, yi)
        logger.debug(
          'Tlow = %s K, the one-phase state is stable: %s',
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
        'Tupp = %s K, the one-phase state is stable: %s',
        Tupp, stabmax.stable,
      )
      if not stabmax.stable:
        Tlow = Tupp
      while not stabmax.stable and Tupp < Tmax:
        Tupp *= c
        stabmax = self.stabsolver.run(P, Tupp, yi)
        logger.debug(
          'Tupp = %s K, the one-phase state is stable: %s',
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
        'Tupp = %s K, the one-phase state is stable: %s',
        Tupp, stabmax.stable,
      )
      while stabmax.stable and Tupp < Tmax:
        Tupp *= c
        stabmax = self.stabsolver.run(P, Tupp, yi)
        logger.debug(
          'Tupp = %s K, the one-phase state is stable: %s',
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
        'Tlow = %s K, the one-phase state is stable: %s',
        Tlow, stabmin.stable,
      )
      while not stabmin.stable and Tlow > Tmin:
        Tlow *= c
        stabmin = self.stabsolver.run(P, Tlow, yi)
        logger.debug(
          'Tlow = %s K, the one-phase state is stable: %s',
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
  logger.debug(
    'Saturation temperature calculation using the SS-method:\n'
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K',
    P, T0, yi, Tlow, Tupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tT = %s K\n\tgnorm = %s\n\tTPD = %s',
    k, kik, Tk, gnorm, TPD,
  )
  while (gnorm > tol or np.abs(TPD) > tol_tpd) and k < maxiter:
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tT = %s K\n\tgnorm = %s\n\tTPD = %s',
      k, kik, Tk, gnorm, TPD,
    )
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
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tTPD = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, TPD, k,
    )
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
  logger.debug(
    'Saturation temperature calculation using the QNSS-method:\n'
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K',
    P, T0, yi, Tlow, Tupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tgnorm = %s\n\tTPD = %s\n\tT = %s K',
    k, kik, gnorm, TPD, Tk,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tgnorm = %s\n\tTPD = %s\n\tT = %s K',
      k, kik, gnorm, TPD, Tk,
    )
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
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tTPD = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, TPD, k,
    )
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
  logger.debug(
    "Saturation temperature calculation using Newton's method (A-form):\n"
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K',
    P, T0, yi, Tlow, Tupp,
  )
  Nc = eos.Nc
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tT = %s K\n\tgnorm = %s',
    k, kik, Tk, gnorm,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tT = %s K\n\tgnorm = %s',
      k, kik, Tk, gnorm,
    )
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
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, k,
    )
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
  logger.debug(
    "Saturation temperature calculation using Newton's method (B-form):\n"
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K',
    P, T0, yi, Tlow, Tupp,
  )
  Nc = eos.Nc
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tT = %s K\n\tgnorm = %s',
    k, kik, Tk, gnorm,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tT = %s K\n\tgnorm = %s',
      k, kik, Tk, gnorm,
    )
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
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, k,
    )
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
  logger.debug(
    "Saturation temperature calculation using Newton's method (C-form):\n"
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K',
    P, T0, yi, Tlow, Tupp,
  )
  solverTPDeq = partial(_TsatPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tlow0=Tlow, Tupp0=Tupp,
                        increasing=upper)
  I = np.eye(eos.Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tT = %s K\n\tgnorm = %s\n\tTPD = %s',
    k, kik, Tk, gnorm, TPD,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tT = %s K\n\tgnorm = %s\n\tTPD = %s',
      k, kik, Tk, gnorm, TPD,
    )
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
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tTPD = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, TPD, k,
    )
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
  logger.debug(
    'The cricondenbar calculation using the SS-method:\n'
    '\tP0 = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa',
    P0, T0, yi, Plow, Pupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
    'gnorm = %s\n\tTPD = %s\n\tdTPDdT = %s',
    k, kik, Pk, Tk, gnorm, TPD, dTPDdT,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdT = %s',
      k, kik, Pk, Tk, gnorm, TPD, dTPDdT,
    )
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
    logger.info(
      'The cricondenbar for yi = %s:\n\t'
      'P = %s Pa\n\tT = %s K\n\tyti = %s\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdT = %s\n\tNiter = %s',
      yi, Pk, Tk, xi, gnorm, TPD, dTPDdT, k,
    )
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
  logger.debug(
    'The cricondenbar calculation using the QNSS-method:\n'
    '\tP0 = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa',
    P0, T0, yi, Plow, Pupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
    'gnorm = %s\n\tTPD = %s\n\tdTPDdT = %s',
    k, kik, Pk, Tk, gnorm, TPD, dTPDdT,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdT = %s',
      k, kik, Pk, Tk, gnorm, TPD, dTPDdT,
    )
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
    logger.info(
      'The cricondenbar for yi = %s:\n\t'
      'P = %s Pa\n\tT = %s K\n\tyti = %s\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdT = %s\n\tNiter = %s',
      yi, Pk, Tk, xi, gnorm, TPD, dTPDdT, k,
    )
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
  logger.debug(
    "The cricondenbar calculation using Newton's method (C-form):\n"
    "\tP0 = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tPlow = %s Pa\n\tPupp = %s Pa",
    P0, T0, yi, Plow, Pupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
    'gnorm = %s\n\tTPD = %s\n\tdTPDdT = %s',
    k, kik, Pk, Tk, gnorm, TPD, dTPDdT,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdT = %s',
      k, kik, Pk, Tk, gnorm, TPD, dTPDdT,
    )
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
    logger.info(
      'The cricondenbar for yi = %s:\n\t'
      'P = %s Pa\n\tT = %s K\n\tyti = %s\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdT = %s\n\tNiter = %s',
      yi, Pk, Tk, xi, gnorm, TPD, dTPDdT, k,
    )
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
  logger.debug(
    'The cricondentherm calculation using the SS-method:\n'
    '\tP0 = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K',
    P0, T0, yi, Tlow, Tupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
    'gnorm = %s\n\tTPD = %s\n\tdTPDdP = %s',
    k, kik, Pk, Tk, gnorm, TPD, dTPDdP,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdP = %s',
      k, kik, Pk, Tk, gnorm, TPD, dTPDdP,
    )
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
    logger.info(
      'The cricondentherm for yi = %s:\n\t'
      'P = %s Pa\n\tT = %s K\n\tyti = %s\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdP = %s\n\tNiter = %s',
      yi, Pk, Tk, xi, gnorm, TPD, dTPDdP, k,
    )
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
  logger.debug(
    'The cricondentherm calculation using the QNSS-method:\n'
    '\tP0 = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K',
    P0, T0, yi, Tlow, Tupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
    'gnorm = %s\n\tTPD = %s\n\tdTPDdP = %s',
    k, kik, Pk, Tk, gnorm, TPD, dTPDdP,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdP = %s',
      k, kik, Pk, Tk, gnorm, TPD, dTPDdP,
    )
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
    logger.info(
      'The cricondentherm for yi = %s:\n\t'
      'P = %s Pa\n\tT = %s K\n\tyti = %s\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdP = %s\n\tNiter = %s',
      yi, Pk, Tk, xi, gnorm, TPD, dTPDdP, k,
    )
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
  logger.debug(
    "The cricondentherm calculation using Newton's method (C-form):\n"
    '\tP0 = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTlow = %s K\n\tTupp = %s K',
    P0, T0, yi, Tlow, Tupp,
  )
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
  logger.debug(
    'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
    'gnorm = %s\n\tTPD = %s\n\tdTPDdP = %s',
    k, kik, Pk, Tk, gnorm, TPD, dTPDdP,
  )
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
    logger.debug(
      'Iteration #%s:\n\tki = %s\n\tP = %s Pa\n\tT = %s K\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdP = %s',
      k, kik, Pk, Tk, gnorm, TPD, dTPDdP,
    )
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
    logger.info(
      'The cricondentherm for yi = %s:\n\t'
      'P = %s Pa\n\tT = %s K\n\tyti = %s\n\t'
      'gnorm = %s\n\tTPD = %s\n\tdTPDdP = %s\n\tNiter = %s',
      yi, Pk, Tk, xi, gnorm, TPD, dTPDdP, k,
    )
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


# class env2pPT(object):
#   def __init__(self, eos: EOSPTType, psatkwargs: dict = {}, **kwargs) -> None:
#     self.eos = eos
#     self.solver = partial(_env2pPT, eos=eos, **kwargs)
#     pass

#   def run(
#     self,
#     yi: VectorType,
#     Plow: ScalarType = 1.,
#     Pupp: ScalarType = 1e8,
#     Tlow: ScalarType = 173.15,
#     Tupp: ScalarType = 973.15,
#     maxpoints: int = 200,
#     Pinit: ScalarType = 2e6,
#     dT: ScalarType = 1.,
#     dTmax: ScalarType = 10.,
#     dTmin: ScalarType = 1e-2,
#     dP: ScalarType = 50.,
#     Niternorm: int = 4,
#   ) -> EnvelopeResult:
#     Fv = 0.
#     logger.debug('Constructing the phase envelope for Fv = %s ...', Fv)
#     Penv = []
#     Tenv = []
#     yjienv = []
#     Zjenv = []
#     k = 0
#     Psat = self.psatsolver()
#     lnyi = np.log(yi)
#     lnkvi0 = np.log(stab.kvji[0])
#     Pk, lnkvi, yji, Zj, Niter, suc = self.solver_P(P0, Tk, yi, Fv, lnkvi0, Pmin, Pmax)
#     Penv.append(Pk)
#     Tenv.append(Tk)
#     yjienv.append(yji)
#     Zjenv.append(Zj)
#     alpha0 = 1.
#     for k in range(1, maxpoints):
#       if Niter > Niternorm:
#         dT *= 1. - step
#       else:
#         dT *= 1. + step
#       if dT > dTmax:
#         dT = dTmax
#       elif dT < dTmin:
#         raise ValueError
#       k += 1
#       Tk += dT
#       if Tk > 360.:
#         break
#       P0, lnkvi0, alpha0 = _xenv2pPT_P(Pk, alpha0, Tk, lnkvi, lnyi, yi, self.eos)
#       Pk, lnkvi, yji, Zj, Niter, suc = self.solver_P(P0, Tk, yi, Fv, lnkvi0, Plow, Pupp)
#       if 0. < alpha0 < 0.05:
#         alpha0 = -1.
#         lnkvi = -lnkvi
#         Fv = 1.
#       # stab1 = self.stabsolver.run(Pk, Tk, yji[0])
#       # stab2 = self.stabsolver.run(Pk, Tk, yji[1])
#       # print(f'The first phase is stable: {stab1.stable}')
#       # print(f'The second phase is stable: {stab2.stable}')
#       # if np.isclose(lnkvi, 0., atol=5e-2).all() and not switch:
#       #   Fv = 1.
#       #   lnkvi = -lnkvi
#       #   alpha0 = -1.
#       #   switch = True
#         # Pcurve = False
#       if suc:
#         Penv.append(Pk)
#         Tenv.append(Tk)
#         yjienv.append(yji)
#         Zjenv.append(Zj)
#       else:
#         # if Pcurve:
#         #   Tk -= .5 * dT
#         continue
#     Penv = np.array(Penv)
#     Tenv = np.array(Tenv)
#     yjienv = np.array(yjienv)
#     Zjenv = np.array(Zjenv)
#     return EnvelopeResult(Pk=Penv, Tk=Tenv, ykji=yjienv, Zkj=Zjenv)


# def _xenv2pPT_P(
#   P0: ScalarType,
#   alpha0: ScalarType,
#   T: ScalarType,
#   lnkpi: VectorType,
#   lnzi: VectorType,
#   zi: VectorType,
#   eos: EOSPTType,
#   tol: ScalarType = 1e-9,
#   maxiter: int = 10,
#   dalpha0: ScalarType = 1e-6,
# ) -> tuple[ScalarType, VectorType]:
#   logger.debug(
#     'Finding approximate pressure and trial phase composition for:\n\t'
#     'P0 = %s Pa\n\tT = %s K\n\tkpi = %s', P0, T, np.exp(lnkpi),
#   )
#   k = 0
#   alphak = alpha0
#   Pk = P0
#   Yi = np.exp(lnzi + alphak * lnkpi)
#   yi = Yi / Yi.sum()
#   lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
#   lnphizi, Zz, dlnphizidP = eos.getPT_lnphii_Z_dP(Pk, T, zi)
#   lnyi = np.log(yi)
#   di = lnyi + lnphiyi - lnzi - lnphizi
#   g1k = yi.dot(di)
#   g2k = zi.dot(di)
#   gnorm = np.sqrt(g1k * g1k + g2k * g2k)
#   logger.debug(
#     'Iteration #%s:\n\tP = %s Pa, alpha = %s, gnorm = %s',
#     k, Pk, alphak, gnorm,
#   )
#   while gnorm > tol and k < maxiter:
#     Yi = np.exp(lnzi + (alphak + dalpha0) * lnkpi)
#     yi = Yi / Yi.sum()
#     lnphiyi = eos.getPT_lnphii(Pk, T, yi)
#     di = np.log(yi) + lnphiyi - lnzi - lnphizi
#     g1kp1 = yi.dot(di)
#     g2kp1 = zi.dot(di)
#     ddidP = dlnphiyidP - dlnphizidP
#     dg1dlnP = Pk * yi.dot(ddidP)
#     dg2dlnP = Pk * zi.dot(ddidP)
#     dg1dalpha = (g1kp1 - g1k) / dalpha0
#     dg2dalpha = (g2kp1 - g2k) / dalpha0
#     D = 1. / (dg1dlnP * dg2dalpha - dg1dalpha * dg2dlnP)
#     dlnP = D * (dg1dalpha * g2kp1 - dg2dalpha * g1kp1)
#     dalpha = D * (dg2dlnP * g1kp1 - dg1dlnP * g2kp1)
#     k += 1
#     Pk *= np.exp(dlnP)
#     alphak += dalpha
#     if np.abs(dalpha / alphak) < tol:
#       break
#     Yi = np.exp(lnzi + alphak * lnkpi)
#     yi = Yi / Yi.sum()
#     lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
#     lnphizi, Zz, dlnphizidP = eos.getPT_lnphii_Z_dP(Pk, T, zi)
#     lnyi = np.log(yi)
#     di = lnyi + lnphiyi - lnzi - lnphizi
#     g1k = yi.dot(di)
#     g2k = zi.dot(di)
#     gnorm = np.sqrt(g1k * g1k + g2k * g2k)
#     logger.debug(
#       'Iteration #%s:\n\tP = %s Pa, alpha = %s, gnorm = %s',
#       k, Pk, alphak, gnorm,
#     )
#   return Pk, (lnyi - lnzi) * np.sign(alphak), alphak


# def _env2pPT(
#   x0: VectorType,
#   yi: VectorType,
#   Fv: ScalarType,
#   sidx: int,
#   sval: ScalarType,
#   eos: EOSPTType,
#   tol: ScalarType = 1e-5,
#   maxiter: int = 10,
#   linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
# ) -> tuple[ScalarType, VectorType, VectorType, ScalarType, ScalarType]:
#   logger.debug(
#     'Solving the system for the phase envelope with Fv = %s.\n\t'
#     'The specified variable index is: %s', Fv, sidx,
#   )
#   Nc = eos.Nc
#   J = np.zeros(shape=(Nc + 2, Nc + 2))
#   J[-1, sidx] = 1.
#   g = np.empty(shape=(Nc + 2,))
#   I = np.eye(Nc)
#   Fl = 1. - Fv
#   k = 0
#   xk = x0
#   ex = np.exp(xk)
#   P = ex[-2]
#   T = ex[-1]
#   lnkvi = xk[:-2]
#   kvi = ex[:-2]
#   di = 1. + Fv * (kvi - 1.)
#   yli = yi / di
#   yvi = kvi * yli
#   (lnphivi, Zv, dlnphividP,
#    dlnphividT, dlnphividyvj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yvi)
#   (lnphili, Zl, dlnphilidP,
#    dlnphilidT, dlnphilidylj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yli)
#   g[:Nc] = lnkvi + lnphivi - lnphili
#   g[Nc] = np.sum(yvi - yli)
#   g[Nc+1] = x[sidx] - sval
#   gnorm = np.linalg.norm(g)
#   logger.debug(
#     'Iteration #%s:\n\tkvi = %s\n\tP = %s Pa\n\tT = %s K\n\tgnorm = %s',
#     k, kvi, P, T, gnorm,
#   )
#   dylidlnkvi = -Fv * yvi / di
#   dyvidlnkvi = yvi + kvi * dylidlnkvi
#   while (gnorm > tol or k < 1) and k < maxiter:
#     J[:Nc,:Nc] = I + dlnphividyvj * dyvidlnkvi - dlnphilidylj * dylidlnkvi
#     J[-2,:Nc] = yvi / di
#     J[:Nc,-2] = Pk * (dlnphividP - dlnphilidP)
#     J[:Nc,-1] = Tk * (dlnphividT - dlnphilidT)
#     try:
#       dx = linsolver(J, -g)
#     except:
#       dx = -g
#     k += 1
#     xk += dx
#     ex = np.exp(x)
#     P = ex[-2]
#     T = ex[-1]
#     lnkvi = x[:-2]
#     kvi = ex[:-2]
#     di = 1. + Fv * (kvi - 1.)
#     yli = yi / di
#     yvi = kvi * yli
#     (lnphivi, Zv, dlnphividP,
#      dlnphividT, dlnphividyvj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yvi)
#     (lnphili, Zl, dlnphilidP,
#      dlnphilidT, dlnphilidylj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yli)
#     g[:Nc] = lnkvi + lnphivi - lnphili
#     g[Nc] = np.sum(yvi - yli)
#     g[Nc+1] = x[sidx] - sval
#     gnorm = np.linalg.norm(g)
#     logger.debug(
#       'Iteration #%s:\n\tkvi = %s\n\tP = %s Pa\n\tT = %s K\n\tgnorm = %s',
#       k, kvi, P, T, gnorm,
#     )
#   suc = gnorm < tol and np.isfinite(ex).all()
#   rhol = yli.dot(eos.mwi) / Zl
#   rhov = yvi.dot(eos.mwi) / Zv
#   if rhov < rhol:
#     yji = np.vstack([yvi, yli])
#     Zj = np.array([Zv, Zl])
#   else:
#     yji = np.vstack([yli, yvi])
#     Zj = np.array([Zl, Zv])
#   return P, T, lnkvi, yji, Zj, k, suc


# def _env2pPT_T(
#   P: ScalarType,
#   T0: ScalarType,
#   yi: VectorType,
#   Fv: ScalarType,
#   Tmin: ScalarType,
#   Tmax: ScalarType,
#   eos: EOSPTType,
#   tol: ScalarType = 1e-6,
#   maxiter: int = 20,
#   linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
# ) -> tuple[ScalarType, VectorType, VectorType, ScalarType, ScalarType,
#            MatrixType]:
#   pass
