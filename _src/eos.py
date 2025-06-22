import logging
import numpy as np

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
)

from constants import (
  R,
)


logger = logging.getLogger('eos')


class vdw(object):
  """Van der Waals equation of state.

  Computes fugacities of `Nc` components and the compressibility
  factor of the mixture using the van der Waals equation of state.

  Parameters
  ----------
  Pci: ndarray, shape (Nc,)
    Critical pressures of `Nc` components [Pa].

  Tci: ndarray, shape (Nc,)
    Critical temperatures of `Nc` components [K].

  mwi: ndarray, shape (Nc,)
    Molar weights of `Nc` components [kg/mol].

  dij: ndarray, shape (Nc * (Nc - 1) // 2,)
    The lower triangle matrix of binary interaction coefficients
    of `Nc` components.

  Methods
  -------
  getPT_Z(P, T, yi) -> float
    Returns the compressiblity factor of the mixture for given
    pressure `P: float` in [Pa], temperature `T: float` in [K] and
    `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc` components.

  getPT_lnphii(P, T, yi) -> ndarray
    Returns logarithms of the fugacity coefficients of `Nc`
    components.

  getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]
    Returns a tuple of logarithms of the fugacity coefficients of
    `Nc` components and the compressiblity factor of the mixture.

  getPT_lnfi(P, T, yi) -> ndarray
    Returns logarithms of the fugacities of `Nc` components.

  getPT_kvguess(P, T, yi, level) -> tuple[ndarray]
    Returns a tuple containing vectors of initial k-values.

  solve_eos(A, B) -> float
    Returns the solution of the EOS corresponding to the lowest Gibb's
    energy. `A: float` and `B: float` are the coefficients used to
    express the EOS in the cubic form.
  """
  def __init__(
    self,
    Pci: VectorType,
    Tci: VectorType,
    mwi: VectorType,
    dij: VectorType,
  ) -> None:
    self.name = 'Van der Waals EOS'
    self.Nc = Pci.shape[0]
    self.Pci = Pci
    self.Tci = Tci
    self.mwi = mwi
    self.sqrtai = .649519052838329 * R * Tci / np.sqrt(Pci)
    self.bi = .125 * R * Tci / Pci
    D = np.zeros(shape=(self.Nc, self.Nc), dtype=Pci.dtype)
    D[np.tril_indices(self.Nc, -1)] = dij
    self.D = 1. - (D + D.T)
    pass

  def getPT_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> ScalarType:
    """Computes the compressiblity factor of the mixture.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The compressibility factor of the mixture calculated using the
    van der Waals equation of state.
    """
    Si = self.sqrtai * (yi * self.sqrtai).dot(self.D)
    am = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = am * P / (R * R * T * T)
    B = bm * P / (R * T)
    Z = self.solve_eos(A, B)
    return Z

  def getPT_lnphii(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacity coefficients of components.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    An array with the natural logarithm of the fugacity coefficient
    of each component.
    """
    Si = self.sqrtai * (yi * self.sqrtai).dot(self.D)
    am = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = am * P / (R * R * T * T)
    B = bm * P / (R * T)
    Z = self.solve_eos(A, B)
    return -np.log(Z - B) + B / bm / (Z - B) * self.bi - 2. * A / Z / am * Si

  def getPT_lnphii_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType]:
    """Computes fugacity coefficients of components and
    the compressiblity factor of the mixture.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple containing an array with the natural logarithm of
    the fugacity coefficient of each component, and the compressibility
    factor of the mixture.
    """
    Si = self.sqrtai * (yi * self.sqrtai).dot(self.D)
    am = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = am * P / (R * R * T * T)
    B = bm * P / (R * T)
    Z = self.solve_eos(A, B)
    return (
      -np.log(Z - B) + B / bm / (Z - B) * self.bi - 2. * A / Z / am * Si,
      Z,
    )

  def getPT_lnfi(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacities of components.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    An array with the natural logarithm of the fugacity of
    each component.
    """
    return self.getPT_lnphii(P, T, yi) + np.log(P * yi)

  def getPT_kvguess(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    level: int = 0,
  ) -> tuple[VectorType]:
    """Computes initial guess of k-values for given pressure,
    temperature and composition.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    level: int
      Regulates an output of this function. Available options are:

        - 0: Wilson's and inverse Wilson's equations.

      Default is `0`

    Returns
    -------
    A tuple containing vectors of initial k-values.
    """
    if level == 0:
      kvi = self.Pci * np.exp(5.3727 * (1. - self.Tci / T)) / P
      return kvi, 1. / kvi
    else:
      raise ValueError(f'Unsupported level number: {level}.')

  @staticmethod
  def solve_eos(A: ScalarType, B: ScalarType) -> ScalarType:
    """Solves the Van-der-Waals equation of state.

    This method implements the Cardano's method to solve the cubic form
    of the van der Waals equation of state, and selectes an appropriate
    root based on the comparison of the Gibbs energy difference.

    Parameters:
    -----------
    A: float
      The coefficient in the cubic form of the van der Waals
      equation of state.

    B: float
      The coefficient in the cubic form of the van der Waals
      equation of state.

    Returns
    -------
    The solution of the van der Waals equation of state corresponding
    to the lowest Gibb's energy.
    """
    b = -1. - B
    c = A
    d = -A * B
    p = (3. * c - b * b) / 3.
    q = (2. * b * b * b - 9. * b * c + 27. * d) / 27.
    s = q * q * .25 + p * p * p / 27.
    if s >= 0.:
      s_ = np.sqrt(s)
      u1 = np.cbrt(-q * .5 + s_)
      u2 = np.cbrt(-q * .5 - s_)
      return u1 + u2 - b / 3.
    else:
      t0 = (2. * np.sqrt(-p / 3.)
            * np.cos(np.arccos(1.5 * q * np.sqrt(-3. / p) / p) / 3.))
      x0 = t0 - b / 3.
      r = b + x0
      k = -d / x0
      D = np.sqrt(r * r - 4. * k)
      x1 = (-r + D) * .5
      x2 = (-r - D) * .5
      fdG = lambda Z1, Z2: (B * (Z2 - Z1) / (Z1 - B) / (Z2 - B)
        + np.log((Z2 - B) / (Z1 - B)) + 2. * A * (Z1 - Z2) / Z1 / Z2)
      if x2 > 0:
        dG = fdG(x0, x2)
        if dG < 0.:
          return x0
        else:
          return x2
      elif x1 > 0.:
        dG = fdG(x0, x1)
        if dG < 0.:
          return x0
        else:
          return x1
      else:
        return x0


class pr78(object):
  """Peng-Robinson (1978) equation of state.

  Computes fugacities of `Nc` components and the compressibility
  factor of the mixture using the modified Peng-Robinson equation
  of state.

  Parameters
  ----------
  Pci: ndarray, shape (Nc,)
    Critical pressures of `Nc` components [Pa].

  Tci: ndarray, shape (Nc,)
    Critical temperatures of `Nc` components [K].

  wi: ndarray, shape (Nc,)
    Acentric factors of `Nc` components.

  mwi: ndarray, shape (Nc,)
    Molar weights of `Nc` components [kg/mol].

  vsi: ndarray, shape (Nc,)
    Volume shift parameters of `Nc` components

  dij: ndarray, shape (Nc * (Nc - 1) // 2,)
    The lower triangle matrix of binary interaction coefficients
    of `Nc` components.

  Methods
  -------
  getPT_Z(P, T, yi) -> float
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K] and `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components, returns the compressiblity factor of the mixture.

  getPT_lnphii(P, T, yi) -> ndarray
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K] and `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components, returns logarithms of the fugacity coefficients of `Nc`
    components.

  getPT_lnfi(P, T, yi) -> ndarray
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K] and `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components, returns logarithms of the fugacities of `Nc` components.

  getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K] and `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components, returns a tuple that contains:
    - logarithms of the fugacity coefficients of `Nc` components,
    - the compressiblity factor of the mixture..

  getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K] and `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components, returns a tuple that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - an array with shape `(Nc,)` of partial derivatives of logarithms
      of the fugacity coefficients with respect to pressure.

  getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K] and `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components, returns a tuple that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - an array with shape `(Nc,)` of partial derivatives of logarithms
      of the fugacity coefficients with respect to temperature.

  getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K], `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components and mixture mole number `n: float`, returns a tuple
    that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - a `(Nc, Nc)` matrix of partial derivatives of logarithms of
      the fugacity coefficients with respect to mole numbers of
      components.

  getPT_lnphii_Z_dnj_dP(P, T, yi, n) -> tuple[ndarray, float, ndarray, ndarray]
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K], `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components and mixture mole number `n: float`, returns a tuple
    that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - a `(Nc, Nc)` matrix of partial derivatives of logarithms of
      the fugacity coefficients with respect to mole numbers of
      components,
    - an array of shape `(Nc,)` of partial derivatives of logarithms
      of the fugacity coefficients with respect to pressure.

  getPT_lnphii_Z_dnj_dT(P, T, yi, n) -> tuple[ndarray, float, ndarray, ndarray]
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K], `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components and mixture mole number `n: float`, returns a tuple
    that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - a `(Nc, Nc)` matrix of partial derivatives of logarithms of
      the fugacity coefficients with respect to mole numbers of
      components,
    - an array of shape `(Nc,)` of partial derivatives of logarithms
      of the fugacity coefficients with respect to temperature.

  getPT_lnfi_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K], `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components and mixture mole number `n: float`, returns a tuple
    that contains:
    - an array of logarithms of the fugacities of `Nc` components,
    - the compressibility factor of the mixture,
    - a `(Nc, Nc)` matrix of partial derivatives of logarithms of the
      fugacities of components with respect to their mole numbers.

  getPT_Zj(P, T, yji) -> ndarray
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K], `yji: ndarray` of shape `(Np, Nc)` mole fractions of `Nc`
    components in `Np` phases, returns an array with shape `(Np,)` of
    the compressibility factor for each phase composition.

  getPT_lnphiji_Zj(P, T, yji) -> tuple[ndarray, ndarray]
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K], `yji: ndarray` of shape `(Np, Nc)` mole fractions of `Nc`
    components in `Np` phases, returns a tuple that contains:
    - a matrix of shape `(Np, Nc)` of logarithms of fugacity
      coefficients of `Nc` components in `Np` phases,
    - an array of shape `(Np,)` of the compressibility factor for each
      phase.

  getPT_kvguess(P, T, yi, level) -> tuple[ndarray]
    For given pressure `P: float` in [Pa], temperature `T: float`
    in [K], `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components, and the parameter that regulates the output of this
    function `level: int`, returns a tuple containing arrays of
    initial k-values.

  getVT_P(V, T, yi, n) -> float
    For given volume `V: float` in [m3], temperature `T: float` in [K],
    composition `yi: ndarray` of shape `(Nc,)`, where `Nc` is the
    number of components, and phase mole number `n: float` in [mol]
    returns pressure in [Pa] calculated using the EOS.

  getVT_lnfi_dnj(V, T, yi, n) -> tuple[ndarray, ndarray]
    For given pressure volume `V: float` in [m3], temperature `T: float`
    in [K], `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components, and phase mole number `n: float` in [mol], returns
    a tuple that contains:
    - an array with the natural logarithm of the fugacity for each
      component,
    - a matrix of their partial derivatives with respect to component
      mole numbers.

  getVT_d3F(V, T, yi, zti, n) -> float
    For given pressure volume `V: float` in [m3], temperature `T: float`
    in [K], `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc`
    components, component nole numbers changes `zti: ndarray` of shape
    `(Nc)`, and phase mole number `n: float` in [mol] returns the cubic
    form of the Helmholtz energy Taylor series decomposition.

  getVT_vmin(T, yi) -> float
    For given temperature `T: float` in [K], `yi: ndarray` of shape
    `(Nc,)` mole fractions of `Nc` components returns the minimal
    molar volume of the mixture in [m3/mol].

  solve_eos(A, B) -> float
    Returns the solution of the EOS corresponding to the lowest Gibb's
    energy. `A: float` and `B: float` are the coefficients used to
    express the EOS in the cubic form.
  """
  def __init__(
    self,
    Pci: VectorType,
    Tci: VectorType,
    wi: VectorType,
    mwi: VectorType,
    vsi: VectorType,
    dij: VectorType,
  ) -> None:
    self.name = 'Peng-Robinson (1978) EOS'
    self.Nc = Pci.shape[0]
    self.Pci = Pci
    self.Tci = Tci
    self.wi = wi
    self.mwi = mwi
    self._Tci = 1. / np.sqrt(Tci)
    self.bi = 0.07779607390388849 * R * Tci / Pci
    self.sqrtai = 0.6761919320144113 * R * Tci / np.sqrt(Pci)
    w2i = wi * wi
    w3i = w2i * wi
    self.kappai = np.where(
      wi <= 0.491,
      0.37464 + 1.54226 * wi - 0.26992 * w2i,
      0.379642 + 1.48503 * wi - 0.164423 * w2i + 0.016666 * w3i,
    )
    D = np.zeros(shape=(self.Nc, self.Nc), dtype=Pci.dtype)
    D[np.tril_indices(self.Nc, -1)] = dij
    self.D = 1. - (D + D.T)
    self.vsi_bi = vsi * self.bi
    pass

  def getPT_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> ScalarType:
    """Computes the compressiblity factor of the mixture.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The compressibility factor of the mixture calculated using
    the modified Peng-Robinson equation of state.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    return Z - yi.dot(self.vsi_bi) * PRT

  def getPT_lnphii(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacity coefficients of components.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    An array with the natural logarithm of the fugacity coefficient
    of each component.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * 0.414213562373095) / (Z + B * 2.414213562373095))
    return -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi

  def getPT_lnfi(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacities of components.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    An array with the natural logarithm of the fugacity of each
    component.
    """
    return self.getPT_lnphii(P, T, yi) + np.log(P * yi)

  def getPT_lnphii_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType]:
    """Computes fugacity coefficients of components and
    the compressiblity factor of the mixture.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - logarithms of the fugacity coefficients of `Nc` components,
    - the compressiblity factor of the mixture.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * 0.414213562373095) / (Z + B * 2.414213562373095))
    return (
      -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi,
      Z - PRT * yi.dot(self.vsi_bi),
    )

  def getPT_lnfi_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacities of components and the compressiblity factor
    of the mixture.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - logarithms of the fugacities of `Nc` components,
    - the compressiblity factor of the mixture.
    """
    lnphii, Z = self.getPT_lnphii_Z(P, T, yi)
    return lnphii + np.log(P * yi), Z

  def getPT_lnphii_Z_dP(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType, VectorType]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to pressure.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - an array with shape `(Nc,)` of partial derivatives of logarithms
      of the fugacity coefficients with respect to pressure.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi
    ddmdA = np.array([0., 1., -B])
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / ZmB - 1. / ZpB
    dfZdB = -0.414213562373095 / ZmB - 2.414213562373095 / ZpB
    dAdP = alpham / (RT * RT)
    dBdP = bm / RT
    ddmdP = ddmdA * dAdP + ddmdB * dBdP
    dqdP = np.power(Z, np.array([2, 1, 0])).dot(ddmdP)
    dZdP = -dqdP / dqdZ
    dgZdP = dgZdZ * dZdP + dgZdB * dBdP
    dfZdP = dfZdZ * dZdP + dfZdB * dBdP
    dlnphiidP = (-dgZdP + dZdP / bm * self.bi + gphii * dfZdP
                 - self.vsi_bi / RT)
    return lnphii, Z - PRT * yi.dot(self.vsi_bi), dlnphiidP

  def getPT_lnphii_Z_dT(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType, VectorType]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to temperature.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - an array with shape `(Nc,)` of partial derivatives of logarithms
      of the fugacity coefficients with respect to temperature.
    """
    RT = R * T
    PRT = P / RT
    sqrtT = np.sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = (0.3535533905932738 * fZ * gphii - gZ + (Z - 1.) / bm * self.bi
              - PRT * self.vsi_bi)
    ddmdA = np.array([0., 1., -B])
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / ZmB - 1. / ZpB
    dfZdB = -0.414213562373095 / ZmB - 2.414213562373095 / ZpB
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * self.D.dot(yi * dsqrtalphaidT)
    dalphamdT = yi.dot(dSidT)
    dAdT = PRT / RT * dalphamdT - 2. * A / T
    dBdT = -bm * PRT / T
    ddmdT = ddmdA * dAdT + ddmdB * dBdT
    dqdT = np.power(Z, np.array([2, 1, 0])).dot(ddmdT)
    dZdT = -dqdT / dqdZ
    dgZdT = dgZdZ * dZdT + dgZdB * dBdT
    dfZdT = dfZdZ * dZdT + dfZdB * dBdT
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - dgZdT + dZdT / bm * self.bi + PRT / T * self.vsi_bi)
    return lnphii, Z - PRT * yi.dot(self.vsi_bi), dlnphiidT

  def getPT_lnphii_Z_dnj(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    n: ScalarType = 1.,
  ) -> tuple[VectorType, ScalarType, MatrixType]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to mole numbers of components.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol].

    Returns
    -------
    A tuple that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - a `(Nc, Nc)` matrix of partial derivatives of logarithms of
      the fugacity coefficients with respect to mole numbers of
      components.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi
    ddmdA = np.array([0., 1., -B])[:,None]
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])[:,None]
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / ZmB - 1. / ZpB
    dfZdB = -0.414213562373095 / ZmB - 2.414213562373095 / ZpB
    dSidnj = (sqrtalphai[:,None] * sqrtalphai * self.D - Si[:,None]) / n
    dalphamdnj = 2. / n * (Si - alpham)
    dbmdnj = (self.bi - bm) / n
    dAdnj = dalphamdnj * PRT / RT
    dBdnj = dbmdnj * PRT
    ddmdnj = ddmdA * dAdnj + ddmdB * dBdnj
    dqdnj = np.power(Z, np.array([2, 1, 0])).dot(ddmdnj)
    dZdnj = -dqdnj / dqdZ
    dgZdnj = dgZdZ * dZdnj + dgZdB * dBdnj
    dfZdnj = dfZdZ * dZdnj + dfZdB * dBdnj
    dgphiidnj = ((2. / alpham * (dSidnj - (Si / alpham)[:,None] * dalphamdnj)
                  + (self.bi / (bm * bm))[:,None] * dbmdnj)
                 * (0.3535533905932738 * A / B)
                 + gphii[:,None] * (dAdnj / A - dBdnj / B))
    dlnphiidnj = ((self.bi / bm)[:,None] * (dZdnj - (Z - 1.) / bm * dbmdnj)
                  + (fZ * dgphiidnj + gphii[:,None] * dfZdnj)
                  - dgZdnj)
    return lnphii, Z - PRT * yi.dot(self.vsi_bi), dlnphiidnj

  def getPT_lnphii_Z_dnj_dP(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    n: ScalarType = 1.,
  ) -> tuple[VectorType, ScalarType, MatrixType, VectorType]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to mole numbers of components and pressure.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol].

    Returns
    -------
    A tuple of:
    - an array with the natural logarithm of the fugacity coefficient
      for each component,
    - the compressibility factor of the mixture,
    - a `(Nc, Nc)` matrix of partial derivatives of logarithms of the
      fugacity coefficients with respect to component mole numbers,
    - an array of partial derivatives of logarithms of the fugacity
      coefficients with respect to pressure
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi
    ddmdA = np.array([0., 1., -B])
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / ZmB - 1. / ZpB
    dfZdB = -0.414213562373095 / ZmB - 2.414213562373095 / ZpB
    dSidnj = (sqrtalphai[:,None] * sqrtalphai * self.D - Si[:,None]) / n
    dalphamdnj = 2. / n * (Si - alpham)
    dbmdnj = (self.bi - bm) / n
    dAdnj = dalphamdnj * PRT / RT
    dBdnj = dbmdnj * PRT
    ddmdnj = ddmdA[:,None] * dAdnj + ddmdB[:,None] * dBdnj
    dqdnj = np.power(Z, np.array([2, 1, 0])).dot(ddmdnj)
    dZdnj = -dqdnj / dqdZ
    dgZdnj = dgZdZ * dZdnj + dgZdB * dBdnj
    dfZdnj = dfZdZ * dZdnj + dfZdB * dBdnj
    dgphiidnj = ((2. / alpham * (dSidnj - (Si / alpham)[:,None] * dalphamdnj)
                  + (self.bi / (bm * bm))[:,None] * dbmdnj)
                 * (0.3535533905932738 * A / B)
                 + gphii[:,None] * (dAdnj / A - dBdnj / B))
    dlnphiidnj = ((self.bi / bm)[:,None] * (dZdnj - (Z - 1.) / bm * dbmdnj)
                  + (fZ * dgphiidnj + gphii[:,None] * dfZdnj)
                  - dgZdnj)
    dAdP = alpham / (RT * RT)
    dBdP = bm / RT
    ddmdP = ddmdA * dAdP + ddmdB * dBdP
    dqdP = np.power(Z, np.array([2, 1, 0])).dot(ddmdP)
    dZdP = -dqdP / dqdZ
    dgZdP = dgZdZ * dZdP + dgZdB * dBdP
    dfZdP = dfZdZ * dZdP + dfZdB * dBdP
    dlnphiidP = (-dgZdP + dZdP / bm * self.bi + gphii * dfZdP
                 - self.vsi_bi / RT)
    return lnphii, Z - PRT * yi.dot(self.vsi_bi), dlnphiidnj, dlnphiidP

  def getPT_lnphii_Z_dnj_dT(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    n: ScalarType = 1.,
  ) -> tuple[VectorType, ScalarType, MatrixType, VectorType]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to mole numbers of components and
    temperature.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol].

    Returns
    -------
    A tuple that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - a `(Nc, Nc)` matrix of partial derivatives of logarithms of
      the fugacity coefficients with respect to mole numbers of
      components,
    - an array with shape `(Nc,)` of partial derivatives of logarithms
      of the fugacity coefficients with respect to temperature.
    """
    RT = R * T
    PRT = P / RT
    sqrtT = np.sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = (0.3535533905932738 * fZ * gphii - gZ + (Z - 1.) / bm * self.bi
              - PRT * self.vsi_bi)
    ddmdA = np.array([0., 1., -B])
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / ZmB - 1. / ZpB
    dfZdB = -0.414213562373095 / ZmB - 2.414213562373095 / ZpB
    dSidnj = (sqrtalphai[:,None] * sqrtalphai * self.D - Si[:,None]) / n
    dalphamdnj = 2. / n * (Si - alpham)
    dbmdnj = (self.bi - bm) / n
    dAdnj = dalphamdnj * PRT / RT
    dBdnj = dbmdnj * PRT
    ddmdnj = ddmdA[:,None] * dAdnj + ddmdB[:,None] * dBdnj
    dqdnj = np.power(Z, np.array([2, 1, 0])).dot(ddmdnj)
    dZdnj = -dqdnj / dqdZ
    dgZdnj = dgZdZ * dZdnj + dgZdB * dBdnj
    dfZdnj = dfZdZ * dZdnj + dfZdB * dBdnj
    dgphiidnj = ((2. / alpham * (dSidnj - (Si / alpham)[:,None] * dalphamdnj)
                  + (self.bi / (bm * bm))[:,None] * dbmdnj) * (A / B)
                 + gphii[:,None] * (dAdnj / A - dBdnj / B))
    dlnphiidnj = ((self.bi / bm)[:,None] * (dZdnj - (Z - 1.) / bm * dbmdnj)
                  + (0.3535533905932738 * fZ * dgphiidnj
                     + (0.3535533905932738 * gphii)[:,None] * dfZdnj)
                  - dgZdnj)
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * self.D.dot(yi * dsqrtalphaidT)
    dalphamdT = yi.dot(dSidT)
    dAdT = PRT / RT * dalphamdT - 2. * A / T
    dBdT = -bm * PRT / T
    ddmdT = ddmdA * dAdT + ddmdB * dBdT
    dqdT = np.power(Z, np.array([2, 1, 0])).dot(ddmdT)
    dZdT = -dqdT / dqdZ
    dgZdT = dgZdZ * dZdT + dgZdB * dBdT
    dfZdT = dfZdZ * dZdT + dfZdB * dBdT
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - dgZdT + dZdT / bm * self.bi + PRT / T * self.vsi_bi)
    return lnphii, Z - PRT * yi.dot(self.vsi_bi), dlnphiidnj, dlnphiidT

  def getPT_lnphii_Z_dP_dT_dyj(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType, VectorType, VectorType, MatrixType]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to pressure, temperature and mole fractions
    of components pressure.

    The mole fraction constraint isn't taken into account.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple of:
    - an array with the natural logarithm of the fugacity coefficient
      for each component,
    - the compressibility factor of the mixture,
    - an array of partial derivatives of logarithms of the fugacity
      coefficients with respect to pressure,
    - an array of partial derivatives of logarithms of the fugacity
      coefficients with respect to temperature,
    - a `(Nc, Nc)` matrix of partial derivatives of logarithms of the
      fugacity coefficients with respect to component mole fractions.
    """
    RT = R * T
    PRT = P / RT
    sqrtT = np.sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi
    ddmdA = np.array([0., 1., -B])
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / ZmB - 1. / ZpB
    dfZdB = -0.414213562373095 / ZmB - 2.414213562373095 / ZpB
    dSidyj = sqrtalphai[:,None] * sqrtalphai * self.D
    dalphamdyj = Si + yi.dot(dSidyj)
    dbmdyj = self.bi
    dAdyj = dalphamdyj * PRT / RT
    dBdyj = dbmdyj * PRT
    ddmdyj = ddmdA[:,None] * dAdyj + ddmdB[:,None] * dBdyj
    dqdyj = np.power(Z, np.array([2, 1, 0])).dot(ddmdyj)
    dZdyj = -dqdyj / dqdZ
    dgZdyj = dgZdZ * dZdyj + dgZdB * dBdyj
    dfZdyj = dfZdZ * dZdyj + dfZdB * dBdyj
    dgphiidyj = ((2. / alpham * (dSidyj - (Si / alpham)[:,None] * dalphamdyj)
                  + (self.bi / (bm * bm))[:,None] * dbmdyj)
                 * (0.3535533905932738 * A / B)
                 + gphii[:,None] * (dAdyj / A - dBdyj / B))
    dlnphiidyj = ((self.bi / bm)[:,None] * (dZdyj - (Z - 1.) / bm * dbmdyj)
                  + (fZ * dgphiidyj + gphii[:,None] * dfZdyj)
                  - dgZdyj)
    dAdP = alpham / (RT * RT)
    dBdP = bm / RT
    ddmdP = ddmdA * dAdP + ddmdB * dBdP
    dqdP = np.power(Z, np.array([2, 1, 0])).dot(ddmdP)
    dZdP = -dqdP / dqdZ
    dgZdP = dgZdZ * dZdP + dgZdB * dBdP
    dfZdP = dfZdZ * dZdP + dfZdB * dBdP
    dlnphiidP = (-dgZdP + dZdP / bm * self.bi + gphii * dfZdP
                 - self.vsi_bi / RT)
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * self.D.dot(yi * dsqrtalphaidT)
    dalphamdT = yi.dot(dSidT)
    dAdT = PRT / RT * dalphamdT - 2. * A / T
    dBdT = -bm * PRT / T
    ddmdT = ddmdA * dAdT + ddmdB * dBdT
    dqdT = np.power(Z, np.array([2, 1, 0])).dot(ddmdT)
    dZdT = -dqdT / dqdZ
    dgZdT = dgZdZ * dZdT + dgZdB * dBdT
    dfZdT = dfZdZ * dZdT + dfZdB * dBdT
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - dgZdT + dZdT / bm * self.bi + PRT / T * self.vsi_bi)
    return (lnphii, Z - PRT * yi.dot(self.vsi_bi),
            dlnphiidP, dlnphiidT, dlnphiidyj)

  def getPT_lnphii_Z_dP_d2P(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType, VectorType, VectorType]:
    """Computes fugacity coefficients of components and their first
    and second partial derivatives with respect to pressure.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - an array with shape `(Nc,)` of partial derivatives of logarithms
      of the fugacity coefficients with respect to pressure.
    - an array with shape `(Nc,)` of second partial derivatives of
      logarithms of the fugacity coefficients with respect to pressure.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi
    ddmdA = np.array([0., 1., -B])
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / ZmB - 1. / ZpB
    dfZdB = -0.414213562373095 / ZmB - 2.414213562373095 / ZpB
    dAdP = alpham / (RT * RT)
    dBdP = bm / RT
    ddmdP = ddmdA * dAdP + ddmdB * dBdP
    m = np.array([2, 1, 0])
    Zpowm = np.power(Z, m)
    dqdP = Zpowm.dot(ddmdP)
    dZdP = -dqdP / dqdZ
    dgZdP = dgZdZ * dZdP + dgZdB * dBdP
    dfZdP = dfZdZ * dZdP + dfZdB * dBdP
    dlnphiidP = (-dgZdP + dZdP / bm * self.bi + gphii * dfZdP
                 - self.vsi_bi / RT)
    d2dmdB2 = np.array([0., -6., 2. + 6. * B])
    d2dmdAdB = np.array([0., 0., -1.])
    d2dmdP2 = dBdP * dBdP * d2dmdB2 + 2. * dAdP * dBdP * d2dmdAdB
    d2qdZ2 = 6. * Z + 2. * (B - 1.)
    d2qdP2 = Zpowm.dot(d2dmdP2)
    d2qdZdP = 2. * ddmdP[0] * Z + ddmdP[1]
    d2ZdP2 = - (d2qdP2 + 2. * dZdP * d2qdZdP + dZdP * dZdP * d2qdZ2) / dqdZ
    d2gZdZ2 = -dgZdZ * dgZdZ
    d2gZdB2 = d2gZdZ2
    d2gZdZdB = -d2gZdB2
    d2gZdP2 = (d2gZdZ2 * dZdP * dZdP + dgZdZ * d2ZdP2 + d2gZdB2 * dBdP * dBdP
               + 2. * d2gZdZdB * dZdP * dBdP)
    d2fZdZ2 = -1. / (ZmB * ZmB) + 1. / (ZpB * ZpB)
    d2fZdB2 = -0.17157287525381 / (ZmB * ZmB) + 5.82842712474619 / (ZpB * ZpB)
    d2fZdZdB = 0.414213562373095 / (ZmB * ZmB) + 2.414213562373095 / (ZpB*ZpB)
    d2fZdP2 = (d2fZdZ2 * dZdP * dZdP + dfZdZ * d2ZdP2 + d2fZdB2 * dBdP * dBdP
               + 2. * d2fZdZdB * dZdP * dBdP)
    d2lnphiidP2 = -d2gZdP2 + d2ZdP2 / bm * self.bi + d2fZdP2 * gphii
    return lnphii, Z - PRT * yi.dot(self.vsi_bi), dlnphiidP, d2lnphiidP2

  def getPT_lnphii_Z_dT_d2T(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType, VectorType, VectorType]:
    """Computes fugacity coefficients of components and their first
    and second partial derivatives with respect to temperature.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - an array of logarithms of the fugacity coefficients of `Nc`
      components,
    - the compressibility factor of the mixture,
    - an array with shape `(Nc,)` of first partial derivatives of
      logarithms of the fugacity coefficients with respect to
      temperature,
    - an array with shape `(Nc,)` of second partial derivatives of
      logarithms of the fugacity coefficients with respect to
      temperature.
    """
    RT = R * T
    PRT = P / RT
    sqrtT = np.sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = (-gZ + (Z - 1.) / bm * self.bi + 0.3535533905932738 * fZ * gphii
              - PRT * self.vsi_bi)
    ddmdA = np.array([0., 1., -B])
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / ZmB - 1. / ZpB
    dfZdB = -0.414213562373095 / ZmB - 2.414213562373095 / ZpB
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT_ = self.D.dot(yi * dsqrtalphaidT)
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * dSidT_
    dalphamdT = yi.dot(dSidT)
    dAdT = PRT / RT * dalphamdT - 2. * A / T
    dBdT = -bm * PRT / T
    ddmdT = ddmdA * dAdT + ddmdB * dBdT
    m = np.array([2, 1, 0])
    Zpowm = np.power(Z, m)
    dqdT = Zpowm.dot(ddmdT)
    dZdT = -dqdT / dqdZ
    dgZdT = dgZdZ * dZdT + dgZdB * dBdT
    dfZdT = dfZdZ * dZdT + dfZdB * dBdT
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - dgZdT + dZdT / bm * self.bi + PRT / T * self.vsi_bi)
    d2dmdB2 = np.array([0., -6., 2. + 6. * B])
    d2dmdAdB = np.array([0., 0., -1.])
    d2sqrtalphaidT2 = dsqrtalphaidT * (-.5 / T)
    d2SidT2 = (d2sqrtalphaidT2 * Si_ + 2. * dsqrtalphaidT * dSidT_
               + sqrtalphai * self.D.dot(yi * d2sqrtalphaidT2))
    d2alphamdT2 = yi.dot(d2SidT2)
    d2AdT2 = PRT / RT * (d2alphamdT2 - dalphamdT / T) - 3. * dAdT / T
    d2BdT2 = 2. * bm * PRT / (T * T)
    d2dmdT2 = (ddmdA * d2AdT2 + d2dmdB2 * dBdT * dBdT + ddmdB * d2BdT2
               + 2. * d2dmdAdB * dAdT * dBdT)
    d2qdT2 = Zpowm.dot(d2dmdT2)
    d2qdZdT = 2. * ddmdT[0] * Z + ddmdT[1]
    d2qdZ2 = 6. * Z + 2. * (B - 1.)
    d2ZdT2 = - (d2qdT2 + 2. * dZdT * d2qdZdT + dZdT * dZdT * d2qdZ2) / dqdZ
    d2gZdZ2 = -dgZdZ * dgZdZ
    d2gZdB2 = d2gZdZ2
    d2gZdZdB = -d2gZdB2
    d2gZdT2 = (d2gZdZ2 * dZdT * dZdT + dgZdZ * d2ZdT2 + d2gZdB2 * dBdT * dBdT
               + dgZdB * d2BdT2 + 2. * d2gZdZdB * dZdT * dBdT)
    d2fZdZ2 = -1. / (ZmB * ZmB) + 1. / (ZpB * ZpB)
    d2fZdB2 = -0.17157287525381 / (ZmB * ZmB) + 5.82842712474619 / (ZpB * ZpB)
    d2fZdZdB = 0.414213562373095 / (ZmB * ZmB) + 2.414213562373095 / (ZpB*ZpB)
    d2fZdT2 = (d2fZdZ2 * dZdT * dZdT + dfZdZ * d2ZdT2 + d2fZdB2 * dBdT * dBdT
               + dfZdB * d2BdT2 + 2. * d2fZdZdB * dZdT * dBdT)
    d2gphiidT2 = ((2. * d2SidT2 - d2alphamdT2 / bm * self.bi) / (RT * bm)
                  - 2. / T * dgphiidT)
    d2lnphiidT2 = (0.3535533905932738 * (fZ * d2gphiidT2 + gphii * d2fZdT2
                                         + 2. * dfZdT * dgphiidT)
                   + d2ZdT2 / bm * self.bi - d2gZdT2
                   - 2. * PRT / (T * T) * self.vsi_bi)
    return lnphii, Z - PRT * yi.dot(self.vsi_bi), dlnphiidT, d2lnphiidT2

  def getPT_lnfi_Z_dnj(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    n: ScalarType = 1.,
  ) -> tuple[VectorType, ScalarType, MatrixType]:
    """Computes fugacities of components and their partial derivatives
    with respect to component mole numbers.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol].

    Returns
    -------
    A tuple of:
    - an array of logarithms of the fugacities of `Nc` components,
    - the compressibility factor of the mixture,
    - a `(Nc, Nc)` matrix of partial derivatives of logarithms of the
      fugacities of components with respect to their mole numbers.
    """
    lnphii, Z, dlnphiidnj = self.getPT_lnphii_Z_dnj(P, T, yi, n)
    lnfi = lnphii + np.log(yi * P)
    dlnfidnj = dlnphiidnj + np.diagflat(1. / n / yi) - 1. / n
    return lnfi, Z, dlnfidnj

  def getPT_Zj(
    self,
    P: ScalarType,
    T: ScalarType,
    yji: MatrixType,
  ) -> VectorType:
    """Computes the compressibility factor for each composition
    of `Np` phases.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yji: ndarray, shape (Np, Nc)
      Mole fractions matrix, where `Np` is the number of phases and
      `Nc` is the number of components.

    Returns
    -------
    An array of shape `(Np,)` of the compressibility factor for each
    phase.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Sji = sqrtalphai * (yji * sqrtalphai).dot(self.D)
    alphamj = np.vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRT / RT
    Bj = bmj * PRT
    Zj = np.vectorize(self.solve_eos)(Aj, Bj) - yji.dot(self.vsi_bi) * PRT
    return Zj

  def getPT_lnphiji_Zj(
    self,
    P: ScalarType,
    T: ScalarType,
    yji: MatrixType,
  ) -> tuple[MatrixType, VectorType]:
    """Computes fugacity coefficients of components and
    the compressiblity factor for each composition of `Np` phases.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yji: ndarray, shape (Np, Nc)
      Mole fractions matrix, where `Np` is the number of phases and
      `Nc` is the number of components.

    Returns
    -------
    A tuple of:
    - a matrix of shape `(Np, Nc)` of logarithms of fugacity
      coefficients of `Nc` components in `Np` phases,
    - an array of shape `(Np,)` of the compressibility factor for each
      phase.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Sji = sqrtalphai * (yji * sqrtalphai).dot(self.D)
    alphamj = np.vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRT / RT
    Bj = bmj * PRT
    Zj = np.vectorize(self.solve_eos)(Aj, Bj)
    gphiji = ((0.3535533905932738 * Aj / Bj)[:,None]
              * (2. / alphamj[:,None] * Sji - self.bi / bmj[:,None]))
    fZj = np.log((Zj - Bj * 0.414213562373095)
                 / (Zj + Bj * 2.414213562373095))
    lnphiji = (self.bi * ((Zj - 1.) / bmj)[:,None]
               + gphiji * fZj[:,None]
               - np.log(Zj - Bj)[:,None]
               - PRT * self.vsi_bi)
    return lnphiji, Zj - yji.dot(self.vsi_bi) * PRT

  def getPT_kvguess(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    level: int = 0,
    idx: int = 0,
    eps: ScalarType = 1e-5,
  ) -> tuple[VectorType]:
    """Computes initial k-values for given pressure, temperature
    and composition.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    level: int
      Regulates an output of this function. Available options are:

      - 0: Wilson's and inverse Wilson's equations;
      - 1: previous + the pure component model;
      - 2: previous + cube roots of Wilson's and inverse Wilson's
           equations;
      - 3: previous + the ideal model (fugacity coefficient).

      Default is `0`.

    idx: int
      Index of a component that is characterized by a higher
      concentration in the trial phase. Default is `0`.

    eps: float
      Summarized mole fraction in the trial phase of other components
      except the component with the index `idx`. Must be greater than zero
      and lower than one. Default is `1e-5`.

    Returns
    -------
    A tuple containing arrays of initial k-values.
    """
    kvi = self.Pci * np.exp(5.3727 * (1. + self.wi) * (1. - self.Tci / T)) / P
    if level == 0:
      return kvi, 1. / kvi
    elif level == 1:
      upi = np.where(
        np.arange(self.Nc) == idx,
        (1. - eps) / yi[idx],
        eps / ((self.Nc - 1) * yi),
      )
      return kvi, 1. / kvi, upi, 1. / upi
    elif level == 2:
      upi = np.where(
        np.arange(self.Nc) == idx,
        (1. - eps) / yi[idx],
        eps / ((self.Nc - 1) * yi),
      )
      cbrtkvi = np.cbrt(kvi)
      return kvi, 1. / kvi, upi, 1. / upi, cbrtkvi, 1. / cbrtkvi
    elif level == 3:
      upi = np.where(
        np.arange(self.Nc) == idx,
        (1. - eps) / yi[idx],
        eps / ((self.Nc - 1) * yi),
      )
      cbrtkvi = np.cbrt(kvi)
      uii = np.exp(self.getPT_lnphii(P, T, yi))
      return kvi, 1. / kvi, upi, 1. / upi, cbrtkvi, 1. / cbrtkvi, uii
    else:
      raise ValueError(f'Unsupported level number: {level}.')

  def getVT_P(
    self,
    V: ScalarType,
    T: ScalarType,
    yi: VectorType,
    n: ScalarType = 1.,
  ) -> ScalarType:
    """Computes pressure for given volume, temperature and composition.

    Parameters
    ----------
    V: float
      Volume of the mixture [m3].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol].

    Returns
    -------
    Pressure in [Pa].
    """
    v = V / n
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    return R * T / (v - bm) - alpham / (v * v + 2. * bm * v - bm * bm)

  def getVT_lnfi_dnj(
    self,
    V: ScalarType,
    T: ScalarType,
    yi: VectorType,
    n: ScalarType = 1.,
  ) -> tuple[VectorType, MatrixType]:
    """Computes fugacities of components and their partial derivatives
    with respect to component mole numbers.

    Partial derivatives formulas were taken from the paper of M.L.
    Michelsen and R.A. Heidemann, 1981 (doi: 10.1002/aic.690270326).

    Parameters
    ----------
    V: float
      Volume of the mixture [m3].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol].

    Returns
    -------
    A tuple that contains:
    - an array with the natural logarithm of the fugacity for each
      component,
    - a matrix of their partial derivatives with respect to component
      mole numbers.
    """
    d1 = 2.414213562373095
    d2 = -0.414213562373095
    v = V / n
    RT = R * T
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    k = v / bm
    bti = self.bi / bm
    gmi = Si / alpham
    F0 = alpham / (bm * RT)
    F1 = 1. / (k - 1.)
    F2 = 2. * (d1 / (k + d1) - d2 / (k + d2)) / (d1 - d2)
    F3 = ((d1 / (k + d1))**2 - (d2 / (k + d2))**2) / (d1 - d2)
    F5 = 2. * np.log((k + d1) / (k + d2)) / (d1 - d2)
    F6 = F2 - F5
    lnfi = (np.log(yi * RT / (v - bm))
            + bti * (F1 - F0 * k / ((k + d2) * (k + d1)))
            - F5 / 2. * F0 * (2. * gmi - bti))
    aij = sqrtalphai[:,None] * sqrtalphai * self.D
    bij = bti[:,None] * bti
    cij = gmi[:,None] * bti
    Q = (np.diagflat(1. / yi) + (bti * F1)[:,None] + (bti * F1) + bij * F1**2
         + (bij * F3 - F5 / alpham * aij + F6 * (bij - cij - cij.T)) * F0) / n
    return lnfi, Q

  def getVT_d3F(
    self,
    V: ScalarType,
    T: ScalarType,
    yi: VectorType,
    zti: VectorType,
    n: ScalarType = 1.,
  ) -> ScalarType:
    """Computes the cubic form of the Helmholtz energy Taylor series
    decomposition for a given vector of component mole number changes.
    This method is used by the critical point calculation procedure.

    Calculation formulas were taken from the paper of M.L. Michelsen and
    R.A. Heidemann, 1981 (doi: 10.1002/aic.690270326).

    Parameters
    ----------
    V: float
      Volume of the mixture [m3].

    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    zti: ndarray, shape (Nc,)
      Component mole number changes [mol].

    n: float
      Mole number of the mixture [mol].

    Returns
    -------
    The cubic form of the Helmholtz energy Taylor series decomposition.
    """
    d1 = 2.414213562373095
    d2 = -0.414213562373095
    v = V / n
    RT = R * T
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    k = v / bm
    bti = self.bi / bm
    gmi = Si / alpham
    F1 = 1. / (k - 1.)
    F2 = 2. * (d1 / (k + d1) - d2 / (k + d2)) / (d1 - d2)
    F3 = ((d1 / (k + d1))**2 - (d2 / (k + d2))**2) / (d1 - d2)
    F4 = ((d1 / (k + d1))**3 - (d2 / (k + d2))**3) / (d1 - d2)
    F5 = 2. * np.log((k + d1) / (k + d2)) / (d1 - d2)
    F6 = F2 - F5
    zts = zti.sum()
    btm = zti.dot(bti)
    gmm = zti.dot(gmi)
    ti = sqrtalphai * self.D.dot(zti * sqrtalphai)
    tm = ti.dot(zti) / alpham
    C = (RT * (3. * zts * (btm * F1)**2 + 2. * (btm * F1)**3
               - np.power(zti, 3).dot(1. / (yi * yi)))
         + (3. * btm**2 * (2. * gmm - btm) * (F3 + F6) - 2. * btm**3 * F4
            - 3. * btm * tm * F6) * alpham / bm)
    return C / (n * n)

  def getVT_vmin(self, T: ScalarType, yi: VectorType) -> ScalarType:
    """Calculates the minimal molar volume.

    Parameters
    ----------
    T: float
      Temperature of the mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The minimal molar volume in [m3/mol].
    """
    return yi.dot(self.bi)

  @staticmethod
  def solve_eos(A: ScalarType, B: ScalarType) -> ScalarType:
    """Solves the modified Peng-Robinson equation of state.

    This method implements the Cardano's method to solve the cubic form
    of the modified Peng-Robinson equation of state, and selectes an
    appropriate root based on the comparison of the Gibbs energy
    difference.

    Parameters:
    -----------
    A: float
      The coefficient in the cubic form of the modified Peng-Robinson
      equation of state.

    B: float
      The coefficient in the cubic form of the modified Peng-Robinson
      equation of state.

    Returns
    -------
    The solution of the modified Peng-Robinson equation of state
    corresponding to the lowest Gibb's energy.
    """
    b = B - 1.
    c = A - 2. * B - 3. * B * B
    d = -A * B + B * B * (1. + B)
    p = (3. * c - b * b) / 3.
    q = (2. * b * b * b - 9. * b * c + 27. * d) / 27.
    s = q * q * .25 + p * p * p / 27.
    if s >= 0.:
      s_ = np.sqrt(s)
      u1 = np.cbrt(-q * .5 + s_)
      u2 = np.cbrt(-q * .5 - s_)
      return u1 + u2 - b / 3.
    else:
      t0 = (2. * np.sqrt(-p / 3.)
            * np.cos(np.arccos(1.5 * q * np.sqrt(-3. / p) / p) / 3.))
      x0 = t0 - b / 3.
      r = b + x0
      k = -d / x0
      D = np.sqrt(r * r - 4. * k)
      x1 = (-r + D) * .5
      x2 = (-r - D) * .5
      fdG = lambda Z1, Z2: (np.log((Z2 - B) / (Z1 - B))
                            + (Z1 - Z2)
                            + np.log((Z1 - B * 0.414213562373095)
                                     * (Z2 + B * 2.414213562373095)
                                     / (Z1 + B * 2.414213562373095)
                                     / (Z2 - B * 0.414213562373095))
                              * 0.3535533905932738 * A / B)
      if x2 > 0:
        dG = fdG(x0, x2)
        if dG < 0.:
          return x0
        else:
          return x2
      elif x1 > 0.:
        dG = fdG(x0, x1)
        if dG < 0.:
          return x0
        else:
          return x1
      else:
        return x0



