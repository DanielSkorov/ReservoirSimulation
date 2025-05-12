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
  factor of a mixture using the van der Waals equation of state.

  Arguments
  ---------
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
    Returns the compressiblity factor of a mixture for given
    pressure `P: float` in [Pa], temperature `T: float` in [K] and
    `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc` components.

  getPT_lnphii(P, T, yi) -> ndarray
    Returns logarithms of the fugacity coefficients of `Nc`
    components.

  getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]
    Returns a tuple of logarithms of the fugacity coefficients of
    `Nc` components and the compressiblity factor of a mixture.

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
    """Computes the compressiblity factor of a mixture.

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
    The compressibility factor of a mixture calculated using the
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
    the compressiblity factor of a mixture.

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
    A tuple containing an array with the natural logarithm of
    the fugacity coefficient of each component, and the compressibility
    factor of a mixture.
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

    Arguments
    ---------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

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

    Arguments:
    ----------
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
  factor of a mixture using the modified Peng-Robinson equation
  of state.

  Arguments
  ---------
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
    Returns the compressiblity factor of a mixture for given
    pressure `P: float` in [Pa], temperature `T: float` in [K] and
    `yi: ndarray` of shape `(Nc,)` mole fractions of `Nc` components.

  getPT_lnphii(P, T, yi) -> ndarray
    Returns logarithms of the fugacity coefficients of `Nc`
    components.

  getPT_lnfi(P, T, yi) -> ndarray
    Returns logarithms of the fugacities of `Nc` components.

  getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]
    Returns a tuple of logarithms of the fugacity coefficients of
    `Nc` components and the compressiblity factor of a mixture.

  getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]
    Returns a tuple of a vector of logarithms of the fugacity
    coefficients of `Nc` components, the compressibility factor of
    a mixture, and a vector of partial derivatives of logarithms of
    the fugacity coefficients with respect to pressure.

  getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]
    Returns a tuple of a vector of logarithms of the fugacity
    coefficients of `Nc` components, the compressibility factor of
    a mixture, and a `(Nc, Nc)` matrix of partial derivatives of
    logarithms of the fugacity coefficients with respect to mole
    numbers of components. `n` is a mixture mole number [mol].

  getPT_lnfi_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]
    Returns a tuple of a vector of logarithms of the fugacities
    of `Nc` components, the compressibility factor of a mixture, and
    a `(Nc, Nc)` matrix of partial derivatives of logarithms of the
    fugacities of components with respect to their mole numbers.
    `n` is a mixture mole number.

  getPT_Zj(P, T, yji) -> ndarray
    Returns a vector of the compressibility factor for each phase
    composition in `yji: ndarray` of shape `(Np, Nc)`, where `Np` is
    the number of phases.

  getPT_lnphiji_Zj(P, T, yji) -> tuple[ndarray, ndarray]
    Returns a tuple of a matrix of shape `(Np, Nc)` of logarithms of
    fugacity coefficients of `Nc` components in `Np` phases for given
    matrix `yji: ndarray` of shape `(Np, Nc)` of phase compositions,
    and a vector of the compressibility factor for each phase.

  getPT_kvguess(P, T, yi, level) -> tuple[ndarray]
    Returns a tuple containing vectors of initial k-values.

  getVT_P(V, T, yi, n) -> float
    Returns pressure in [Pa] calculated using the EOS for given
    volume `V: float` in [m3], temperature `T: float` in [K],
    composition `yi: ndarray` of shape `(Nc,)`, where `Nc` is the
    number of components, and phase mole number `n: float` in [mol].

  getVT_lnfi_dnj(V, T, yi, n) -> tuple[ndarray, ndarray]
    Returns a tuple of an array with the natural logarithm of the
    fugacity for each component and a matrix of their partial
    derivatives with respect to component mole numbers.

  getVT_d3F(V, T, yi, zti, n) -> float
    Returns the cubic form of the Helmholtz energy Taylor series
    decomposition for given component nole numbers changes
    `zti: ndarray` of shape `(Nc)`, where `Nc` is the number of
    components.

  getVT_vmin(T, yi) -> float
    Returns the minimal molar volume.

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
    """Computes the compressiblity factor of a mixture.

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
    The compressibility factor of a mixture calculated using
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
    gphii = .35355339 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * .41421356) / (Z + B * 2.41421356))
    return -gZ + self.bi * (Z - 1.) / bm + gphii * fZ - PRT * self.vsi_bi

  def getPT_lnfi(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacities of components.

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
    An array with the natural logarithm of the fugacity of each
    component.
    """
    return self.getPT_lnphii(P, T, yi,) + np.log(P * yi)

  def getPT_lnphii_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType]:
    """Computes fugacity coefficients of components and
    the compressiblity factor of a mixture.

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
    A tuple containing an array with the natural logarithm of the
    fugacity coefficient of each component, and the compressibility
    factor of a mixture.
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
    gphii = .35355339 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * .41421356) / (Z + B * 2.41421356))
    return (
      -gZ + self.bi * (Z - 1.) / bm + gphii * fZ - PRT * self.vsi_bi,
      Z - PRT * yi.dot(self.vsi_bi),
    )

  def getPT_lnphii_Z_dP(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType, VectorType]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to pressure.

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
    A tuple of an array with the natural logarithm of the fugacity
    coefficient for each component, the compressibility factor of
    a mixture, and a vector of partial derivatives of logarithms of
    the fugacity coefficients with respect to pressure.
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
    gphii = .35355339 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * .41421356) / (Z + B * 2.41421356))
    lnphii = -gZ + self.bi * (Z - 1.) / bm + gphii * fZ - PRT * self.vsi_bi
    ddmdA = np.array([0., 1., -B])[:,None]
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])[:,None]
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / (Z - B * .41421356) - 1. / (Z + B * 2.41421356)
    dfZdB = (- 0.41421356 / (Z - B * 0.41421356)
             - 2.41421356 / (Z + B * 2.41421356))
    dAdP = alpham / (RT * RT)
    dBdP = bm / RT
    ddmdP = ddmdA * dAdP + ddmdB * dBdP
    dqdP = np.power(Z, np.array([2, 1, 0])).dot(ddmdP)
    dZdP = -dqdP / dqdZ
    dgZdP = dgZdZ * dZdP + dgZdB * dBdP
    dfZdP = dfZdZ * dZdP + dfZdB * dBdP
    dgphiidP = gphii * ((B * dAdP - A * dBdP) / (A * B))
    dlnphiidP = (-dgZdP + self.bi * dZdP / bm + fZ * dgphiidP + gphii * dfZdP
                 - self.vsi_bi / RT)
    return lnphii, Z - PRT * yi.dot(self.vsi_bi), dlnphiidP

  def getPT_lnphii_Z_dnj(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    n: ScalarType = 1.,
  ) -> tuple[VectorType, ScalarType, MatrixType]:
    """Computes fugacities of components and their partial derivatives
    with respect to component mole numbers.

    Arguments
    ---------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a phase [gmole].

    Returns
    -------
    A tuple of an array with the natural logarithm of the fugacity
    coefficient for each component, the compressibility factor of
    a mixture, and a `(Nc, Nc)` matrix of partial derivatives of
    logarithms of the fugacity coefficients with respect to component
    mole numbers.
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
    gphii = .35355339 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * .41421356) / (Z + B * 2.41421356))
    lnphii = -gZ + self.bi * (Z - 1.) / bm + gphii * fZ - PRT * self.vsi_bi
    ddmdA = np.array([0., 1., -B])[:,None]
    ddmdB = np.array([1., -2. - 6. * B, B * (2. + 3. * B) - A])[:,None]
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dgZdZ = 1. / (Z - B)
    dgZdB = -dgZdZ
    dfZdZ = 1. / (Z - B * .41421356) - 1. / (Z + B * 2.41421356)
    dfZdB = (- 0.41421356 / (Z - B * 0.41421356)
             - 2.41421356 / (Z + B * 2.41421356))
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
                  + (self.bi / bm**2)[:,None] * dbmdnj) * (.35355339 * A / B)
                 + gphii[:,None] * (dAdnj / A - dBdnj / B))
    dlnphiidnj = ((self.bi / bm)[:,None] * (dZdnj - (Z - 1.) / bm * dbmdnj)
                  + (fZ * dgphiidnj + gphii[:,None] * dfZdnj)
                  - dgZdnj)
    return lnphii, Z - PRT * yi.dot(self.vsi_bi), dlnphiidnj

  def getPT_lnfi_Z_dnj(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    n: ScalarType = 1.,
  ) -> tuple[VectorType, ScalarType, MatrixType]:
    """Computes fugacities of components and their partial derivatives
    with respect to component mole numbers.

    Arguments
    ---------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a phase [gmole].

    Returns
    -------
    A tuple of an array with the natural logarithm of the fugacity
    for each component, the compressibility factor of a mixture, and
    a `(Nc, Nc)` matrix of partial derivatives of logarithms of the
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

    Arguments
    ---------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

    yji: ndarray, shape (Np, Nc)
      Mole fractions matrix, where `Np` is the number of phases and
      `Nc` is the number of components.

    Returns
    -------
    An array of the compressibility factor for each phase.
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

    Arguments
    ---------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

    yji: ndarray, shape (Np, Nc)
      Mole fractions matrix, where `Np` is the number of phases and
      `Nc` is the number of components.

    Returns
    -------
    A tuple with an array of the natural logarithm of fugacity
    coefficients for each component in each phase, and an array of
    compressibility factors for each phase.
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
    gphiji = ((.35355339 * Aj / Bj)[:,None]
              * (2. / alphamj[:,None] * Sji - self.bi / bmj[:,None]))
    fZj = np.log((Zj - Bj * 0.41421356) / (Zj + Bj * 2.41421356))
    lnphiji = (self.bi * ((Zj - 1.) / bmj)[:,None]
               + gphiji * fZj[:,None]
               - np.log(Zj - Bj)[:,None]
               - PRT * self.vsi_bi)
    return (lnphiji, Zj - yji.dot(self.vsi_bi) * PRT)

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

    Arguments
    ---------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    level: int
      Regulates an output of this function. Available options are:

        - 0: Wilson's and inverse Wilson's equations;
        - 1: previous + the first and last pure components.

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
    A tuple containing vectors of initial k-values.
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
      u3i = np.cbrt(kvi)
      return kvi, 1. / kvi, upi, 1. / upi, u3i, 1. / u3i
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

    Arguments
    ---------
    V: float
      Volume of a mixture [m3].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a phase [gmole].

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

    Arguments
    ---------
    V: float
      Volume of a mixture [m3].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a phase [gmole].

    Returns
    -------
    A tuple of an array with the natural logarithm of the fugacity for
    each component and a matrix of their partial derivatives with
    respect to component mole numbers.
    """
    d1 = 2.41421356
    d2 = -.41421356
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

    Arguments
    ---------
    V: float
      Volume of a mixture [m3].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    zti: ndarray, shape (Nc,)
      Component mole number changes [gmole].

    n: float
      Mole number of a phase [gmole].

    Returns
    -------
    The cubic form of the Helmholtz energy Taylor series decomposition.
    """
    d1 = 2.41421356
    d2 = -.41421356
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

    Arguments
    ---------
    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The minimal molar volume.
    """
    return yi.dot(self.bi)

  @staticmethod
  def solve_eos(A: ScalarType, B: ScalarType) -> ScalarType:
    """Solves the modified Peng-Robinson equation of state.

    This method implements the Cardano's method to solve the cubic form
    of the modified Peng-Robinson equation of state, and selectes an
    appropriate root based on the comparison of the Gibbs energy
    difference.

    Arguments:
    ----------
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
                            + np.log((Z1 - B * 0.41421356)
                                     * (Z2 + B * 2.41421356)
                                     / (Z1 + B * 2.41421356)
                                     / (Z2 - B * 0.41421356))
                              * 0.35355339 * A / B)
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



