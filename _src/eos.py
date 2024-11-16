import logging
import numpy as np

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
)


logger = logging.getLogger('pvt')


R: ScalarType = np.float64(8.3144598)  # Universal gas constant [J/mol/K]


class vdw(object):
  """Van der Waals equation of state.

  Computes fugacities of `Nc` components and the compressibility
  factor of a mixture using the van der Waals equation of state.

  Arguments
  ---------
    Pci : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Critical pressures of `(Nc,)` components [Pa].

    Tci : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Critical temperatures of `(Nc,)` components [K].

    dij : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      The lower triangle matrix of binary interaction coefficients
      expressed as a numpy.array of the shape `(Nc * (Nc - 1) // 2,)`.
  """
  def __init__(
    self,
    Pci: VectorType,
    Tci: VectorType,
    dij: VectorType,
  ) -> None:
    self.Nc = Pci.shape[0]
    self.Pci = Pci
    self.Tci = Tci
    self.sqrtai = .649519052838329 * R * Tci / np.sqrt(Pci)
    self.bi = .125 * R * Tci / Pci
    D = np.zeros(shape=(self.Nc, self.Nc), dtype=Pci.dtype)
    D[np.tril_indices(self.Nc, -1)] = dij
    self.D = 1. - (D + D.T)
    pass


  def get_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> ScalarType:
    """Computes the compressiblity factor of a mixture.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns the compressibility factor of a mixture calculated using
    the van der Waals equation of state.
    """
    Si = self.sqrtai * (yi * self.sqrtai).dot(self.D)
    am = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = am * P / (R * R * T * T)
    B = bm * P / (R * T)
    Z = self.solve_eos(A, B)
    return Z

  def get_lnphii(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacity coefficients of components.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns an array with the natural logarithm of the fugacity
    coefficient of each component.
    """
    Si = self.sqrtai * (yi * self.sqrtai).dot(self.D)
    am = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = am * P / R / R / T / T
    B = bm * P / R / T
    Z = self.solve_eos(A, B)
    return -np.log(Z - B) + B / bm / (Z - B) * self.bi - 2. * A / Z / am * Si

  def get_lnphii_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType]:
    """Computes fugacity coefficients of components and
    the compressiblity factor of a mixture.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns a tuple containing an array with the natural logarithm of
    the fugacity coefficient of each component, and the compressibility
    factor of a mixture.
    """
    Si = self.sqrtai * (yi * self.sqrtai).dot(self.D)
    am = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = am * P / R / R / T / T
    B = bm * P / R / T
    Z = self.solve_eos(A, B)
    return (
      -np.log(Z - B) + B / bm / (Z - B) * self.bi - 2. * A / Z / am * Si,
      Z,
    )

  def get_lnfi(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacities of components.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns an array with the natural logarithm of the fugacity of
    each component.
    """
    return self.get_lnphii(P, T, yi) + np.log(P * yi)

  def get_kvguess(
    self,
    P: ScalarType,
    T: ScalarType,
    # level: int,
  ) -> MatrixType:
    """Computes initial guess of k-values for given pressure `P` and
    temperature `T` using the Wilson's correlation.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

    Returns a matrix containing two vectors of initial k-values
    calculated using the Wilson's correlation and inverse Wilson's
    correlation.
    """
    kvi = self.Pci * np.exp(5.3727 * (1. - self.Tci / T)) / P
    return np.vstack([kvi, 1. / kvi])

  @staticmethod
  def solve_eos(A: ScalarType, B: ScalarType) -> ScalarType:
    """Solves the Van-der-Waals equation of state.

    This method implements the Cardano's method to solve the cubic form
    of the van der Waals equation of state, and selectes an appropriate
    root based on the comparison of the Gibbs energy difference.

    Arguments:
    ----------
      A : numpy.float64
        Coefficient in the cubic form of the van der Waals
        equation of state.

      B : numpy.float64
        Coefficient in the cubic form of the van der Waals
        equation of state.

    Returns the solution of the van der Waals equation of state
    corresponding to the lowest Gibb's energy.
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
    Pci : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Critical pressures of `(Nc,)` components [Pa].

    Tci : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Critical temperatures of `(Nc,)` components [K].

    wi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Acentric factors of `(Nc,)` components.

    vsi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      Volume shift parameters of `(Nc,)` components

    dij : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
      The lower triangle matrix of binary interaction coefficients
      expressed as a numpy.array of the shape `(Nc * (Nc - 1) // 2,)`.
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
    self.Nc = Pci.shape[0]
    self.Pci = Pci
    self.Tci = Tci
    self.wi = wi
    self.mwi = mwi
    self._Tci = np.sqrt(1. / Tci)
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

  def get_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> ScalarType:
    """Computes the compressiblity factor of a mixture.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns the compressibility factor of a mixture calculated using
    the modified Peng-Robinson equation of state.
    """
    PRT = P / R / T
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / R / T
    B = bm * PRT
    Z = self.solve_eos(A, B)
    return Z - yi.dot(self.vsi_bi) * PRT

  def get_lnphii(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacity coefficients of components.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns an array with the natural logarithm of the fugacity
    coefficient of each component.
    """
    PRT = P / R / T
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / R / T
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = .35355339 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * .41421356) / (Z + B * 2.41421356))
    return -gZ + self.bi * (Z - 1.) / bm + gphii * fZ - PRT * self.vsi_bi

  def get_lnphii_Z(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, ScalarType]:
    """Computes fugacity coefficients of components and
    the compressiblity factor of a mixture.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns a tuple containing an array with the natural logarithm of
    the fugacity coefficient of each component, and the compressibility
    factor of a mixture.
    """
    PRT = P / R / T
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / R / T
    B = bm * PRT
    Z = self.solve_eos(A, B)
    gZ = np.log(Z - B)
    gphii = .35355339 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * .41421356) / (Z + B * 2.41421356))
    return (
      -gZ + self.bi * (Z - 1.) / bm + gphii * fZ - PRT * self.vsi_bi,
      Z - PRT * yi.dot(self.vsi_bi),
    )

  def get_Zj(
    self,
    P: ScalarType,
    T: ScalarType,
    yji: MatrixType,
  ) -> VectorType:
    """Computes the compressibility factor for each composition
    of `Np` phases.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yji : numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]
        mole fractions array of shape `(Np, Nc)`.

    Returns an array of the compressibility factor for each phase.
    """
    PRT = P / R / T
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Sji = sqrtalphai * (yji * sqrtalphai).dot(self.D)
    alphamj = np.sum(yji * Sji, axis=1)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRT / R / T
    Bj = bmj * PRT
    Zj = np.vectorize(self.solve_eos)(Aj, Bj) - yji.dot(self.vsi_bi) * PRT
    return Zj

  def get_lnphiji_Zj(
    self,
    P: ScalarType,
    T: ScalarType,
    yji: MatrixType,
  ) -> tuple[MatrixType, VectorType]:
    """Computes fugacity coefficients of components and
    the compressiblity factor for each composition of `Np` phases.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yji : numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]
        mole fractions array of shape `(Np, Nc)`.

    Returns a tuple with an array of the natural logarithm of fugacity
    coefficients for each component in each phase, and an array of
    compressibility factors for each phase.
    """
    PRT = P / R / T
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Sji = sqrtalphai * (yji * sqrtalphai).dot(self.D)
    alphamj = np.sum(yji * Sji, axis=1)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRT / R / T
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

  def get_lnfi(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> VectorType:
    """Computes fugacities of components.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns an array with the natural logarithm of the fugacity
    of each component.
    """
    return self.get_lnphii(P, T, yi,) + np.log(P * yi)

  def get_kvguess(
    self,
    P: ScalarType,
    T: ScalarType,
    # level: int,
  ) -> MatrixType:
    """Computes initial guess of k-values for given pressure `P` and
    temperature `T` using the Wilson's correlation.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

    Returns a matrix containing two vectors of initial k-values
    calculated using the Wilson's correlation and inverse Wilson's
    correlation.
    """
    kvi = self.Pci * np.exp(5.3727 * (1. + self.wi) * (1. - self.Tci / T)) / P
    return np.vstack([kvi, 1. / kvi])

  @staticmethod
  def solve_eos(
    A: np.float64,
    B: np.float64,
  ) -> np.float64:
    """Solves the modified Peng-Robinson equation of state.

    This method implements the Cardano's method to solve the cubic form
    of the modified Peng-Robinson equation of state, and selectes an
    appropriate root based on the comparison of the Gibbs energy
    difference.

    Arguments:
    ----------
      A : numpy.float64
        Coefficient in the cubic form of the modified Peng-Robinson
        equation of state.

      B : numpy.float64
        Coefficient in the cubic form of the modified Peng-Robinson
        equation of state.

    Returns the solution of the modified Peng-Robinson equation of state
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
      fdG = lambda Z1, Z2: (np.log((Z2 - B) / (Z1 - B)) + (Z1 - Z2)
        + 0.35355339 * A / B * np.log((Z1 - B * 0.41421356)
          * (Z2 + B * 2.41421356) / (Z1 + B * 2.41421356)
          / (Z2 - B * 0.41421356)))
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



