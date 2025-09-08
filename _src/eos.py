import logging

from dataclasses import (
  dataclass,
)

import numpy as np

from constants import (
  R,
)

from typing import (
  Optional,
)

from custom_types import (
  Scalar,
  Vector,
  Matrix,
  Tensor,
  IVector,
)


logger = logging.getLogger('eos')


class pr78(object):
  """Peng-Robinson (1978) equation of state.

  Computes fugacities of components, the compressibility factor of the
  mixture and their partial derivatives using the modified Peng-Robinson
  equation of state.

  Parameters
  ----------
  Pci: Vector, shape (Nc,)
    Critical pressures of `Nc` components [Pa].

  Tci: Vector, shape (Nc,)
    Critical temperatures of `Nc` components [K].

  wi: Vector, shape (Nc,)
    Acentric factors of `Nc` components.

  mwi: Vector, shape (Nc,)
    Molar weights of `Nc` components [kg/mol].

  vsi: Vector, shape (Nc,)
    Volume shift parameters of `Nc` components

  dij: Vector, shape (Nc * (Nc - 1) // 2,)
    The lower triangle matrix of binary interaction coefficients
    of `Nc` components.

  kvlevel: int
    Regulates an output of the function `getPT_kvguess` that generates
    initial k-values. Available options are:

    - 0: Wilson's and inverse Wilson's equations;
    - 1: the pure component model for heaviest and lightest components
         + Wilson's and inverse Wilson's equations;
    - 2: the pure component model for heaviest and lightest components
         + the ideal model + Wilson's and inverse Wilson's equations;
    - 3: the pure component model for heaviest, next heaviest, lightest
         and next lightest components + the ideal model + Wilson's and
         inverse Wilson's equations;
    - 4: the pure component model for each component + the ideal model
         + Wilson's and inverse Wilson's equations + cube roots of
         Wilson's and inverse Wilson's equations.

    Default is `0`.

  purefrc: Scalar
    This parameter governs the pure component model for initial
    k-values. It is the summarized mole fraction of other components
    in the trial phase except the specific component. Must be greater
    than zero and lower than one. Default is `1e-8`.
  """
  def __init__(
    self,
    Pci: Vector,
    Tci: Vector,
    wi: Vector,
    mwi: Vector,
    vsi: Vector,
    dij: Vector,
    kvlevel: int = 0,
    purefrc: Scalar = 1e-8,
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
    self.m = np.array([2, 1, 0])
    self.kvlevel = kvlevel
    idxs = np.argsort(self.mwi)
    h1 = idxs[-1]
    h2 = idxs[-2]
    l1 = idxs[0]
    l2 = idxs[1]
    upi = np.full(shape=(self.Nc,), fill_value=purefrc)
    yp = 1. - purefrc * (self.Nc - 1)
    self.h1i = upi.copy()
    self.h1i[h1] = yp
    self.l1i = upi.copy()
    self.l1i[l1] = yp
    self.h2i = upi.copy()
    self.h2i[h2] = yp
    self.l2i = upi.copy()
    self.l2i[l2] = yp
    self.upji = np.full(shape=(self.Nc, self.Nc), fill_value=upi)
    np.fill_diagonal(self.upji, yp)
    pass

  def getPT_Z(self, P: Scalar, T: Scalar, yi: Vector) -> tuple[int, Scalar]:
    """Computes the compressibility factor of the mixture.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple of:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture.
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
    s, Z = self.solve(A, B)
    return s, Z - yi.dot(self.vsi_bi) * PRT

  def getPT_lnphii(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Vector]:
    """Computes fugacity coefficients of components.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple of:
    - the designated phase (`0` = vapour, `1` = liquid),
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`.
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
    s, Z = self.solve(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * 0.414213562373095) / (Z + B * 2.414213562373095))
    return s, -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi

  def getPT_lnfi(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Vector]:
    """Computes fugacities of components.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple of:
    - the designated phase (`0` = vapour, `1` = liquid),
    - logarithms of fugacities of components as a `Vector` of shape
      `(Nc,)`.
    """
    s, lnphii = self.getPT_lnphii(P, T, yi)
    return s, lnphii + np.log(P * yi)

  def getPT_Z_lnphii(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Scalar, Vector]:
    """Computes fugacity coefficients of components and
    the compressibility factor of the mixture.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`.
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
    s, Z = self.solve(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    fZ = np.log((Z - B * 0.414213562373095) / (Z + B * 2.414213562373095))
    return (
      s,
      Z - PRT * yi.dot(self.vsi_bi),
      -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi,
    )

  def getPT_Z_lnfi(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Scalar, Vector]:
    """Computes fugacities of components and the compressibility factor
    of the mixture.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacities of components as a `Vector` of
      shape `(Nc,)`.
    """
    s, Z, lnphii = self.getPT_Z_lnphii(P, T, yi)
    return s, Z, lnphii + np.log(P * yi)

  def getPT_Z_lnphii_dP(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Scalar, Vector, Vector]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to pressure.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector` of shape `(Nc,)`.
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
    s, Z = self.solve(A, B)
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
    dqdP = np.power(Z, self.m).dot(ddmdP)
    dZdP = -dqdP / dqdZ
    dgZdP = dgZdZ * dZdP + dgZdB * dBdP
    dfZdP = dfZdZ * dZdP + dfZdB * dBdP
    dlnphiidP = (-dgZdP + dZdP / bm * self.bi + gphii * dfZdP
                 - self.vsi_bi / RT)
    return s, Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidP

  def getPT_Z_lnphii_dT(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Scalar, Vector, Vector]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to temperature.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector` of shape `(Nc,)`.
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
    s, Z = self.solve(A, B)
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
    dqdT = np.power(Z, self.m).dot(ddmdT)
    dZdT = -dqdT / dqdZ
    dgZdT = dgZdZ * dZdT + dgZdB * dBdT
    dfZdT = dfZdZ * dZdT + dfZdB * dBdT
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - dgZdT + dZdT / bm * self.bi + PRT / T * self.vsi_bi)
    return s, Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidT

  def getPT_Z_lnphii_dnj(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar = 1.,
  ) -> tuple[int, Scalar, Vector, Matrix]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to mole numbers of components.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    n: Scalar
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple that contains:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to mole numbers of components as a `Vector` of
      shape `(Nc, Nc)`.
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
    s, Z = self.solve(A, B)
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
    dAdnj = PRT / RT * dalphamdnj
    dBdnj = dbmdnj * PRT
    ddmdnj = ddmdA * dAdnj[:,None] + ddmdB * dBdnj[:,None]
    dqdnj = ddmdnj.dot(np.power(Z, self.m))
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
    return s, Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidnj

  def getPT_Z_lnphii_dP_dT(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Scalar, Vector, Vector, Vector]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to pressure and temperature.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple of:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector` of shape `(Nc,)`.
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
    s, Z = self.solve(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi
    Zpm = np.power(Z, self.m)
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
    dqdP = Zpm.dot(ddmdP)
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
    dqdT = Zpm.dot(ddmdT)
    dZdT = -dqdT / dqdZ
    dgZdT = dgZdZ * dZdT + dgZdB * dBdT
    dfZdT = dfZdZ * dZdT + dfZdB * dBdT
    dgphiidT = ((2. * dSidT - dalphamdT / bm * self.bi)
                / (2.82842712474619 * RT * bm) - gphii / T)
    dlnphiidT = (dfZdT * gphii + fZ * dgphiidT - dgZdT + dZdT / bm * self.bi
                 + PRT / T * self.vsi_bi)
    return s, Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidP, dlnphiidT

  def getPT_Z_lnphii_dP_dnj(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar = 1.,
  ) -> tuple[int, Scalar, Vector, Vector, Matrix]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to mole numbers of components and pressure.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    n: Scalar
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple of:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to mole numbers of components as a `Matrix` of
      shape `(Nc, Nc)`.
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
    s, Z = self.solve(A, B)
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
    dAdnj = PRT / RT * dalphamdnj
    dBdnj = dbmdnj * PRT
    ddmdnj = ddmdA * dAdnj[:,None] + ddmdB * dBdnj[:,None]
    Zpm = np.power(Z, self.m)
    dqdnj = ddmdnj.dot(Zpm)
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
    dqdP = Zpm.dot(ddmdP)
    dZdP = -dqdP / dqdZ
    dgZdP = dgZdZ * dZdP + dgZdB * dBdP
    dfZdP = dfZdZ * dZdP + dfZdB * dBdP
    dlnphiidP = (-dgZdP + dZdP / bm * self.bi + gphii * dfZdP
                 - self.vsi_bi / RT)
    return s, Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidP, dlnphiidnj

  def getPT_Z_lnphii_dT_dnj(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar = 1.,
  ) -> tuple[int, Scalar, Vector, Vector, Matrix]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to mole numbers of components and
    temperature.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    n: Scalar
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple that contains:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to mole numbers of components as a `Matrix` of
      shape `(Nc, Nc)`.
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
    s, Z = self.solve(A, B)
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
    dAdnj = PRT / RT * dalphamdnj
    dBdnj = dbmdnj * PRT
    ddmdnj = ddmdA * dAdnj[:,None] + ddmdB * dBdnj[:,None]
    Zpm = np.power(Z, self.m)
    dqdnj = ddmdnj.dot(Zpm)
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
    dqdT = Zpm.dot(ddmdT)
    dZdT = -dqdT / dqdZ
    dgZdT = dgZdZ * dZdT + dgZdB * dBdT
    dfZdT = dfZdZ * dZdT + dfZdB * dBdT
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - dgZdT + dZdT / bm * self.bi + PRT / T * self.vsi_bi)
    return s, Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidT, dlnphiidnj

  def getPT_Z_lnphii_dP_dT_dyj(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Scalar, Vector, Vector, Vector, Matrix]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to pressure, temperature and mole fractions
    of components pressure.

    The mole fraction constraint isn't taken into account.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple of:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to mole fractions of components as a `Matrix` of
      shape `(Nc, Nc)`.
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
    s, Z = self.solve(A, B)
    gZ = np.log(Z - B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = Z - B * 0.414213562373095
    ZpB = Z + B * 2.414213562373095
    fZ = np.log(ZmB / ZpB)
    lnphii = -gZ + (Z - 1.) / bm * self.bi + fZ * gphii - PRT * self.vsi_bi
    Zpm = np.power(Z, np.array([2, 1, 0]))
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
    dAdyj = PRT / RT * dalphamdyj
    dBdyj = dbmdyj * PRT
    ddmdyj = ddmdA[:,None] * dAdyj + ddmdB[:,None] * dBdyj
    dqdyj = Zpm.dot(ddmdyj)
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
    dqdP = Zpm.dot(ddmdP)
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
    dqdT = Zpm.dot(ddmdT)
    dZdT = -dqdT / dqdZ
    dgZdT = dgZdZ * dZdT + dgZdB * dBdT
    dfZdT = dfZdZ * dZdT + dfZdB * dBdT
    dgphiidT = ((2. * dSidT - dalphamdT / bm * self.bi)
                / (2.82842712474619* RT * bm) - gphii / T)
    dlnphiidT = (dfZdT * gphii + fZ * dgphiidT - dgZdT + dZdT / bm * self.bi
                 + PRT / T * self.vsi_bi)
    return (s, Z - PRT * yi.dot(self.vsi_bi), lnphii,
            dlnphiidP, dlnphiidT, dlnphiidyj)

  def getPT_Z_lnphii_dP_d2P(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Scalar, Vector, Vector, Vector]:
    """Computes fugacity coefficients of components and their first
    and second partial derivatives with respect to pressure.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector` of shape `(Nc,)`,
    - second partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector` of shape `(Nc,)`.
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
    s, Z = self.solve(A, B)
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
    Zpm = np.power(Z, self.m)
    dqdP = Zpm.dot(ddmdP)
    dZdP = -dqdP / dqdZ
    dgZdP = dgZdZ * dZdP + dgZdB * dBdP
    dfZdP = dfZdZ * dZdP + dfZdB * dBdP
    dlnphiidP = (-dgZdP + dZdP / bm * self.bi + gphii * dfZdP
                 - self.vsi_bi / RT)
    d2dmdB2 = np.array([0., -6., 2. + 6. * B])
    d2dmdAdB = np.array([0., 0., -1.])
    d2dmdP2 = dBdP * dBdP * d2dmdB2 + 2. * dAdP * dBdP * d2dmdAdB
    d2qdZ2 = 6. * Z + 2. * (B - 1.)
    d2qdP2 = Zpm.dot(d2dmdP2)
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
    return s, Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidP, d2lnphiidP2

  def getPT_Z_lnphii_dT_d2T(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[int, Scalar, Vector, Vector, Vector]:
    """Computes fugacity coefficients of components and their first
    and second partial derivatives with respect to temperature.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a `Vector`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector` of shape `(Nc,)`,
    - second partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector` of shape `(Nc,)`.
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
    s, Z = self.solve(A, B)
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
    Zpm = np.power(Z, self.m)
    dqdT = Zpm.dot(ddmdT)
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
    d2dmdT2 = (ddmdA * d2AdT2 + dBdT * dBdT * d2dmdB2 + ddmdB * d2BdT2
               + 2. * dAdT * dBdT * d2dmdAdB)
    d2qdT2 = Zpm.dot(d2dmdT2)
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
    return s, Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidT, d2lnphiidT2

  def getPT_Z_lnfi_dnj(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar = 1.,
  ) -> tuple[int, Scalar, Vector, Matrix]:
    """Computes fugacities of components and their partial derivatives
    with respect to component mole numbers.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    n: Scalar
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple of:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the compressibility factor of the mixture,
    - logarithms of fugacities of components as a `Vector` of shape
      `(Nc,)`,
    - partial derivatives of logarithms of fugacities with respect
      to mole numbers of components as a `Matrix` of shape `(Nc, Nc)`.
    """
    s, Z, lnphii, dlnphiidnj = self.getPT_Z_lnphii_dnj(P, T, yi, n)
    lnfi = lnphii + np.log(yi * P)
    dlnfidnj = dlnphiidnj + np.diagflat(1. / (n * yi)) - 1. / n
    return s, Z, lnfi, dlnfidnj

  def getPT_Zj(
    self,
    Pj: Scalar | Vector,
    Tj: Scalar | Vector,
    yji: Vector | Matrix,
  ) -> tuple[IVector, Vector]:
    """Computes compressibility factors for each mixture.

    Parameters
    ----------
    Pj: Scalar | Vector, shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: Scalar | Vector, shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector, shape (Nc,) | Matrix, shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    Returns
    -------
    A tuple of:
    - a 1d-array of integers with the shape `(Np,)`, each of which
      corresponds to the designated phase (`0` = vapour, `1` = liquid),
    - compressibility factors of mixtures as a `Vector` of shape
      `(Np,)`.
    """
    Pj = np.atleast_1d(Pj)
    Tj = np.atleast_1d(Tj)
    yji = np.atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    multji = 1. + self.kappai * (1. - np.sqrt(Tj)[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji = sqrtalphaji * (yji * sqrtalphaji).dot(self.D)
    alphamj = np.vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    sj, Zj = np.vectorize(self.solve)(Aj, Bj)
    return sj, Zj - yji.dot(self.vsi_bi) * PRTj

  def getPT_Zj_lnphiji(
    self,
    Pj: Scalar | Vector,
    Tj: Scalar | Vector,
    yji: Vector | Matrix,
  ) -> tuple[IVector, Vector, Matrix]:
    """Computes fugacity coefficients of components and compressibility
    factors for each mixture.

    Parameters
    ----------
    Pj: Scalar | Vector, shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: Scalar | Vector, shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector, shape (Nc,) | Matrix, shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    Returns
    -------
    A tuple of:
    - a 1d-array of integers with the shape `(Np,)`, each of which
      corresponds to the designated phase (`0` = vapour, `1` = liquid),
    - compressibility factors of mixtures as a `Vector` of shape
      `(Np,)`,
    - logarithms of fugacity coefficients of components in mixtures as
      a `Matrix` of shape `(Np, Nc)`.
    """
    Pj = np.atleast_1d(Pj)
    Tj = np.atleast_1d(Tj)
    yji = np.atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    multji = 1. + self.kappai * (1. - np.sqrt(Tj)[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji = sqrtalphaji * (yji * sqrtalphaji).dot(self.D)
    alphamj = np.vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    sj, Zj = np.vectorize(self.solve)(Aj, Bj)
    gphiji = ((0.3535533905932738 * Aj / Bj)[:,None]
              * (2. / alphamj[:,None] * Sji - self.bi / bmj[:,None]))
    fZj = np.log((Zj - Bj * 0.414213562373095)
                 / (Zj + Bj * 2.414213562373095))
    lnphiji = (self.bi * ((Zj - 1.) / bmj)[:,None]
               + gphiji * fZj[:,None]
               - np.log(Zj - Bj)[:,None]
               - PRTj[:,None] * self.vsi_bi)
    return sj, Zj - yji.dot(self.vsi_bi) * PRTj, lnphiji

  def getPT_Zj_lnphiji_dnk(
    self,
    Pj: Scalar | Vector,
    Tj: Scalar | Vector,
    yji: Vector | Matrix,
    nj: Scalar | Matrix = 1.,
  ) -> tuple[IVector, Vector, Matrix, Tensor]:
    """Computes fugacity coefficients of components, compressibility
    factors and partial derivatives of fugacity coefficients with
    respect to component mole numbers for each mixture.

    Parameters
    ----------
    Pj: Scalar | Vector, shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: Scalar | Vector, shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector, shape (Nc,) | Matrix, shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    nj: Scalar | Vector, shape (Np,)
      Mole number(s) of mixtures [mol]. It is allowed to specify
      different mole number for each mixture. In that case, `Np` is
      the number of mixtures. Default is `1.0` [mol].

    Returns
    -------
    A tuple of:
    - a 1d-array of integers with the shape `(Np,)`, each of which
      corresponds to the designated phase (`0` = vapour, `1` = liquid),
    - compressibility factors of mixtures as a `Vector` of shape
      `(Np,)`,
    - logarithms of fugacity coefficients of components in mixtures as
      a `Matrix` of shape `(Np, Nc)`,
    - partial derivatives of logarithms of fugacity coefficients with
      respect to component mole numbers for each mixture as a `Tensor`
      of shape `(Np, Nc, Nc)`.
    """
    Pj = np.atleast_1d(Pj)
    Tj = np.atleast_1d(Tj)
    yji = np.atleast_2d(yji)
    nj = np.atleast_1d(nj)
    RTj = R * Tj
    PRTj = Pj / RTj
    multji = 1. + self.kappai * (1. - np.sqrt(Tj)[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji = sqrtalphaji * (yji * sqrtalphaji).dot(self.D)
    alphamj = np.vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    sj, Zj = np.vectorize(self.solve)(Aj, Bj)
    gphiji = ((0.3535533905932738 * Aj / Bj)[:,None]
              * (2. / alphamj[:,None] * Sji - self.bi / bmj[:,None]))
    ZmBj = Zj - Bj * 0.414213562373095
    ZpBj = Zj + Bj * 2.414213562373095
    fZj = np.log(ZmBj / ZpBj)
    lnphiji = (self.bi * ((Zj - 1.) / bmj)[:,None]
               + gphiji * fZj[:,None]
               - np.log(Zj - Bj)[:,None]
               - PRTj[:,None] * self.vsi_bi)
    ddmdAj = np.zeros(shape=(Bj.shape[0], 3))
    ddmdAj[:,1] = 1.
    ddmdAj[:,2] = -Bj
    ddmdBj = np.ones(shape=(Bj.shape[0], 3))
    ddmdBj[:,1] = -2. - 6. * Bj
    ddmdBj[:,2] = Bj * (2. + 3. * Bj) - Aj
    dqdZj = 3. * Zj * Zj + 2. * (Bj - 1.) * Zj + (Aj - 2. * Bj - 3. * Bj * Bj)
    dgZdZj = 1. / (Zj - Bj)
    dgZdBj = -dgZdZj
    dfZdZj = 1. / ZmBj - 1. / ZpBj
    dfZdBj = -0.414213562373095 / ZmBj - 2.414213562373095 / ZpBj
    dSjidnk = (sqrtalphaji[:,:,None] * sqrtalphaji[:,None,:] * self.D
               - Sji[:,:,None]) / nj[:,None,None]
    dalphamjdnk = 2. / nj[:,None] * (Sji - alphamj[:,None])
    dbmjdnk = (self.bi - bmj[:,None]) / nj[:,None]
    dAjdnk = dalphamjdnk * (PRTj / RTj)[:,None]
    dBjdnk = dbmjdnk * PRTj[:,None]
    ddmdnk = (ddmdAj[:,None,:] * dAjdnk[:,:,None]
              + ddmdBj[:,None,:] * dBjdnk[:,:,None])
    dqjdnk = np.vecdot(ddmdnk, np.power(Zj[:,None], self.m)[:,None,:])
    dZjdnk = -dqjdnk / dqdZj[:,None]
    dgZjdnk = dgZdZj[:,None] * dZjdnk + dgZdBj[:,None] * dBjdnk
    dfZjdnk = dfZdZj[:,None] * dZjdnk + dfZdBj[:,None] * dBjdnk
    dgphijidnk = (
      ((2. / alphamj)[:,None,None]
       * (dSjidnk - (Sji / alphamj[:,None])[:,:,None] * dalphamjdnk[:,None,:])
       + (self.bi / (bmj * bmj)[:,None])[:,:,None] * dbmjdnk[:,None,:])
      * (0.3535533905932738 * Aj / Bj)[:,None,None]
      + ((dAjdnk / Aj[:,None] - dBjdnk / Bj[:,None])[:,None,:]
         * gphiji[:,:,None])
    )
    dlnphiidnj = (
      ((self.bi / bmj[:,None])[:,:,None]
       * (dZjdnk - ((Zj - 1.) / bmj)[:,None] * dbmjdnk)[:,None,:])
      + (fZj[:,None,None] * dgphijidnk + gphiji[:,:,None] * dfZjdnk[:,None,:])
      - dgZjdnk[:,None,:])
    return sj, Zj - yji.dot(self.vsi_bi) * PRTj, lnphiji, dlnphiidnj

  def getPT_kvguess(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, ...]:
    """Computes initial k-values for given pressure, temperature
    and composition.

    Parameters
    ----------
    P: Scalar
      Pressure of the mixture [Pa].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple containing arrays of initial k-values.
    """
    kvi = self.Pci * np.exp(5.3727 * (1. + self.wi) * (1. - self.Tci / T)) / P
    if self.kvlevel == 0:
      return kvi, 1. / kvi
    elif self.kvlevel == 1:
      return self.h1i, self.l1i, kvi, 1. / kvi
    elif self.kvlevel == 2:
      uii = np.exp(self.getPT_lnphii(P, T, yi)[1])
      return self.h1i, self.l1i, uii, kvi, 1. / kvi
    elif self.kvlevel == 3:
      uii = np.exp(self.getPT_lnphii(P, T, yi)[1])
      return self.h1i, self.l1i, self.h2i, self.l2i, uii, kvi, 1. / kvi
    elif self.kvlevel == 4:
      uii = np.exp(self.getPT_lnphii(P, T, yi)[1])
      cbrtkvi = np.cbrt(kvi)
      return *self.upji, uii, kvi, 1. / kvi, cbrtkvi, 1. / cbrtkvi
    else:
      raise ValueError(f'Unsupported level number: {level}.')

  def getVT_P(
    self,
    V: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar = 1.,
  ) -> Scalar:
    """Computes pressure for given volume, temperature and composition.

    Parameters
    ----------
    V: Scalar
      Volume of the mixture [m3].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    n: Scalar
      Mole number of the mixture [mol]. Default is `1.0` [mol].

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
    V: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar = 1.,
  ) -> tuple[Vector, Matrix]:
    """Computes fugacities of components and their partial derivatives
    with respect to component mole numbers.

    Partial derivatives formulas were taken from the paper of M.L.
    Michelsen and R.A. Heidemann, 1981 (doi: 10.1002/aic.690270326).

    Parameters
    ----------
    V: Scalar
      Volume of the mixture [m3].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    n: Scalar
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple that contains:
    - logarithms of fugacities of components as a `Vector` of shape
      `(Nc,)`,
    - partial derivatives of logarithms of fugacities with respect
      to mole numbers of components as a `Matrix` of shape `(Nc, Nc)`.
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
    V: Scalar,
    T: Scalar,
    yi: Vector,
    zti: Vector,
    n: Scalar = 1.,
  ) -> Scalar:
    """Computes the cubic form of the Helmholtz energy Taylor series
    decomposition for a given vector of component mole number changes.
    This method is used by the critical point calculation procedure.

    Calculation formulas were taken from the paper of M.L. Michelsen and
    R.A. Heidemann, 1981 (doi: 10.1002/aic.690270326).

    Parameters
    ----------
    V: Scalar
      Volume of the mixture [m3].

    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    zti: Vector, shape (Nc,)
      Component mole number changes [mol].

    n: Scalar
      Mole number of the mixture [mol]. Default is `1.0` [mol].

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

  def getVT_vmin(self, T: Scalar, yi: Vector) -> Scalar:
    """Calculates the minimal molar volume.

    Parameters
    ----------
    T: Scalar
      Temperature of the mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The minimum molar volume in [m3/mol].
    """
    return yi.dot(self.bi)

  @staticmethod
  def fdG(Z1: Scalar, Z2: Scalar, A: Scalar, B: Scalar) -> Scalar:
    """Calculates the Gibbs energy difference between two states
    corresponding to the roots of the equation of state.

    Parameters
    ----------
    Z1: Scalar
      The first root (compressibility factor) of the equation of state.

    Z2: Scalar
      The second root (compressibility factor) of the equation of state.

    A: Scalar
      The coefficient of the cubic form of the equation of state.

    B: Scalar
      The coefficient of the cubic form of the equation of state.

    Returns
    -------
    The Gibbs energy difference between two states corresponding to the
    roots of the equation of state.
    """
    return (np.log((Z2 - B) / (Z1 - B))
            + (Z1 - Z2)
            + np.log((Z1 - B * 0.414213562373095)
                     * (Z2 + B * 2.414213562373095)
                     / ((Z1 + B * 2.414213562373095)
                        * (Z2 - B * 0.414213562373095)))
              * 0.3535533905932738 * A / B)

  def solve(self, A: Scalar, B: Scalar) -> tuple[int, Scalar]:
    """Solves the modified Peng-Robinson equation of state.

    This method implements the Cardano's method to solve the cubic form
    of the equation of state.

    When the equation of state has two (or more) real roots, the correct
    solution will be chosen based on the comparison of Gibbs energies
    corresponding to these roots. If that procedure selects the largest
    real root, then the fluid is said to be vapour. Similarly, if the
    smallest real root is chosen, the fluid is said to be liquid.

    When the equation of state has only one real root, the fluid is
    designated as liquid or vapour based on the algorithm presented
    by R. Gosset et al, 1986 (doi: 10.1016/0378-3812(86)85061-0).

    Parameters:
    -----------
    A: Scalar
      The coefficient in the cubic form of the modified Peng-Robinson
      equation of state.

    B: Scalar
      The coefficient in the cubic form of the modified Peng-Robinson
      equation of state.

    Returns
    -------
    A tuple of:
    - the designated phase (`0` = vapour, `1` = liquid),
    - the solution of the modified Peng-Robinson equation of state
      corresponding to the lowest Gibbs energy.
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
      x = u1 + u2 - b / 3.
      if A > 5.877359948604401 * B and x < 3.9513730355914394 * B:
        return 1, x
      else:
        return 0, x
    else:
      t0 = (2. * np.sqrt(-p / 3.)
            * np.cos(np.arccos(1.5 * q * np.sqrt(-3. / p) / p) / 3.))
      x0 = t0 - b / 3.
      r = b + x0
      k = -d / x0
      D = np.sqrt(r * r - 4. * k)
      x1 = (-r + D) * .5
      x2 = (-r - D) * .5
      if x2 > B:
        dG = self.fdG(x0, x2, A, B)
        if dG < 0.:
          return 0, x0
        else:
          return 1, x2
      elif x1 > B:
        dG = self.fdG(x0, x1, A, B)
        if dG < 0.:
          return 0, x0
        else:
          return 1, x1
      else:
        return 0, x0
