import logging

import numpy as np

from constants import (
  R,
)

from typing import (
  Protocol,
)

from customtypes import (
  Integer,
  Double,
  Vector,
  Matrix,
  Tensor,
)


logger = logging.getLogger('eos')


class Eos(Protocol):
  name: str
  Nc: int
  mwi: Vector[Double]


class pr78(object):
  """Peng-Robinson (1978) equation of state.

  This class can be used to compute fugacities of components, the
  compressibility factor of a mixture and their partial derivatives
  with respect to pressure, temperature and composition using the
  modified Peng-Robinson equation of state. Classical (van der Waals)
  mixing rules are used to calculate attraction and repulsion parameters
  for mixtures.

  Parameters
  ----------
  Pci: Vector[Double], shape (Nc,)
    Critical pressures of `Nc` components [Pa].

  Tci: Vector[Double], shape (Nc,)
    Critical temperatures of `Nc` components [K].

  wi: Vector[Double], shape (Nc,)
    Acentric factors of `Nc` components.

  mwi: Vector[Double], shape (Nc,)
    Molar weights of `Nc` components [kg/mol].

  vsi: Vector[Double], shape (Nc,)
    Volume shift parameters of `Nc` components

  dij: Vector[Double], shape (Nc * (Nc - 1) // 2,)
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

  purefrc: float
    This parameter governs the pure component model for initial
    k-values. It is the summarized mole fraction of other components
    in the trial phase except the specific component. Must be greater
    than zero and lower than one. Default is `1e-8`.

  phaseid: str
    The phase designation method. Should be one of:

    - `'wilson'` (use Wilson's k-values),
    - `'vcTc'` (use pseudo-critical volume and temperature),
    - `'dadT'` (use the derivative with respect to temperature of
                the thermal expansion coefficient),
    - `'pip'` (use the complex parameter; for the details, see the
               paper of G. Venkatarathnam and L.R. Oellrich, 2011
               (doi: 10.1016/j.fluid.2010.12.001)).

    Default is `'wilson'`. For the details of available phase
    identification methods, see the paper of J. Bennett and K.A.G.
    Schmidt, 2016 (doi: 10.1021/acs.energyfuels.6b02316). The above
    mentioned methods are listed in the order of the complexity
    increase.

  vci: Vector[Double] | None, shape (Nc,)
    Critical molar volumes [m3/mol] of `Nc` components. Default is
    `None` which means that they will be calculated using the
    critical compressibility factor of the EOS (0.3074...).
  """
  def __init__(
    self,
    Pci: Vector[Double],
    Tci: Vector[Double],
    wi: Vector[Double],
    mwi: Vector[Double],
    vsi: Vector[Double],
    dij: Vector[Double],
    kvlevel: int = 0,
    purefrc: float = 1e-8,
    phaseid: str = 'wilson',
    vci: Vector[Double] | None = None,
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
    if phaseid == 'wilson':
      self._getPT_PID = self._getPT_PID_wilson
      self._getPT_PIDj = self._getPT_PIDj_wilson
    elif phaseid == 'vcTc':
      if vci is None:
        self.vci = 0.3074013086987038 * R * Tci / Pci
      else:
        self.vci = vci
      self._getPT_PID = self._getPT_PID_vcTc
      self._getPT_PIDj = self._getPT_PIDj_vcTc
    elif phaseid == 'dadT':
      self._getPT_PID = self._getPT_PID_dadT
      self._getPT_PIDj = self._getPT_PIDj_dadT
    elif phaseid == 'pip':
      self._getPT_PID = self._getPT_PID_pip
      self._getPT_PIDj = self._getPT_PIDj_pip
    else:
      raise ValueError(f'Unknown phase identification method: {phaseid}.')
    pass

  def getPT_Z(self, P: float, T: float, yi: Vector[Double]) -> float:
    """Computes the compressibility factor of the mixture.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The compressibility factor of the mixture.
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
    Z = self.solve(A, B)
    return Z - yi.dot(self.vsi_bi) * PRT

  def getPT_lnphii(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> Vector[Double]:
    """Computes fugacity coefficients of components.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    Logarithms of fugacity coefficients of components as a
    `Vector[Double]` of shape `(Nc,)`.
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
    Z = self.solve(A, B)
    fZ = np.log((Z - B * 0.414213562373095) / (Z + B * 2.414213562373095))
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    return ((Z - 1.) / bm * self.bi
            - np.log(Z - B)
            + fZ * gphii
            - PRT * self.vsi_bi)

  def getPT_lnfi(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> Vector[Double]:
    """Computes fugacities of components.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    Logarithms of fugacities of components as a `Vector[Double]` of
    shape `(Nc,)`.
    """
    return self.getPT_lnphii(P, T, yi) + np.log(P * yi)

  def getPT_Z_lnphii(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[float, Vector[Double]]:
    """Computes fugacity coefficients of components and
    the compressibility factor of the mixture.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]` of shape `(Nc,)`.
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
    Z = self.solve(A, B)
    fZ = np.log((Z - B * 0.414213562373095) / (Z + B * 2.414213562373095))
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    lnphii = ((Z - 1.) / bm * self.bi
              - np.log(Z - B)
              + fZ * gphii
              - PRT * self.vsi_bi)
    return Z - PRT * yi.dot(self.vsi_bi), lnphii

  def getPT_Z_lnfi(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[float, Vector[Double]]:
    """Computes fugacities of components and the compressibility factor
    of the mixture.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the compressibility factor of the mixture,
    - logarithms of fugacities of components as a `Vector[Double]` of
      shape `(Nc,)`.
    """
    Z, lnphii = self.getPT_Z_lnphii(P, T, yi)
    return Z, lnphii + np.log(P * yi)

  def getPT_Z_lnphii_dP(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[float, Vector[Double], Vector[Double]]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to pressure.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector[Double]` of shape `(Nc,)`.
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
    Z = self.solve(A, B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    lnphii = (-np.log(Z - B)
              + (Z - 1.) / bm * self.bi
              + np.log(ZpB / ZmB) * gphii
              - PRT * self.vsi_bi)
    dZdP = ((B * (2. * (A - B) - 3. * B * B)
             + Z * (6. * B * B + 2. * B - A)
             - B * Z * Z)
            / (P * (3. * Z * Z
                    + 2. * (B - 1.) * Z
                    + A - 2. * B - 3. * B * B)))
    dlnphiidP = ((B / P - dZdP) / (Z - B)
                 + dZdP / bm * self.bi
                 + gphii * (dZdP * (ZmB - ZpB)
                            - B / P * (0.414213562373095 * ZmB
                                       + 2.414213562373095 * ZpB))
                 - self.vsi_bi / RT)
    return Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidP

  def getPT_Z_lnphii_dT(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[float, Vector[Double], Vector[Double]]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to temperature.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector[Double]` of shape
      `(Nc,)`.
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
    Z = self.solve(A, B)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = np.log(ZpB / ZmB)
    lnphii = (0.3535533905932738 * fZ * gphii
              - np.log(Z - B)
              + (Z - 1.) / bm * self.bi
              - PRT * self.vsi_bi)
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * self.D.dot(yi * dsqrtalphaidT)
    dalphamdT = yi.dot(dSidT)
    dBdT = -B / T
    dZdT = ((dBdT * (A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z)
             + (PRT / RT * dalphamdT - 2. * A / T) * (B - Z))
            / (3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B))
    dfZdT = dZdT * (ZmB - ZpB) - dBdT * (0.414213562373095 * ZmB
                                         + 2.414213562373095 * ZpB)
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - (dZdT - dBdT) / (Z - B)
                 + dZdT / bm * self.bi
                 + PRT / T * self.vsi_bi)
    return Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidT

  def getPT_Z_lnphii_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
    n: float = 1.,
  ) -> tuple[float, Vector[Double], Matrix[Double]]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to mole numbers of components.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple that contains:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to mole numbers of components as a `Vector[Double]`
      of shape `(Nc, Nc)`.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = PRT / RT * alpham
    B = PRT * bm
    Z = self.solve(A, B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = np.log(ZpB / ZmB)
    lnphii = ((Z - 1.) / bm * self.bi
              - np.log(Z - B)
              + fZ * gphii
              - PRT * self.vsi_bi)
    dSidnj = (sqrtalphai[:,None] * sqrtalphai * self.D - Si[:,None]) / n
    dalphamdnj = 2. / n * (Si - alpham)
    dbmdnj = (self.bi - bm) / n
    dAdnj = PRT / RT * dalphamdnj
    dBdnj = PRT * dbmdnj
    dZdnj = ((dBdnj * (A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z)
              + dAdnj * (B - Z))
             / (3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B))
    dfZdnj = (dZdnj * (ZmB - ZpB) - dBdnj * (0.414213562373095 * ZmB
                                             + 2.414213562373095 * ZpB))
    dgphiidnj = ((2. / alpham * (dSidnj - (Si / alpham)[:,None] * dalphamdnj)
                  + (self.bi / (bm * bm))[:,None] * dbmdnj)
                 * (0.3535533905932738 * A / B)
                 + gphii[:,None] * (dAdnj / A - dBdnj / B))
    dlnphiidnj = ((self.bi / bm)[:,None] * (dZdnj - (Z - 1.) / bm * dbmdnj)
                  + (fZ * dgphiidnj + gphii[:,None] * dfZdnj)
                  - (dZdnj - dBdnj) / (Z - B))
    return Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidnj

  def getPT_Z_lnphii_dP_dT(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[float, Vector[Double], Vector[Double], Vector[Double]]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to pressure and temperature.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple of:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector[Double]` of shape
      `(Nc,)`.
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
    Z = self.solve(A, B)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = np.log(ZpB / ZmB)
    lnphii = ((Z - 1.) / bm * self.bi
              - np.log(Z - B)
              + 0.3535533905932738 * fZ * gphii
              - PRT * self.vsi_bi)
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B
    mdqdA = B - Z
    mdqdB = A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z
    dfZdZ = ZmB - ZpB
    dfZdB = -0.414213562373095 * ZmB - 2.414213562373095 * ZpB
    dZdP = (B * mdqdB + A * mdqdA) / (P * dqdZ)
    dlnphiidP = ((B / P - dZdP) / (Z - B)
                 + dZdP / bm * self.bi
                 + 0.3535533905932738 * (dZdP * dfZdZ + B / P * dfZdB) * gphii
                 - self.vsi_bi / RT)
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * self.D.dot(yi * dsqrtalphaidT)
    dalphamdT = yi.dot(dSidT)
    dBdT = -B / T
    dZdT = (dBdT * mdqdB + (PRT / RT * dalphamdT - 2. * A / T) * mdqdA) / dqdZ
    dfZdT = dZdT * dfZdZ + dBdT * dfZdB
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - (dZdT - dBdT) / (Z - B)
                 + dZdT / bm * self.bi
                 + PRT / T * self.vsi_bi)
    return Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidP, dlnphiidT

  def getPT_Z_lnphii_dP_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
    n: float = 1.,
  ) -> tuple[float, Vector[Double], Vector[Double], Matrix[Double]]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to mole numbers of components and pressure.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple of:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]`
      of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to mole numbers of components as a `Matrix[Double]`
      of shape `(Nc, Nc)`.
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
    Z = self.solve(A, B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = np.log(ZpB / ZmB)
    lnphii = ((Z - 1.) / bm * self.bi
              - np.log(Z - B)
              + fZ * gphii
              - PRT * self.vsi_bi)
    mdqdA = B - Z
    mdqdB = A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dfZdZ = ZmB - ZpB
    dfZdB = -0.414213562373095 * ZmB - 2.414213562373095 * ZpB
    dSidnj = (sqrtalphai[:,None] * sqrtalphai * self.D - Si[:,None]) / n
    dalphamdnj = 2. / n * (Si - alpham)
    dbmdnj = (self.bi - bm) / n
    dAdnj = PRT / RT * dalphamdnj
    dBdnj = dbmdnj * PRT
    dZdnj = (dBdnj * mdqdB + dAdnj * mdqdA) / dqdZ
    dgphiidnj = ((2. / alpham * (dSidnj - (Si / alpham)[:,None] * dalphamdnj)
                  + (self.bi / (bm * bm))[:,None] * dbmdnj)
                 * (0.3535533905932738 * A / B)
                 + gphii[:,None] * (dAdnj / A - dBdnj / B))
    dlnphiidnj = ((self.bi / bm)[:,None] * (dZdnj - (Z - 1.) / bm * dbmdnj)
                  + (fZ * dgphiidnj + gphii[:,None] * (dfZdZ * dZdnj
                                                       + dfZdB * dBdnj))
                  - (dZdnj - dBdnj) / (Z - B))
    dZdP = (B * mdqdB + A * mdqdA) / (P * dqdZ)
    dlnphiidP = ((B / P - dZdP) / (Z - B)
                 + dZdP / bm * self.bi
                 + gphii * (dZdP * dfZdZ + B / P * dfZdB)
                 - self.vsi_bi / RT)
    return Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidP, dlnphiidnj

  def getPT_Z_lnphii_dT_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
    n: float = 1.,
  ) -> tuple[float, Vector[Double], Vector[Double], Matrix[Double]]:
    """Computes fugacity coefficients of components and their partial
    derivatives with respect to mole numbers of components and
    temperature.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple that contains:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector[Double]` of shape
      `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to mole numbers of components as a `Matrix[Double]`
      of shape `(Nc, Nc)`.
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
    Z = self.solve(A, B)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = np.log(ZpB / ZmB)
    lnphii = (0.3535533905932738 * fZ * gphii
              - np.log(Z - B)
              + (Z - 1.) / bm * self.bi
              - PRT * self.vsi_bi)
    mdqdA = B - Z
    mdqdB = A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dfZdZ = ZmB - ZpB
    dfZdB = -0.414213562373095 * ZmB - 2.414213562373095 * ZpB
    dSidnj = (sqrtalphai[:,None] * sqrtalphai * self.D - Si[:,None]) / n
    dalphamdnj = 2. / n * (Si - alpham)
    dbmdnj = (self.bi - bm) / n
    dAdnj = PRT / RT * dalphamdnj
    dBdnj = dbmdnj * PRT
    dZdnj = (dBdnj * mdqdB + dAdnj * mdqdA) / dqdZ
    dfZdnj = dfZdZ * dZdnj + dfZdB * dBdnj
    dgphiidnj = ((2. / alpham * (dSidnj - (Si / alpham)[:,None] * dalphamdnj)
                  + (self.bi / (bm * bm))[:,None] * dbmdnj) * (A / B)
                 + gphii[:,None] * (dAdnj / A - dBdnj / B))
    dlnphiidnj = ((self.bi / bm)[:,None] * (dZdnj - (Z - 1.) / bm * dbmdnj)
                  + (0.3535533905932738 * fZ * dgphiidnj
                     + (0.3535533905932738 * gphii)[:,None] * dfZdnj)
                  - (dZdnj - dBdnj) / (Z - B))
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * self.D.dot(yi * dsqrtalphaidT)
    dalphamdT = yi.dot(dSidT)
    dBdT = -bm * PRT / T
    dZdT = (dBdT * mdqdB + (PRT / RT * dalphamdT - 2. * A / T) * mdqdA) / dqdZ
    dfZdT = dfZdZ * dZdT + dfZdB * dBdT
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - (dZdT - dBdT) / (Z - B)
                 + dZdT / bm * self.bi
                 + PRT / T * self.vsi_bi)
    return Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidT, dlnphiidnj

  def getPT_Z_lnphii_dP_dT_dyj(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[float, Vector[Double], Vector[Double],
             Vector[Double], Matrix[Double]]:
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

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple of:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector[Double]` of shape
      `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to mole fractions of components as a `Matrix[Double]`
      of shape `(Nc, Nc)`.
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
    Z = self.solve(A, B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = np.log(ZpB / ZmB)
    lnphii = ((Z - 1.) / bm * self.bi
              - np.log(Z - B)
              + fZ * gphii
              - PRT * self.vsi_bi)
    mdqdA = B - Z
    mdqdB = A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + (A - 2. * B - 3. * B * B)
    dfZdZ = ZmB - ZpB
    dfZdB = -0.414213562373095 * ZmB - 2.414213562373095 * ZpB
    dSidyj = sqrtalphai[:,None] * sqrtalphai * self.D
    dalphamdyj = Si + yi.dot(dSidyj)
    dbmdyj = self.bi
    dAdyj = PRT / RT * dalphamdyj
    dBdyj = PRT * dbmdyj
    dZdyj = (dBdyj * mdqdB + dAdyj * mdqdA) / dqdZ
    dfZdyj = dfZdZ * dZdyj + dfZdB * dBdyj
    dgphiidyj = ((2. / alpham * (dSidyj - (Si / alpham)[:,None] * dalphamdyj)
                  + (self.bi / (bm * bm))[:,None] * dbmdyj)
                 * (0.3535533905932738 * A / B)
                 + gphii[:,None] * (dAdyj / A - dBdyj / B))
    dlnphiidyj = ((self.bi / bm)[:,None] * (dZdyj - (Z - 1.) / bm * dbmdyj)
                  + (fZ * dgphiidyj + gphii[:,None] * dfZdyj)
                  - (dZdyj - dBdyj) / (Z - B))
    dZdP = (B * mdqdB + A * mdqdA) / (P * dqdZ)
    dlnphiidP = ((B / P - dZdP) / (Z - B)
                 + dZdP / bm * self.bi
                 + gphii * (dZdP * dfZdZ + B / P * dfZdB)
                 - self.vsi_bi / RT)
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * self.D.dot(yi * dsqrtalphaidT)
    dalphamdT = yi.dot(dSidT)
    dBdT = -B / T
    dZdT = (dBdT * mdqdB + (PRT / RT * dalphamdT - 2. * A / T) * mdqdA) / dqdZ
    dfZdT = dZdT * dfZdZ + dBdT * dfZdB
    dgphiidT = (((2. * dSidT - dalphamdT / bm * self.bi)
                 / (2.82842712474619 * RT * bm))
                - gphii / T)
    dlnphiidT = (dfZdT * gphii
                 + fZ * dgphiidT
                 - (dZdT - dBdT) / (Z - B)
                 + dZdT / bm * self.bi
                 + PRT / T * self.vsi_bi)
    return (Z - PRT * yi.dot(self.vsi_bi), lnphii,
            dlnphiidP, dlnphiidT, dlnphiidyj)

  def getPT_Z_lnphii_dP_d2P(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[float, Vector[Double], Vector[Double], Vector[Double]]:
    """Computes fugacity coefficients of components and their first
    and second partial derivatives with respect to pressure.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector[Double]` of shape `(Nc,)`,
    - second partial derivatives of logarithms of fugacity coefficients
      with respect to pressure as a `Vector[Double]` of shape `(Nc,)`.
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
    Z = self.solve(A, B)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    lnphii = ((Z - 1.) / bm * self.bi
              - np.log(Z - B)
              + np.log(ZpB / ZmB) * gphii
              - PRT * self.vsi_bi)
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B
    dAdP = A / P
    dBdP = B / P
    dZdP = ((B * (2. * (A - B) - 3. * B * B)
             + Z * (6. * B * B + 2. * B - A)
             - B * Z * Z)
            / (P * dqdZ))
    dlnphiidP = ((dBdP - dZdP) / (Z - B)
                 + dZdP / bm * self.bi
                 + gphii * (dZdP * (ZmB - ZpB)
                            - dBdP * (0.414213562373095 * ZmB
                                      + 2.414213562373095 * ZpB))
                 - self.vsi_bi / RT)
    d2ZdP2 = -2. * (dBdP * (dBdP * (3. * B - 3. * Z + 1.) - dAdP)
                    + dZdP * (2. * dBdP * (Z - 3. * B - 1.) + dAdP)
                    + dZdP * dZdP * (3. * Z + B - 1.)) / dqdZ
    d2fZdP2 = (dZdP * dZdP * (ZpB * ZpB - ZmB * ZmB)
               + d2ZdP2 * (ZmB - ZpB)
               + dBdP * dBdP * (5.82842712474619 * ZpB * ZpB
                                - 0.17157287525381 * ZmB * ZmB)
               + 2. * dZdP * dBdP * (0.414213562373095 * ZmB * ZmB
                                     + 2.414213562373095 * ZpB * ZpB))
    d2lnphiidP2 = (d2ZdP2 / bm * self.bi
                   + (((dZdP - dBdP) / (Z - B))**2 - d2ZdP2 / (Z - B))
                   + d2fZdP2 * gphii)
    return Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidP, d2lnphiidP2

  def getPT_Z_lnphii_dT_d2T(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[float, Vector[Double], Vector[Double], Vector[Double]]:
    """Computes fugacity coefficients of components and their first
    and second partial derivatives with respect to temperature.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple that contains:
    - the compressibility factor of the mixture,
    - logarithms of fugacity coefficients of components as a
      `Vector[Double]` of shape `(Nc,)`,
    - partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector[Double]` of shape
      `(Nc,)`,
    - second partial derivatives of logarithms of fugacity coefficients
      with respect to temperature as a `Vector[Double]` of shape
      `(Nc,)`.
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
    Z = self.solve(A, B)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = np.log(ZpB / ZmB)
    lnphii = ((Z - 1.) / bm * self.bi
              - np.log(Z - B)
              + 0.3535533905932738 * fZ * gphii
              - PRT * self.vsi_bi)
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B
    dfZdZ = ZmB - ZpB
    dfZdB = -0.414213562373095 * ZmB - 2.414213562373095 * ZpB
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT_ = self.D.dot(yi * dsqrtalphaidT)
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * dSidT_
    dalphamdT = yi.dot(dSidT)
    dAdT = PRT / RT * dalphamdT - 2. * A / T
    dBdT = -B / T
    dZdT = (dBdT * (A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z)
            + dAdT * (B - Z)) / dqdZ
    dfZdT = dZdT * dfZdZ + dBdT * dfZdB
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - (dZdT - dBdT) / (Z - B)
                 + dZdT / bm * self.bi
                 + PRT / T * self.vsi_bi)
    d2sqrtalphaidT2 = dsqrtalphaidT * (-.5 / T)
    d2SidT2 = (d2sqrtalphaidT2 * Si_
               + 2. * dsqrtalphaidT * dSidT_
               + sqrtalphai * self.D.dot(yi * d2sqrtalphaidT2))
    d2alphamdT2 = yi.dot(d2SidT2)
    d2AdT2 = PRT / RT * (d2alphamdT2 - dalphamdT / T) - 3. * dAdT / T
    d2BdT2 = -2. * dBdT / T
    d2ZdT2 = (2. * dAdT * dBdT
              - d2AdT2 * (Z - B)
              - d2BdT2 * (Z * (Z - 6. * B - 2.) + 3. * B * B + 2. * B - A)
              - 2. * dBdT * dBdT * (1. - 3. * (Z - B))
              - 2. * dZdT * (2. * dBdT * (Z - 3. * B - 1.) + dAdT)
              - 2. * dZdT * dZdT * (3. * Z + B - 1.)) / dqdZ
    d2fZdT2 = (dZdT * dZdT * (ZpB * ZpB - ZmB * ZmB)
               + dBdT * dBdT * (5.82842712474619 * ZpB * ZpB
                                - 0.17157287525381 * ZmB * ZmB)
               + 2. * dZdT * dBdT * (0.414213562373095 * ZmB * ZmB
                                     + 2.414213562373095 * ZpB * ZpB)
               + d2ZdT2 * dfZdZ + d2BdT2 * dfZdB)
    d2gphiidT2 = ((2. * d2SidT2 - d2alphamdT2 / bm * self.bi) / (RT * bm)
                  - 2. / T * dgphiidT)
    d2lnphiidT2 = (0.3535533905932738 * (fZ * d2gphiidT2 + gphii * d2fZdT2
                                         + 2. * dfZdT * dgphiidT)
                   + d2ZdT2 / bm * self.bi
                   + (((dZdT - dBdT)/(Z - B))**2 - (d2ZdT2 - d2BdT2)/(Z - B))
                   - 2. * PRT / (T * T) * self.vsi_bi)
    return Z - PRT * yi.dot(self.vsi_bi), lnphii, dlnphiidT, d2lnphiidT2

  def getPT_Z_lnfi_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
    n: float = 1.,
  ) -> tuple[float, Vector[Double], Matrix[Double]]:
    """Computes fugacities of components and their partial derivatives
    with respect to component mole numbers.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple of:
    - the compressibility factor of the mixture,
    - logarithms of fugacities of components as a `Vector[Double]` of
      shape `(Nc,)`,
    - partial derivatives of logarithms of fugacities with respect
      to mole numbers of components as a `Matrix[Double]` of shape
      `(Nc, Nc)`.
    """
    Z, lnphii, dlnphiidnj = self.getPT_Z_lnphii_dnj(P, T, yi, n)
    lnfi = lnphii + np.log(yi * P)
    dlnfidnj = dlnphiidnj + np.diagflat(1. / (n * yi)) - 1. / n
    return Z, lnfi, dlnfidnj

  def getPT_Zj(
    self,
    Pj: float | Vector[Double],
    Tj: float | Vector[Double],
    yji: Vector[Double] | Matrix[Double],
  ) -> Vector[Double]:
    """Computes compressibility factors for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Double], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Double], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Double], shape (Nc,) | Matrix[Double], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    Returns
    -------
    Compressibility factors of mixtures as a `Vector[Double]` of shape
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
    Zj = np.vectorize(self.solve)(Aj, Bj)
    return Zj - yji.dot(self.vsi_bi) * PRTj

  def getPT_Zj_lnphiji(
    self,
    Pj: float | Vector[Double],
    Tj: float | Vector[Double],
    yji: Vector[Double] | Matrix[Double],
  ) -> tuple[Vector[Double], Matrix[Double]]:
    """Computes fugacity coefficients of components and compressibility
    factors for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Double], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Double], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Double], shape (Nc,) | Matrix[Double], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    Returns
    -------
    A tuple of:
    - compressibility factors of mixtures as a `Vector[Double]` of shape
      `(Np,)`,
    - logarithms of fugacity coefficients of components in mixtures as
      a `Matrix[Double]` of shape `(Np, Nc)`.
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
    Zj = np.vectorize(self.solve)(Aj, Bj)
    gphiji = ((0.3535533905932738 * Aj / Bj)[:,None]
              * (2. / alphamj[:,None] * Sji - self.bi / bmj[:,None]))
    fZj = np.log((Zj - Bj * 0.414213562373095)
                 / (Zj + Bj * 2.414213562373095))
    lnphiji = (self.bi * ((Zj - 1.) / bmj)[:,None]
               + gphiji * fZj[:,None]
               - np.log(Zj - Bj)[:,None]
               - PRTj[:,None] * self.vsi_bi)
    return Zj - yji.dot(self.vsi_bi) * PRTj, lnphiji

  def getPT_Zj_lnphiji_dnk(
    self,
    Pj: float | Vector[Double],
    Tj: float | Vector[Double],
    yji: Vector[Double] | Matrix[Double],
    nj: float | Matrix[Double] = 1.,
  ) -> tuple[Vector[Double], Matrix[Double], Tensor[Double]]:
    """Computes fugacity coefficients of components, compressibility
    factors and partial derivatives of fugacity coefficients with
    respect to component mole numbers for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Double], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Double], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Double], shape (Nc,) | Matrix[Double], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    nj: float | Vector[Double], shape (Np,)
      Mole number(s) of mixtures [mol]. It is allowed to specify
      different mole number for each mixture. In that case, `Np` is
      the number of mixtures. Default is `1.0` [mol].

    Returns
    -------
    A tuple of:
    - compressibility factors of mixtures as a `Vector[Double]` of shape
      `(Np,)`,
    - logarithms of fugacity coefficients of components in mixtures as
      a `Matrix[Double]` of shape `(Np, Nc)`,
    - partial derivatives of logarithms of fugacity coefficients with
      respect to component mole numbers for each mixture as a `Tensor[Double]`
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
    Zj = np.vectorize(self.solve)(Aj, Bj)
    gphiji = ((0.3535533905932738 * Aj / Bj)[:,None]
              * (2. / alphamj[:,None] * Sji - self.bi / bmj[:,None]))
    ZmBj = 1. / (Zj - Bj * 0.414213562373095)
    ZpBj = 1. / (Zj + Bj * 2.414213562373095)
    fZj = np.log(ZpBj / ZmBj)
    lnphiji = (self.bi * ((Zj - 1.) / bmj)[:,None]
               + gphiji * fZj[:,None]
               - np.log(Zj - Bj)[:,None]
               - PRTj[:,None] * self.vsi_bi)
    dSjidnk = (sqrtalphaji[:,:,None] * sqrtalphaji[:,None,:] * self.D
               - Sji[:,:,None]) / nj[:,None,None]
    dalphamjdnk = 2. / nj[:,None] * (Sji - alphamj[:,None])
    dbmjdnk = (self.bi - bmj[:,None]) / nj[:,None]
    dAjdnk = dalphamjdnk * (PRTj / RTj)[:,None]
    dBjdnk = dbmjdnk * PRTj[:,None]
    dZjdnk = ((dBjdnk * (Aj - 2. * Bj - 3. * Bj * Bj
                         + 6. * Zj * Bj + 2. * Zj - Zj * Zj)[:,None]
               + dAjdnk * (Bj - Zj)[:,None])
              / (3. * Zj * Zj + 2. * (Bj - 1.) * Zj
                 + Aj - 2. * Bj - 3. * Bj * Bj)[:,None])
    dfZjdnk = (dZjdnk * (ZmBj - ZpBj)[:,None]
               - dBjdnk * (0.414213562373095 * ZmBj
                           + 2.414213562373095 * ZpBj)[:,None])
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
      - ((dZjdnk - dBjdnk) / (Zj - Bj)[:,None])[:,None,:])
    return Zj - yji.dot(self.vsi_bi) * PRTj, lnphiji, dlnphiidnj

  def getPT_kvguess(
    self,
    P: float,
    T: float,
    yi: Vector[Double],
  ) -> tuple[Vector[Double], ...]:
    """Computes initial k-values for given pressure, temperature
    and composition.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
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
      uii = np.exp(self.getPT_lnphii(P, T, yi))
      return self.h1i, self.l1i, uii, kvi, 1. / kvi
    elif self.kvlevel == 3:
      uii = np.exp(self.getPT_lnphii(P, T, yi))
      return self.h1i, self.l1i, self.h2i, self.l2i, uii, kvi, 1. / kvi
    elif self.kvlevel == 4:
      uii = np.exp(self.getPT_lnphii(P, T, yi))
      cbrtkvi = np.cbrt(kvi)
      return *self.upji, uii, kvi, 1. / kvi, cbrtkvi, 1. / cbrtkvi
    else:
      raise ValueError(f'Unsupported level number: {self.kvlevel}.')

  def _getPT_PID_wilson(self, P: float, T: float, yi: Vector[Double]) -> int:
    kvim1 = (np.exp(5.3727 * (1. + self.wi) * (1. - self.Tci / T))
             * self.Pci / P - 1.)
    if yi.dot(kvim1 / (1. + .5 * kvim1)) > 0.:
      return 0
    else:
      return 1

  def _getPT_PIDj_wilson(
    self,
    Pj: float | Vector[Double],
    Tj: float | Vector[Double],
    yji: Vector[Double] | Matrix[Double],
  ) -> Vector[Integer]:
    Pj = np.atleast_1d(Pj)
    Tj = np.atleast_1d(Tj)
    yji = np.atleast_2d(yji)
    kvjim1 = (np.exp(5.3727 * (1. + self.wi) * (1. - self.Tci / Tj[:,None]))
             * self.Pci / Pj[:,None] - 1.)
    return np.where(np.vecdot(yji, kvjim1 / (1. + .5 * kvjim1)) > 0., 0, 1)

  def _getPT_PID_vcTc(self, P: float, T: float, yi: Vector[Double]) -> int:
    Z = self.getPT_Z(P, T, yi)
    v = Z * R * T / P
    vpc = yi.dot(self.vci)
    Tpc = yi.dot(self.Tci)
    if v * T * T > vpc * Tpc * Tpc:
      return 0
    else:
      return 1

  def _getPT_PIDj_vcTc(
    self,
    Pj: float | Vector[Double],
    Tj: float | Vector[Double],
    yji: Vector[Double] | Matrix[Double],
  ) -> Vector[Integer]:
    Zj = self.getPT_Zj(Pj, Tj, yji)
    vj = Zj * R * Tj / Pj
    vpcj = yji.dot(self.vci)
    Tpcj = yji.dot(self.Tci)
    return np.where(vj * Tj * Tj > vpcj * Tpcj * Tpcj, 0, 1)

  def _getPT_PID_dadT(self, P: float, T: float, yi: Vector[Double]) -> int:
    return 0

  def _getPT_PIDj_dadT(
    self,
    Pj: float | Vector[Double],
    Tj: float | Vector[Double],
    yji: Vector[Double] | Matrix[Double],
  ) -> Vector[Integer]:
    return np.array([0])

  def _getPT_PID_pip(self, P: float, T: float, yi: Vector[Double]) -> int:
    return 0

  def _getPT_PIDj_pip(
    self,
    Pj: float | Vector[Double],
    Tj: float | Vector[Double],
    yji: Vector[Double] | Matrix[Double],
  ) -> Vector[Integer]:
    return np.array([0])

  def getPT_PID(self, P: float, T: float, yi: Vector[Double]) -> int:
    """Disignate phase composition for a given pressure and temperature.
    The output of this method depends on the `phaseid` parameter of the
    `pr78` class.

    Parameters
    ----------
    P: float
      Pressure of the mixture [Pa].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The designated phase (`0` = vapour, `1` = liquid).
    """
    return self._getPT_PID(P, T, yi)

  def getPT_PIDj(
    self,
    Pj: float | Vector[Double],
    Tj: float | Vector[Double],
    yji: Vector[Double] | Matrix[Double],
  ) -> Vector[Integer]:
    """Disignate phase compositions for a given pressure(s) and
    temperature(s). The output of this method depends on the `phaseid`
    parameter of the `pr78` class.

    Parameters
    ----------
    Pj: float | Vector[Double], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Double], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Double], shape (Nc,) | Matrix[Double], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    Returns
    -------
    The designated phases (`0` = vapour, `1` = liquid) as a
    `Vector[Integer]` of shape `(Np,)`.
    """
    return self._getPT_PIDj(Pj, Tj, yji)

  def getVT_P(
    self,
    V: float,
    T: float,
    yi: Vector[Double],
    n: float = 1.,
  ) -> float:
    """Computes pressure for given volume, temperature and composition.

    Parameters
    ----------
    V: float
      Volume of the mixture [m3].

    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    Pressure in [Pa].
    """
    v = V / n + yi.dot(self.vsi_bi)
    multi = 1. + self.kappai * (1. - np.sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    return R * T / (v - bm) - alpham / (v * v + 2. * bm * v - bm * bm)

  def getVT_lnfi_dnj(
    self,
    V: float,
    T: float,
    yi: Vector[Double],
    n: float = 1.,
  ) -> tuple[Vector[Double], Matrix[Double]]:
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

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of the mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple that contains:
    - logarithms of fugacities of components as a `Vector[Double]` of
      shape `(Nc,)`,
    - partial derivatives of logarithms of fugacities with respect
      to mole numbers of components as a `Matrix[Double]` of shape
      `(Nc, Nc)`.
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
    V: float,
    T: float,
    yi: Vector[Double],
    zti: Vector[Double],
    n: float = 1.,
  ) -> float:
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

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    zti: Vector[Double], shape (Nc,)
      Component mole number changes [mol].

    n: float
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

  def getVT_vmin(self, T: float, yi: Vector[Double]) -> float:
    """Calculates the minimal molar volume.

    Parameters
    ----------
    T: float
      Temperature of the mixture [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The minimum molar volume in [m3/mol].
    """
    return yi.dot(self.bi)

  @staticmethod
  def fdG(Z1: float, Z2: float, A: float, B: float) -> float:
    """Calculates the Gibbs energy difference between two states
    corresponding to the roots of the equation of state.

    Parameters
    ----------
    Z1: float
      The first root (compressibility factor) of the equation of state.

    Z2: float
      The second root (compressibility factor) of the equation of state.

    A: float
      The coefficient of the cubic form of the equation of state.

    B: float
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

  def solve(self, A: float, B: float) -> float:
    """Solves the modified Peng-Robinson equation of state.

    This method implements Cardano's method to solve the cubic form of
    the equation of state.

    When the equation of state has two (or more) real roots, the correct
    solution will be chosen based on the comparison of Gibbs energies
    corresponding to these roots.

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
      return x
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
          return x0
        else:
          return x2
      elif x1 > B:
        dG = self.fdG(x0, x1, A, B)
        if dG < 0.:
          return x0
        else:
          return x1
      else:
        return x0
