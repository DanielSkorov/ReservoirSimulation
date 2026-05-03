from logging import (
  getLogger,
)

from typing import (
  Iterable,
  Literal,
  Self,
)

from math import (
  acos,
  cbrt,
  cos,
  log,
  sqrt,
)

from numpy import (
  argsort as np_argsort,
  atleast_1d as np_atleast_1d,
  atleast_2d as np_atleast_2d,
  cbrt as np_cbrt,
  exp as np_exp,
  fill_diagonal as np_fill_diagonal,
  full as np_full,
  log as np_log,
  power as np_power,
  sqrt as np_sqrt,
  tril_indices as np_tril_indices,
  vecdot as np_vecdot,
  vectorize as np_vectorize,
  where as np_where,
  zeros as np_zeros,
)

from resim.pvt.datatypes import (
  Logical,
  Integer,
  Float,
  Vector,
  Matrix,
  Tensor,
)

from resim.pvt.constants import (
  R,
)


logger = getLogger('eos')


class pr78(object):
  """Modified Peng-Robinson equation of state (1978).

  Parameters
  ----------
  Pci: Vector[Float], shape (Nc,)
    Critical pressures of `Nc` components [Pa].

  Tci: Vector[Float], shape (Nc,)
    Critical temperatures of `Nc` components [K].

  wi: Vector[Float], shape (Nc,)
    Acentric factors of `Nc` components.

  mwi: Vector[Float], shape (Nc,)
    Molar weights of `Nc` components [kg/mol].

  dij: Vector[Float], shape (Nc * (Nc - 1) // 2,)
    Binary interaction coefficients of `Nc` components as a lower
    triangle matrix.

  s0i: Vector[Float], shape (Nc,) | None
    Volume shift coefficients of `Nc` components. Default is `None`
    which means that they are all equal to zero.

  s1i: Vector[Float], shape (Nc,) | None
    Gradients of the linear temperature-dependent volume shift
    coefficients of `Nc` components. Default is `None` which means
    that they are all equal to zero.

  Trsi: Vector[Float], shape (Nc,) | None
    Reference temperatures in the linear temperature-dependent volume
    shift coefficients of `Nc` components. Default is `None` which means
    that they are all equal to `293.15` [K].

  kvlevel: int
    Regulates outputs of methods that generate initial guesses of
    k-values. Available options are listed in the following table:

    +-------+----------------------------------------------------------+
    | Value | Description                                              |
    +=======+==========================================================+
    | `0`   | The Wilson equation and its inverse.                     |
    +-------+----------------------------------------------------------+
    | `1`   | Previous + the pure component model for the heaviest and |
    |       | lightest components.                                     |
    +-------+----------------------------------------------------------+
    | `2`   | Previous + the ideal model.                              |
    +-------+----------------------------------------------------------+
    | `3`   | Previous + the pure component model for the next         |
    |       | heaviest and next lightest components.                   |
    +-------+----------------------------------------------------------+
    | `4`   | The Wilson equation and its inverse                      |
    |       | + the pure component model for each component            |
    |       | + the ideal model                                        |
    |       | + the cubic root of the Wilson equation and its inverse. |
    +-------+----------------------------------------------------------+

    Default is `0`. The above mentioned options are listed in the order
    of the complexity increase. For hydrocarbon systems, the default
    option is suited in most cases. If the water component is present
    in a mixture, the `kvlevel` should be increased. The cubic roots of
    the Wilson and its inverse can be useful for systems near the
    critical point, where k-values are close to unity.

  purefrc: float
    This parameter is used in the formula of the pure component model
    to calculate initial guesses of k-values. It is the summarized mole
    fraction of other components in the trial phase except the specific
    component. Must be greater than zero and lower than one. Default is
    `1e-8`.

  phaseid: int
    The phase identification method. Available options are listed in the
    following table:

    +-------+----------------------------------------------------------+
    | Value | Description                                              |
    +=======+==========================================================+
    | `0`   | The Wilson correlation of k-values and the value of the  |
    |       | Rachford-Rice function at the non-reference phase mole   |
    |       | fraction equals to `0.5` are used to designate a phase   |
    |       | state.                                                   |
    +-------+----------------------------------------------------------+
    | `1`   | The pseudo-critical volume and temperature of a mixture  |
    |       | are used to designate a phase state.                     |
    +-------+----------------------------------------------------------+
    | `2`   | The derivative with respect to temperature of the        |
    |       | thermal expansion coefficient of a mixture is used to    |
    |       | designate a phase state.                                 |
    +-------+----------------------------------------------------------+
    | `3`   | The derivative with respect to temperature of the        |
    |       | isothermal compressibility is used to designate a phase  |
    |       | state; for the VT-based approach, it is equivalent to    |
    |       | the complex parameter from G. Venkatarathnam and L.R.    |
    |       | Oellrich, 2011 (doi: 10.1016/j.fluid.2010.12.001).       |
    +-------+----------------------------------------------------------+

    Default is `0`. For the details of available phase identification
    methods, see the paper of J. Bennett and K.A.G. Schmidt, 2016
    (doi: 10.1021/acs.energyfuels.6b02316). The above mentioned methods
    are listed in the order of the complexity increase.

  vci: Vector[Float], shape (Nc,) | None
    Critical molar volumes [m³/mol] of `Nc` components. Default is
    `None` which means that they will be calculated using the
    critical compressibility factor of the EOS: `0.3074...`.

  form: Literal['PT', 'VT']
    The formalism of the equation of state. Available options are listed
    in the following table:

    +--------+---------------------------------------------------------+
    | `form` | Formalism (mode)                                        |
    +========+=========================================================+
    | `'PT'` | An initialized instance of this class behaves as a PT-  |
    |        | based equation of state.                                |
    +--------+---------------------------------------------------------+
    | `'VT'` | An initialized instance of this class behaves as a VT-  |
    |        | based equation of state.                                |
    +--------+---------------------------------------------------------+

    Default is `'PT'`.

  mwc5: float | Vector[Logical]
    Components with a molecular weight [kg/mol] above this value are
    attributed to the C5+ group. It also can be a vector of shape
    `(Nc,)` of boolean flags indicating whether each component must
    be included into the C5+ group. Default is `0.072` [kg/mol].

  Notes
  -----
  This class can be used to compute the fugacities of components,
  compressibility factor of a mixture and their partial derivatives
  with respect to pressure, temperature, and composition using the
  modified Peng-Robinson equation of state. Classical (van der Waals)
  mixing rules are used to calculate the attraction and repulsion
  parameters for mixtures.
  """
  def __init__(
    self,
    Pci: Vector[Float],
    Tci: Vector[Float],
    wi: Vector[Float],
    mwi: Vector[Float],
    dij: Vector[Float],
    s0i: Vector[Float] | None = None,
    s1i: Vector[Float] | None = None,
    Trsi: Vector[Float] | None = None,
    kvlevel: int = 0,
    purefrc: float = 1e-8,
    phaseid: int = 0,
    vci: Vector[Float] | None = None,
    form: Literal['PT', 'VT'] = 'PT',
    mwc5: float | Vector[Logical] = 0.072,
  ) -> None:
    self.name = 'Peng-Robinson (1978)'
    self.Nc = Pci.shape[0]
    self.Pci = Pci
    self.Tci = Tci
    self.wi = wi
    self.mwi = mwi
    self._Tci = 1. / np_sqrt(Tci)
    self.bi = 0.07779607390388849 * R * Tci / Pci
    self.sqrtai = 0.6761919320144113 * R * Tci / np_sqrt(Pci)
    w2i = wi * wi
    w3i = w2i * wi
    self.kappai = np_where(
      wi <= 0.491,
      0.37464 + 1.54226 * wi - 0.26992 * w2i,
      0.379642 + 1.48503 * wi - 0.164423 * w2i + 0.016666 * w3i,
    )
    D = np_zeros(shape=(self.Nc, self.Nc))
    self.ltridx = np_tril_indices(self.Nc, -1)
    D[self.ltridx] = dij
    self.D = 1. - (D + D.T)
    if s0i is None:
      self.vsibi = np_zeros(shape=(self.Nc,))
    else:
      self.vsibi = s0i * self.bi
    if s1i is None:
      self.vstibi = np_zeros(shape=(self.Nc,))
    else:
      self.vstibi = s1i * self.bi
    if Trsi is None:
      self.Trsi = np_full((self.Nc,), 293.15)
    else:
      self.Trsi = Trsi
    self.kvlevel = kvlevel
    idxs = np_argsort(self.mwi)
    h0 = idxs[-1]
    h1 = idxs[-2]
    l0 = idxs[0]
    l1 = idxs[1]
    upi = np_full(shape=(self.Nc,), fill_value=purefrc)
    yp = 1. - purefrc * (self.Nc - 1)
    self.h1i = upi.copy()
    self.h1i[h0] = yp
    self.l1i = upi.copy()
    self.l1i[l0] = yp
    self.h2i = upi.copy()
    self.h2i[h1] = yp
    self.l2i = upi.copy()
    self.l2i[l1] = yp
    self.upji = np_full(shape=(self.Nc, self.Nc), fill_value=upi)
    np_fill_diagonal(self.upji, yp)
    if vci is None:
      self.vci = (0.3074013086987038 * R * Tci / Pci
                  - (self.vsibi + self.vstibi * (Tci - self.Trsi)))
    else:
      self.vci = vci
    self.phaseid = phaseid
    self.form: Literal['PT', 'VT'] = form
    if isinstance(mwc5, float):
      self.c5pi = mwi > mwc5
    else:
      self.c5pi = mwc5
    self.mwc5pi = mwi[self.c5pi]
    pass

  def getPT_Z(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> float:
    """Compute the compressibility factor of a mixture.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    Compressibility factor of a mixture.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    return Z - PRT * yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))

  def getPT_Z_dP(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[float, float]:
    """Compute the compressibility factor of a mixture and its partial
    derivative with respect to pressure.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor of a mixture,
    - partial derivative of the compressibility factor with respect to
      pressure [1/Pa].
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    dZdP = ((B * (2. * (A - B) - 3. * B * B)
             + Z * (6. * B * B + 2. * B - A)
             - B * Z * Z)
            / (P * (3. * Z * Z
                    + 2. * (B - 1.) * Z
                    + A - 2. * B - 3. * B * B)))
    Zs = PRT * yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    return Z - Zs, dZdP - Zs / P

  def getPT_Z_dT(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[float, float]:
    """Compute the compressibility factor of a mixture and its partial
    derivative with respect to temperature.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor of a mixture,
    - partial derivative of the compressibility factor with respect to
      temperature [1/K].
    """
    RT = R * T
    PRT = P / RT
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT_ = self.D.dot(yi * dsqrtalphaidT)
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * dSidT_
    dalphamdT = yi.dot(dSidT)
    dAdT = PRT / RT * dalphamdT - 2. * A / T
    dBdT = -B / T
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B
    dZdT = (dBdT * (A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z)
            + dAdT * (B - Z)) / dqdZ
    Zs = PRT * yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    dZsdT = PRT * yi.dot(self.vstibi) - Zs / T
    return Z - Zs, dZdT - dZsdT

  def getPT_Z_dP_dP2(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[float, float, float]:
    """Compute the compressibility factor of a mixture and its first
    and second partial derivatives with respect to pressure.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor of a mixture,
    - partial derivative of the compressibility factor with respect to
      pressure [1/Pa].
    - second partial derivative of the compressibility factor with
      respect to pressure [1/Pa²].
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B
    dAdP = A / P
    dBdP = B / P
    dZdP = ((B * (2. * (A - B) - 3. * B * B)
             + Z * (6. * B * B + 2. * B - A)
             - B * Z * Z)
            / (P * dqdZ))
    d2ZdP2 = -2. * (dBdP * (dBdP * (3. * B - 3. * Z + 1.) - dAdP)
                    + dZdP * (2. * dBdP * (Z - 3. * B - 1.) + dAdP)
                    + dZdP * dZdP * (3. * Z + B - 1.)) / dqdZ
    Zs = PRT * yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    return Z - Zs, dZdP - Zs / P, d2ZdP2

  def getPT_Z_dT_dT2(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[float, float, float]:
    """Compute the compressibility factor of a mixture and its first
    and second partial derivatives with respect to temperature.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor of a mixture,
    - partial derivative of the compressibility factor with respect to
      temperature [1/K].
    - second partial derivative of the compressibility factor with
      respect to temperature [1/K²].
    """
    RT = R * T
    PRT = P / RT
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT_ = self.D.dot(yi * dsqrtalphaidT)
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * dSidT_
    dalphamdT = yi.dot(dSidT)
    dAdT = PRT / RT * dalphamdT - 2. * A / T
    dBdT = -B / T
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B
    mdqdA = B - Z
    mdqdB = A - B * (2. + 3. * B - 6. * Z) - Z * (Z - 2.)
    dZdT = (mdqdA * dAdT + mdqdB * dBdT) / dqdZ
    d2sqrtalphaidT2 = dsqrtalphaidT * (-.5 / T)
    d2SidT2 = (d2sqrtalphaidT2 * Si_
               + 2. * dsqrtalphaidT * dSidT_
               + sqrtalphai * self.D.dot(yi * d2sqrtalphaidT2))
    d2alphamdT2 = yi.dot(d2SidT2)
    d2AdT2 = PRT / RT * (d2alphamdT2 - dalphamdT / T) - 3. * dAdT / T
    d2BdT2 = -2. * dBdT / T
    d2ZdT2 = (2. * dAdT * dBdT
              + d2AdT2 * mdqdA
              + d2BdT2 * mdqdB
              - 2. * dBdT * dBdT * (1. - 3. * (Z - B))
              - 2. * dZdT * (2. * dBdT * (Z - 3. * B - 1.) + dAdT)
              - 2. * dZdT * dZdT * (3. * Z + B - 1.)) / dqdZ
    Zs = PRT * yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    dZsdT = PRT * yi.dot(self.vstibi) - Zs / T
    return Z - Zs, dZdT - dZsdT, d2ZdT2 + 2. / T * dZsdT

  def getPT_Z_dP_dT_dPdT(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[float, float, float, float]:
    """Compute the compressibility factor of a mixture and its partial
    derivatives with respect to pressure and temperature; and the second
    partial derivative with respect to pressure and temperature.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor of a mixture,
    - partial derivative of the compressibility factor with respect to
      pressure [1/Pa],
    - partial derivative of the compressibility factor with respect to
      temperature [1/K],
    - second partial derivative of the compressibility factor with
      respect to pressure and temperature [1/Pa/K].
    """
    RT = R * T
    PRT = P / RT
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B
    mdqdA = B - Z
    mdqdB = A - 2. * B - 3. * B * B + (6. * B + 2.) * Z - Z * Z
    dZdP = (mdqdA * A + mdqdB * B) / (P * dqdZ)
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT_ = self.D.dot(yi * dsqrtalphaidT)
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * dSidT_
    dalphamdT = yi.dot(dSidT)
    dAdT = PRT / RT * dalphamdT - 2. * A / T
    dBdT = -B / T
    dZdT = (mdqdA * dAdT + mdqdB * dBdT) / dqdZ
    d2ZdPdT = ((dAdT * (B + mdqdA)
                + dBdT * (A - 2. * B * (3. * (B - Z) + 1.) + mdqdB)
                - dZdT * (A + 2. * B * (Z - 3. * B - 1.))) / P
               - dZdP * (2. * dBdT * (Z - 3. * B - 1.)
                         + 2. * dZdT * (3. * Z + B - 1.) + dAdT)) / dqdZ
    Zs = PRT * yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    dZsdT = PRT * yi.dot(self.vstibi) - Zs / T
    return Z - Zs, dZdP - Zs / P, dZdT - dZsdT, d2ZdPdT - dZsdT / P

  def getPT_lnphii(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> Vector[Float]:
    """Compute natural logarithms of fugacity coefficients of
    components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A `Vector[Float]` of shape `(Nc,)` of logarithms of fugacity
    coefficients of components.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    fZ = log((Z - B * 0.414213562373095) / (Z + B * 2.414213562373095))
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    return ((Z - 1.) / bm * self.bi
            - log(Z - B)
            + fZ * gphii
            - PRT * (self.vsibi + self.vstibi * (T - self.Trsi)))

  def getPT_lnphii_dP(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[Vector[Float], Vector[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to pressure.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure [1/Pa].
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    lnphisi = PRT * (self.vsibi + self.vstibi * (T - self.Trsi))
    lnphii = ((Z - 1.) / bm * self.bi
              - log(Z - B)
              + log(ZpB / ZmB) * gphii
              - lnphisi)
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
                 - lnphisi / P)
    return lnphii, dlnphiidP

  def getPT_lnphii_dT(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[Vector[Float], Vector[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to
    temperature.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to temperature [1/K].
    """
    RT = R * T
    PRT = P / RT
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = log(ZpB / ZmB)
    lnphisi = PRT * (self.vsibi + self.vstibi * (T - self.Trsi))
    lnphii = (0.3535533905932738 * fZ * gphii
              - log(Z - B)
              + (Z - 1.) / bm * self.bi
              - lnphisi)
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
                 + lnphisi / T
                 - PRT * self.vstibi)
    return lnphii, dlnphiidT

  def getPT_lnphii_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
    pid: int = -1,
  ) -> tuple[Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to mole
    numbers of components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a mixture [mol]. Default is `1.0` [mol].

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to mole numbers of components [1/mol].
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = PRT / RT * alpham
    B = PRT * bm
    Z = self.solve(A, B, pid)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = log(ZpB / ZmB)
    lnphii = ((Z - 1.) / bm * self.bi
              - log(Z - B)
              + fZ * gphii
              - PRT * (self.vsibi + self.vstibi * (T - self.Trsi)))
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
    return lnphii, dlnphiidnj

  def getPT_lnphii_dyj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to mole
    fractions of components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to mole fractions of components.
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = log(ZpB / ZmB)
    lnphii = ((Z - 1.) / bm * self.bi
              - log(Z - B)
              + fZ * gphii
              - PRT * (self.vsibi + self.vstibi * (T - self.Trsi)))
    dSidyj = sqrtalphai[:,None] * sqrtalphai * self.D
    dalphamdyj = Si + yi.dot(dSidyj)
    dbmdyj = self.bi
    dAdyj = PRT / RT * dalphamdyj
    dBdyj = PRT * dbmdyj
    dZdyj = ((dBdyj * (A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z)
              + dAdyj * (B - Z))
             / (3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B))
    dfZdyj = dZdyj * (ZmB - ZpB) - dBdyj * (0.414213562373095 * ZmB
                                            + 2.414213562373095 * ZpB)
    dgphiidyj = ((2. / alpham * (dSidyj - (Si / alpham)[:,None] * dalphamdyj)
                  + (self.bi / (bm * bm))[:,None] * dbmdyj)
                 * (0.3535533905932738 * A / B)
                 + gphii[:,None] * (dAdyj / A - dBdyj / B))
    dlnphiidyj = ((self.bi / bm)[:,None] * (dZdyj - (Z - 1.) / bm * dbmdyj)
                  + (fZ * dgphiidyj + gphii[:,None] * dfZdyj)
                  - (dZdyj - dBdyj) / (Z - B))
    return lnphii, dlnphiidyj

  def getPT_lnphii_dP_dT(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to pressure
    and temperature.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure [1/Pa],
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to temperature [1/K].
    """
    RT = R * T
    PRT = P / RT
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = log(ZpB / ZmB)
    lnphisi = PRT * (self.vsibi + self.vstibi * (T - self.Trsi))
    lnphii = ((Z - 1.) / bm * self.bi
              - log(Z - B)
              + 0.3535533905932738 * fZ * gphii
              - lnphisi)
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B
    mdqdA = B - Z
    mdqdB = A - 2. * B - 3. * B * B + 6. * Z * B + 2. * Z - Z * Z
    dfZdZ = ZmB - ZpB
    dfZdB = -0.414213562373095 * ZmB - 2.414213562373095 * ZpB
    dZdP = (B * mdqdB + A * mdqdA) / (P * dqdZ)
    dlnphiidP = ((B / P - dZdP) / (Z - B)
                 + dZdP / bm * self.bi
                 + 0.3535533905932738 * (dZdP * dfZdZ + B / P * dfZdB) * gphii
                 - lnphisi / P)
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
                 + lnphisi / T
                 - PRT * self.vstibi)
    return lnphii, dlnphiidP, dlnphiidT

  def getPT_lnphii_dP_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
    pid: int = -1,
  ) -> tuple[Vector[Float], Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to pressure
    and mole numbers of components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a mixture [mol]. Default is `1.0` [mol].

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure [1/Pa],
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to mole numbers of components [1/mol].
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = log(ZpB / ZmB)
    lnphisi = PRT * (self.vsibi + self.vstibi * (T - self.Trsi))
    lnphii = (Z - 1.) / bm * self.bi - log(Z - B) + fZ * gphii - lnphisi
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
                 - lnphisi / P)
    return lnphii, dlnphiidP, dlnphiidnj

  def getPT_lnphii_dT_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
    pid: int = -1,
  ) -> tuple[Vector[Float], Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to temperature
    and mole numbers of components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a mixture [mol]. Default is `1.0` [mol].

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to temperature [1/K],
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to mole numbers of components [1/mol].
    """
    RT = R * T
    PRT = P / RT
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = log(ZpB / ZmB)
    lnphisi = PRT * (self.vsibi + self.vstibi * (T - self.Trsi))
    lnphii = (0.3535533905932738 * fZ * gphii
              - log(Z - B)
              + (Z - 1.) / bm * self.bi
              - lnphisi)
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
                 + lnphisi / T
                 - PRT * self.vstibi)
    return lnphii, dlnphiidT, dlnphiidnj

  def getPT_lnphii_dP_dT_dyj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to pressure,
    temperature, and mole fractions of components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure [1/Pa],
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to temperature [1/K],
    - a `Matrix[Float]` of shape `(Nc, Nc)` partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to mole fractions of components.

    Notes
    -----
    The mole fraction constraint isn't taken into account.
    """
    RT = R * T
    PRT = P / RT
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = log(ZpB / ZmB)
    lnphisi = PRT * (self.vsibi + self.vstibi * (T - self.Trsi))
    lnphii = (Z - 1.) / bm * self.bi - log(Z - B) + fZ * gphii - lnphisi
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
                 - lnphisi / P)
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
                 + lnphisi / T
                 - PRT * self.vstibi)
    return lnphii, dlnphiidP, dlnphiidT, dlnphiidyj

  def getPT_lnphii_dP_dP2(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their first and second partial derivatives with
    respect to pressure.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure [1/Pa],
    - a `Vector[Float]` of shape `(Nc,)` of second partial derivatives
      of natural logarithms of fugacity coefficients of components with
      respect to pressure [1/Pa²].
    """
    RT = R * T
    PRT = P / RT
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    gphii = 0.3535533905932738 * A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    lnphisi = PRT * (self.vsibi + self.vstibi * (T - self.Trsi))
    lnphii = ((Z - 1.) / bm * self.bi
              - log(Z - B)
              + log(ZpB / ZmB) * gphii
              - lnphisi)
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
                 - lnphisi / P)
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
    return lnphii, dlnphiidP, d2lnphiidP2

  def getPT_lnphii_dT_dT2(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their first and second partial derivatives with
    respect to temperature.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to temperature [1/K],
    - a `Vector[Float]` of shape `(Nc,)` of second partial derivatives
      of natural logarithms of fugacity coefficients of components with
      respect to temperature [1/K²].
    """
    RT = R * T
    PRT = P / RT
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = alpham * PRT / RT
    B = bm * PRT
    Z = self.solve(A, B, pid)
    gphii = A / B * (2. / alpham * Si - self.bi / bm)
    ZmB = 1. / (Z - B * 0.414213562373095)
    ZpB = 1. / (Z + B * 2.414213562373095)
    fZ = log(ZpB / ZmB)
    lnphisi = PRT * (self.vsibi + self.vstibi * (T - self.Trsi))
    lnphii = ((Z - 1.) / bm * self.bi
              - log(Z - B)
              + 0.3535533905932738 * fZ * gphii
              - lnphisi)
    dqdZ = 3. * Z * Z + 2. * (B - 1.) * Z + A - 2. * B - 3. * B * B
    mdqdA = B - Z
    mdqdB = A - B * (2. + 3. * B - 6. * Z) - Z * (Z - 2.)
    dfZdZ = ZmB - ZpB
    dfZdB = -0.414213562373095 * ZmB - 2.414213562373095 * ZpB
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT_ = self.D.dot(yi * dsqrtalphaidT)
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * dSidT_
    dalphamdT = yi.dot(dSidT)
    dAdT = PRT / RT * dalphamdT - 2. * A / T
    dBdT = -B / T
    dZdT = (mdqdA * dAdT + mdqdB * dBdT) / dqdZ
    dfZdT = dZdT * dfZdZ + dBdT * dfZdB
    dgphiidT = (2. * dSidT - dalphamdT / bm * self.bi) / (RT * bm) - gphii / T
    dlnphisidT = PRT * self.vstibi - lnphisi / T
    dlnphiidT = (0.3535533905932738 * (dfZdT * gphii + fZ * dgphiidT)
                 - (dZdT - dBdT) / (Z - B)
                 + dZdT / bm * self.bi
                 - dlnphisidT)
    d2sqrtalphaidT2 = dsqrtalphaidT * (-.5 / T)
    d2SidT2 = (d2sqrtalphaidT2 * Si_
               + 2. * dsqrtalphaidT * dSidT_
               + sqrtalphai * self.D.dot(yi * d2sqrtalphaidT2))
    d2alphamdT2 = yi.dot(d2SidT2)
    d2AdT2 = PRT / RT * (d2alphamdT2 - dalphamdT / T) - 3. * dAdT / T
    d2BdT2 = -2. * dBdT / T
    d2ZdT2 = (2. * dAdT * dBdT
              + d2AdT2 * mdqdA
              + d2BdT2 * mdqdB
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
                   + 2. / T * dlnphisidT)
    return lnphii, dlnphiidT, d2lnphiidT2

  def getPT_lnfi(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    pid: int = -1,
  ) -> Vector[Float]:
    """Compute fugacities of components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A `Vector[Float]` of shape `(Nc,)` of natural logarithms of
    fugacities of components.
    """
    return self.getPT_lnphii(P, T, yi, pid) + np_log(P * yi)

  def getPT_lnfi_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
    pid: int = -1,
  ) -> tuple[Vector[Float], Matrix[Float]]:
    """Compute fugacities of components and their partial derivatives
    with respect to mole numbers of components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a mixture [mol]. Default is `1.0` [mol].

    pid: int
      The phase designation index. Defines the cubic root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacities of components,
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacities of components with respect to
      mole numbers of components [1/mol].
    """
    lnphii, dlnphiidnj = self.getPT_lnphii_dnj(P, T, yi, n, pid)
    lnfi = lnphii + np_log(P * yi)
    np_fill_diagonal(dlnphiidnj, dlnphiidnj.diagonal() + 1. / (n * yi))
    return lnfi, dlnphiidnj - 1. / n

  def getPT_Zj(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
    pidj: int | Integer | Iterable[int | Integer] = -1,
  ) -> Vector[Float]:
    """Compute the compressibility factor for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole compositions for each mixture. In that case,
      `Np` is the number of mixtures.

    pidj: int | Integer | Iterable[int | Integer], shape (Np,)
      The phase designation index for each mixture. Defines the cubic
      root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A `Vector[Float]` of shape `(Np,)` of compressibility factors.
    """
    Pj = np_atleast_1d(Pj)
    Tj = np_atleast_1d(Tj)
    yji = np_atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    multji = 1. + self.kappai * (1. - np_sqrt(Tj)[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji = sqrtalphaji * (yji * sqrtalphaji).dot(self.D)
    alphamj = np_vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    if isinstance(pidj, (int, Integer)):
      Zj = np_vectorize(self.solve, excluded={2})(Aj, Bj, pidj)
    else:
      Zj = np_vectorize(self.solve)(Aj, Bj, pidj)
    return Zj - PRTj * np_vecdot(yji, (self.vsibi
                                       + self.vstibi
                                         * (Tj[:,None] - self.Trsi)))

  def getPT_Zj_dP(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
    pidj: int | Integer | Iterable[int | Integer] = -1,
  ) -> tuple[Vector[Float], Vector[Float]]:
    """Compute the compressibility factor and its partial derivative
    with respect to pressure for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    pidj: int | Integer | Iterable[int | Integer], shape (Np,)
      The phase designation index for each mixture. Defines the cubic
      root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor for each mixture as a `Vector[Float]`
      of shape `(Np,)`,
    - the partial derivative of the compressibility factor with respect
      to pressure for each mixture as a `Vector[Float]` of shape
      `(Np,)` [1/Pa].
    """
    Pj = np_atleast_1d(Pj)
    Tj = np_atleast_1d(Tj)
    yji = np_atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    multji = 1. + self.kappai * (1. - np_sqrt(Tj)[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji = sqrtalphaji * (yji * sqrtalphaji).dot(self.D)
    alphamj = np_vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    if isinstance(pidj, (int, Integer)):
      Zj = np_vectorize(self.solve, excluded={2})(Aj, Bj, pidj)
    else:
      Zj = np_vectorize(self.solve)(Aj, Bj, pidj)
    dZjdPj = ((Bj * (2. * (Aj - Bj) - 3. * Bj * Bj)
               + Zj * (6. * Bj * Bj + 2. * Bj - Aj)
               - Bj * Zj * Zj)
              / (Pj * (3. * Zj * Zj
                      + 2. * (Bj - 1.) * Zj
                      + Aj - 2. * Bj - 3. * Bj * Bj)))
    Zsj = PRTj * np_vecdot(yji, (self.vsibi
                                 + self.vstibi * (Tj[:,None] - self.Trsi)))
    return Zj - Zsj, dZjdPj - Zsj / Pj

  def getPT_Zj_dT(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
    pidj: int | Integer | Iterable[int | Integer] = -1,
  ) -> tuple[Vector[Float], Vector[Float]]:
    """Compute the compressibility factor and its partial derivatives
    with respect to temperature for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    pidj: int | Integer | Iterable[int | Integer], shape (Np,)
      The phase designation index for each mixture. Defines the cubic
      root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor for each mixture as a `Vector[Float]`
      of shape `(Np,)`,
    - the partial derivative of the compressibility factor with respect
      to temperature for each mixture as a `Vector[Float]` of shape
      `(Np,)` [1/K].
    """
    Pj = np_atleast_1d(Pj)
    Tj = np_atleast_1d(Tj)
    yji = np_atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    sqrtTj = np_sqrt(Tj)
    multji = 1. + self.kappai * (1. - sqrtTj[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji_ = (yji * sqrtalphaji).dot(self.D)
    Sji = sqrtalphaji * Sji_
    alphamj = np_vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    if isinstance(pidj, (int, Integer)):
      Zj = np_vectorize(self.solve, excluded={2})(Aj, Bj, pidj)
    else:
      Zj = np_vectorize(self.solve)(Aj, Bj, pidj)
    dmultjidTj = (-.5 / sqrtTj)[:,None] * self.kappai * self._Tci
    dsqrtalphajidTj = self.sqrtai * dmultjidTj
    dSjidTj_ = (yji * dsqrtalphajidTj).dot(self.D)
    dSjidTj = dsqrtalphajidTj * Sji_ + sqrtalphaji * dSjidTj_
    dalphamjdTj = np_vecdot(yji, dSjidTj)
    dAjdTj = PRTj / RTj * dalphamjdTj - 2. * Aj / Tj
    dBjdTj = -Bj / Tj
    dqjdZj = 3. * Zj * Zj + 2. * (Bj - 1.) * Zj + Aj - 2. * Bj - 3. * Bj * Bj
    dZjdTj = (dBjdTj * (Aj - Bj * (2. + 3. * Bj - 6. * Zj) - Zj * (Zj - 2.))
              + dAjdTj * (Bj - Zj)) / dqjdZj
    Zsj = PRTj * np_vecdot(yji, (self.vsibi
                                 + self.vstibi * (Tj[:,None] - self.Trsi)))
    dZsjdTj = PRTj * yji.dot(self.vstibi) - Zsj / Tj
    return Zj - Zsj, dZjdTj - dZsjdTj

  def getPT_Zj_dP_dP2(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
    pidj: int | Integer | Iterable[int | Integer] = -1,
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float]]:
    """Compute the compressibility factor and its first and second
    partial derivatives with respect to pressure for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    pidj: int | Integer | Iterable[int | Integer], shape (Np,)
      The phase designation index for each mixture. Defines the cubic
      root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor for each mixture as a `Vector[Float]`
      of shape `(Np,)`,
    - the first partial derivative of the compressibility factor with
      respect to pressure for each mixture as a `Vector[Float]` of shape
      `(Np,)` [1/Pa],
    - the second partial derivative of the compressibility factor with
      respect to pressure for each mixture as a `Vector[Float]` of shape
      `(Np,)` [1/Pa²].
    """
    Pj = np_atleast_1d(Pj)
    Tj = np_atleast_1d(Tj)
    yji = np_atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    multji = 1. + self.kappai * (1. - np_sqrt(Tj)[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji = sqrtalphaji * (yji * sqrtalphaji).dot(self.D)
    alphamj = np_vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    if isinstance(pidj, (int, Integer)):
      Zj = np_vectorize(self.solve, excluded={2})(Aj, Bj, pidj)
    else:
      Zj = np_vectorize(self.solve)(Aj, Bj, pidj)
    dqjdZj = 3. * Zj * Zj + 2. * (Bj - 1.) * Zj + Aj - 2. * Bj - 3. * Bj * Bj
    dAjdPj = Aj / Pj
    dBjdPj = Bj / Pj
    dZjdPj = ((Bj * (2. * (Aj - Bj) - 3. * Bj * Bj)
               + Zj * (6. * Bj * Bj + 2. * Bj - Aj)
               - Bj * Zj * Zj)
              / (Pj * dqjdZj))
    d2ZjdP2j = -2. * (dBjdPj * (dBjdPj * (3. * Bj - 3. * Zj + 1.) - dAjdPj)
                      + dZjdPj * (2. * dBjdPj * (Zj - 3. * Bj - 1.) + dAjdPj)
                      + dZjdPj * dZjdPj * (3. * Zj + Bj - 1.)) / dqjdZj
    Zsj = PRTj * np_vecdot(yji, (self.vsibi
                                 + self.vstibi * (Tj[:,None] - self.Trsi)))
    return Zj - Zsj, dZjdPj - Zsj / Pj, d2ZjdP2j

  def getPT_Zj_dT_dT2(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
    pidj: int | Integer | Iterable[int | Integer] = -1,
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float]]:
    """Compute the compressibility factor and its first and second
    partial derivatives with respect to temperature for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    pidj: int | Integer | Iterable[int | Integer], shape (Np,)
      The phase designation index for each mixture. Defines the cubic
      root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor for each mixture as a `Vector[Float]`
      of shape `(Np,)`,
    - the first partial derivative of the compressibility factor with
      respect to temperature for each mixture as a `Vector[Float]` of
      shape `(Np,)` [1/K],
    - the second partial derivative of the compressibility factor with
      respect to temperature for each mixture as a `Vector[Float]` of
      shape `(Np,)` [1/K²].
    """
    Pj = np_atleast_1d(Pj)
    Tj = np_atleast_1d(Tj)
    yji = np_atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    sqrtTj = np_sqrt(Tj)
    multji = 1. + self.kappai * (1. - sqrtTj[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji_ = (yji * sqrtalphaji).dot(self.D)
    Sji = sqrtalphaji * Sji_
    alphamj = np_vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    if isinstance(pidj, (int, Integer)):
      Zj = np_vectorize(self.solve, excluded={2})(Aj, Bj, pidj)
    else:
      Zj = np_vectorize(self.solve)(Aj, Bj, pidj)
    dmultjidTj = (-.5 / sqrtTj)[:,None] * self.kappai * self._Tci
    dsqrtalphajidTj = self.sqrtai * dmultjidTj
    dSjidTj_ = (yji * dsqrtalphajidTj).dot(self.D)
    dSjidTj = dsqrtalphajidTj * Sji_ + sqrtalphaji * dSjidTj_
    dalphamjdTj = np_vecdot(yji, dSjidTj)
    dAjdTj = PRTj / RTj * dalphamjdTj - 2. * Aj / Tj
    dBjdTj = -Bj / Tj
    dqjdZj = 3. * Zj * Zj + 2. * (Bj - 1.) * Zj + Aj - 2. * Bj - 3. * Bj * Bj
    mdqjdAj = Bj - Zj
    mdqjdBj = Aj - Bj * (2. + 3. * Bj - 6. * Zj) - Zj * (Zj - 2.)
    dZjdTj = (mdqjdAj * dAjdTj + mdqjdBj * dBjdTj) / dqjdZj
    d2sqrtalphajidT2j = dsqrtalphajidTj * (-.5 / Tj)[:,None]
    d2SjidT2 = (d2sqrtalphajidT2j * Sji_
                + 2. * dsqrtalphajidTj * dSjidTj_
                + sqrtalphaji * (yji * d2sqrtalphajidT2j).dot(self.D))
    d2alphamjdT2j = np_vecdot(yji, d2SjidT2)
    d2AjdT2j = (PRTj / RTj * (d2alphamjdT2j - dalphamjdTj / Tj)
                - 3. * dAjdTj / Tj)
    d2BjdT2j = -2. * dBjdTj / Tj
    d2ZjdT2j = (2. * dAjdTj * dBjdTj
                + d2AjdT2j * mdqjdAj
                + d2BjdT2j * mdqjdBj
                - 2. * dBjdTj * dBjdTj * (1. - 3. * (Zj - Bj))
                - 2. * dZjdTj * (2. * dBjdTj * (Zj - 3. * Bj - 1.) + dAjdTj)
                - 2. * dZjdTj * dZjdTj * (3. * Zj + Bj - 1.)) / dqjdZj
    Zsj = PRTj * np_vecdot(yji, (self.vsibi
                                 + self.vstibi * (Tj[:,None] - self.Trsi)))
    dZsjdTj = PRTj * yji.dot(self.vstibi) - Zsj / Tj
    return Zj - Zsj, dZjdTj - dZsjdTj, d2ZjdT2j + 2. / Tj * dZsjdTj

  def getPT_Zj_dP_dT_dPdT(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
    pidj: int | Integer | Iterable[int | Integer] = -1,
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float], Vector[Float]]:
    """Compute the compressibility factor and its partial derivatives
    with respect to pressure and temperature for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    pidj: int | Integer | Iterable[int | Integer], shape (Np,)
      The phase designation index for each mixture. Defines the cubic
      root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - the compressibility factor for each mixture as a `Vector[Float]`
      of shape `(Np,)`,
    - the first partial derivative of the compressibility factor with
      respect to pressure for each mixture as a `Vector[Float]` of
      shape `(Np,)` [1/Pa],
    - the first partial derivative of the compressibility factor with
      respect to temperature for each mixture as a `Vector[Float]` of
      shape `(Np,)` [1/K],
    - the second partial derivative of the compressibility factor with
      respect to pressure and temperature for each mixture as a
      `Vector[Float]` of shape `(Np,)` [1/Pa/K].
    """
    Pj = np_atleast_1d(Pj)
    Tj = np_atleast_1d(Tj)
    yji = np_atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    sqrtTj = np_sqrt(Tj)
    multji = 1. + self.kappai * (1. - sqrtTj[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji_ = (yji * sqrtalphaji).dot(self.D)
    Sji = sqrtalphaji * Sji_
    alphamj = np_vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    if isinstance(pidj, (int, Integer)):
      Zj = np_vectorize(self.solve, excluded={2})(Aj, Bj, pidj)
    else:
      Zj = np_vectorize(self.solve)(Aj, Bj, pidj)
    mdqjdAj = Bj - Zj
    mdqjdBj = Aj - Bj * (2. + 3. * Bj - 6. * Zj) - Zj * (Zj - 2.)
    dqjdZj = 3. * Zj * Zj + 2. * (Bj - 1.) * Zj + Aj - 2. * Bj - 3. * Bj * Bj
    dZjdPj = (mdqjdAj * Aj + mdqjdBj * Bj) / (Pj * dqjdZj)
    dmultjidTj = (-.5 / sqrtTj)[:,None] * self.kappai * self._Tci
    dsqrtalphajidTj = self.sqrtai * dmultjidTj
    dSjidTj_ = (yji * dsqrtalphajidTj).dot(self.D)
    dSjidTj = dsqrtalphajidTj * Sji_ + sqrtalphaji * dSjidTj_
    dalphamjdTj = np_vecdot(yji, dSjidTj)
    dAjdTj = PRTj / RTj * dalphamjdTj - 2. * Aj / Tj
    dBjdTj = -Bj / Tj
    dZjdTj = (dBjdTj * mdqjdBj + dAjdTj * mdqjdAj) / dqjdZj
    d2ZjdPjdTj = ((dAjdTj * (Bj + mdqjdAj)
                   + dBjdTj * (Aj - 2. * Bj * (3. * (Bj - Zj) + 1.) + mdqjdBj)
                   - dZjdTj * (Aj + 2. * Bj * (Zj - 3. * Bj - 1.))) / Pj
                  - dZjdPj * (2. * dBjdTj * (Zj - 3. * Bj - 1.)
                               + 2. * dZjdTj * (3. * Zj + Bj - 1.)
                               + dAjdTj)) / dqjdZj
    Zsj = PRTj * np_vecdot(yji, (self.vsibi
                                 + self.vstibi * (Tj[:,None] - self.Trsi)))
    dZsjdTj = PRTj * yji.dot(self.vstibi) - Zsj / Tj
    return (
      Zj - Zsj,
      dZjdPj - Zsj / Pj,
      dZjdTj - dZsjdTj,
      d2ZjdPjdTj - dZsjdTj / Pj,
    )

  def getPT_lnphiji(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
    pidj: int | Integer | Iterable[int | Integer] = -1,
  ) -> Matrix[Float]:
    """Compute natural logarithms of fugacity coefficients of
    components for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    pidj: int | Integer | Iterable[int | Integer], shape (Np,)
      The phase designation index for each mixture. Defines the cubic
      root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A `Matrix[Float]` of shape `(Np, Nc)` of natural logarithms of
    fugacity coefficients of components in mixtures.
    """
    Pj = np_atleast_1d(Pj)
    Tj = np_atleast_1d(Tj)
    yji = np_atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    multji = 1. + self.kappai * (1. - np_sqrt(Tj)[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji = sqrtalphaji * (yji * sqrtalphaji).dot(self.D)
    alphamj = np_vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    if isinstance(pidj, (int, Integer)):
      Zj = np_vectorize(self.solve, excluded={2})(Aj, Bj, pidj)
    else:
      Zj = np_vectorize(self.solve)(Aj, Bj, pidj)
    gphiji = ((0.3535533905932738 * Aj / Bj)[:,None]
              * (2. / alphamj[:,None] * Sji - self.bi / bmj[:,None]))
    fZj = np_log((Zj - Bj * 0.414213562373095)
                 / (Zj + Bj * 2.414213562373095))
    lnphisji = PRTj[:,None] * (self.vsibi
                               + self.vstibi * (Tj[:,None] - self.Trsi))
    lnphiji = (self.bi * ((Zj - 1.) / bmj)[:,None]
               + gphiji * fZj[:,None]
               - np_log(Zj - Bj)[:,None]
               - lnphisji)
    return lnphiji

  def getPT_lnphiji_dnk(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
    nj: float | Vector[Float] = 1.,
    pidj: int | Integer | Iterable[int | Integer] = -1,
  ) -> tuple[Matrix[Float], Tensor[Float]]:
    """Compute natural logarithms of fugacity coefficients of components
    and their partial derivatives with respect to component mole numbers
    for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    nj: float | Vector[Float], shape (Np,)
      Mole number(s) of mixtures [mol]. It is allowed to specify
      different mole number for each mixture. In that case, `Np` is
      the number of mixtures. Default is `1.0` [mol].

    pidj: int | Integer | Iterable[int | Integer], shape (Np,)
      The phase designation index for each mixture. Defines the cubic
      root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Matrix[Float]` of shape `(Np, Nc)` of natural logarithms of
      fugacity coefficients of components in mixtures,
    - a `Tensor[Float]` of shape `(Np, Nc, Nc)` of partial derivatives
      of natural logarithms of fugacity coefficients of components with
      respect to mole numbers of components in mixtures [1/mol].
    """
    Pj = np_atleast_1d(Pj)
    Tj = np_atleast_1d(Tj)
    yji = np_atleast_2d(yji)
    nj = np_atleast_1d(nj)
    RTj = R * Tj
    PRTj = Pj / RTj
    multji = 1. + self.kappai * (1. - np_sqrt(Tj)[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji = sqrtalphaji * (yji * sqrtalphaji).dot(self.D)
    alphamj = np_vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    if isinstance(pidj, (int, Integer)):
      Zj = np_vectorize(self.solve, excluded={2})(Aj, Bj, pidj)
    else:
      Zj = np_vectorize(self.solve)(Aj, Bj, pidj)
    gphiji = ((0.3535533905932738 * Aj / Bj)[:,None]
              * (2. / alphamj[:,None] * Sji - self.bi / bmj[:,None]))
    ZmBj = 1. / (Zj - Bj * 0.414213562373095)
    ZpBj = 1. / (Zj + Bj * 2.414213562373095)
    fZj = np_log(ZpBj / ZmBj)
    lnphisji = PRTj[:,None] * (self.vsibi
                               + self.vstibi * (Tj[:,None] - self.Trsi))
    lnphiji = (self.bi * ((Zj - 1.) / bmj)[:,None]
               + gphiji * fZj[:,None]
               - np_log(Zj - Bj)[:,None]
               - lnphisji)
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
    dlnphijidnk = (
      ((self.bi / bmj[:,None])[:,:,None]
       * (dZjdnk - ((Zj - 1.) / bmj)[:,None] * dbmjdnk)[:,None,:])
      + (fZj[:,None,None] * dgphijidnk + gphiji[:,:,None] * dfZjdnk[:,None,:])
      - ((dZjdnk - dBjdnk) / (Zj - Bj)[:,None])[:,None,:])
    return lnphiji, dlnphijidnk

  def getPT_lnphiji_dP_dT_dyk(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
    pidj: int | Integer | Iterable[int | Integer] = -1,
  ) -> tuple[Matrix[Float], Matrix[Float], Matrix[Float], Tensor[Float]]:
    """Compute natural logarithms of fugacity coefficients of components
    and their partial derivatives with respect to pressure, tenperature,
    and component mole fractions for each mixture.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    pidj: int | Integer | Iterable[int | Integer], shape (Np,)
      The phase designation index for each mixture. Defines the cubic
      root selection:
      - `-1`: root with the lower Gibbs energy (default);
      - `0`: vapor phase (largest) root;
      - other: liquid phase (lowest) root.

    Returns
    -------
    A tuple containing:
    - a `Matrix[Float]` of shape `(Np, Nc)` of natural logarithms of
      fugacity coefficients of components in mixtures,
    - a `Matrix[Float]` of shape `(Np, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure in mixtures [1/Pa],
    - a `Matrix[Float]` of shape `(Np, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to temperature in mixtures [1/K],
    - a `Tensor[Float]` of shape `(Np, Nc, Nc)` of partial derivatives
      of natural logarithms of fugacity coefficients of components with
      respect to mole fractions of components in mixtures [1/mol].

    Notes
    -----
    The mole fraction constraint isn't taken into account.
    """
    Pj = np_atleast_1d(Pj)
    Tj = np_atleast_1d(Tj)
    yji = np_atleast_2d(yji)
    RTj = R * Tj
    PRTj = Pj / RTj
    sqrtTj = np_sqrt(Tj)
    multji = 1. + self.kappai * (1. - sqrtTj[:,None] * self._Tci)
    sqrtalphaji = self.sqrtai * multji
    Sji_ = (yji * sqrtalphaji).dot(self.D)
    Sji = sqrtalphaji * Sji_
    alphamj = np_vecdot(yji, Sji)
    bmj = yji.dot(self.bi)
    Aj = alphamj * PRTj / RTj
    Bj = bmj * PRTj
    if isinstance(pidj, (int, Integer)):
      Zj = np_vectorize(self.solve, excluded={2})(Aj, Bj, pidj)
    else:
      Zj = np_vectorize(self.solve)(Aj, Bj, pidj)
    gphiji = ((0.3535533905932738 * Aj / Bj)[:,None]
              * (2. / alphamj[:,None] * Sji - self.bi / bmj[:,None]))
    ZmBj = 1. / (Zj - Bj * 0.414213562373095)
    ZpBj = 1. / (Zj + Bj * 2.414213562373095)
    fZj = np_log(ZpBj / ZmBj)
    lnphisji = PRTj[:,None] * (self.vsibi
                               + self.vstibi * (Tj[:,None] - self.Trsi))
    lnphiji = (self.bi * ((Zj - 1.) / bmj)[:,None]
               + gphiji * fZj[:,None]
               - np_log(Zj - Bj)[:,None]
               - lnphisji)
    mdqjdAj = Bj - Zj
    mdqjdBj = Aj - Bj * (3. * Bj + 2.) + Zj * (6. * Bj + 2. - Zj)
    dfZjdZj = ZmBj - ZpBj
    dfZjdBj = -0.414213562373095 * ZmBj - 2.414213562373095 * ZpBj
    dqjdZj = Zj * (3. * Zj + 2. * Bj - 2.) - Bj * (3. * Bj + 2.) + Aj
    dZjdPj = (Bj * mdqjdBj + Aj * mdqjdAj) / (Pj * dqjdZj)
    dlnphijidPj = (((Bj / Pj - dZjdPj) / (Zj - Bj))[:,None]
                   + (dZjdPj / bmj)[:,None] * self.bi
                   + (dZjdPj * dfZjdZj + Bj / Pj * dfZjdBj)[:,None] * gphiji
                   - lnphisji / Pj[:,None])
    dmultjidTj = (-.5 / sqrtTj)[:,None] * (self.kappai * self._Tci)
    dsqrtalphajidTj = self.sqrtai * dmultjidTj
    dSjidTj = (dsqrtalphajidTj * Sji_
               + sqrtalphaji * (yji * dsqrtalphajidTj).dot(self.D))
    dalphamjdTj = np_vecdot(yji, dSjidTj)
    dAjdTj = PRTj / RTj * dalphamjdTj - 2. * Aj / Tj
    dBjdTj = -Bj / Tj
    dZjdTj = (dBjdTj * mdqjdBj + dAjdTj * mdqjdAj) / dqjdZj
    dfZjdTj = dZjdTj * dfZjdZj + dBjdTj * dfZjdBj
    dgphijidTj = (((2. * dSjidTj - (dalphamjdTj / bmj)[:,None] * self.bi)
                   / (2.82842712474619 * (RTj * bmj))[:,None])
                  - gphiji / Tj[:,None])
    dlnphijidTj = (dfZjdTj[:,None] * gphiji
                   + fZj[:,None] * dgphijidTj
                   - ((dZjdTj - dBjdTj) / (Zj - Bj))[:,None]
                   + (dZjdTj / bmj)[:,None] * self.bi
                   + lnphisji / Tj[:,None]
                   - PRTj[:,None] * self.vstibi)
    dSjidyk = sqrtalphaji[:,:,None] * sqrtalphaji[:,None,:] * self.D[None,:,:]
    dalphamjdyk = Sji + np_vecdot(dSjidyk, yji[:,:,None], axes=[(1,), (1,)])
    dbmjdyk = self.bi[None,:]
    dAjdyk = (PRTj / RTj)[:,None] * dalphamjdyk
    dBjdyk = PRTj[:,None] * dbmjdyk
    dZjdyk = (
      dBjdyk * mdqjdBj[:,None] + dAjdyk * mdqjdAj[:,None]
    ) / dqjdZj[:,None]
    dfZjdyk = dfZjdZj[:,None] * dZjdyk + dfZjdBj[:,None] * dBjdyk
    dgphijidyk = (0.3535533905932738 * (Aj / Bj))[:,None,None] * (
      (2. / alphamj)[:,None,None] * (
        dSjidyk - (Sji / alphamj[:,None])[:,:,None] * dalphamjdyk[:,None,:]
      )
      + (self.bi / (bmj * bmj)[:,None])[:,:,None] * dbmjdyk[:,None,:]
    ) + gphiji[:,:,None] * (dAjdyk/Aj[:,None] - dBjdyk/Bj[:,None])[:,None,:]
    dlnphijidyk = (
      (self.bi / bmj[:,None])[:,:,None]
      * (dZjdyk - ((Zj - 1.) / bmj)[:,None] * dbmjdyk)[:,None,:]
      + (fZj[:,None,None] * dgphijidyk + gphiji[:,:,None] * dfZjdyk[:,None,:])
      - ((dZjdyk - dBjdyk) / (Zj - Bj)[:,None])[:,None,:]
    )
    return lnphiji, dlnphijidPj, dlnphijidTj, dlnphijidyk

  def getPT_kvguess(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[Vector[Float], ...]:
    """Compute initial k-values for a given pressure, temperature,
    and composition.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple containing vectors of shape `(Nc,)` of initial k-values.
    """
    kvi = self.Pci * np_exp(5.3727 * (1. + self.wi) * (1. - self.Tci / T)) / P
    if self.kvlevel == 0:
      return kvi, 1. / kvi
    elif self.kvlevel == 1:
      return kvi, 1. / kvi, self.h1i / yi, self.l1i / yi
    elif self.kvlevel == 2:
      kvpi = np_exp(self.getPT_lnphii(P, T, yi))
      return kvi, 1. / kvi, self.h1i / yi, self.l1i / yi, kvpi
    elif self.kvlevel == 3:
      kvpi = np_exp(self.getPT_lnphii(P, T, yi))
      return (kvi, 1. / kvi, self.h1i / yi, self.l1i / yi, self.h2i / yi,
              self.l2i / yi, kvpi)
    elif self.kvlevel == 4:
      kvpi = np_exp(self.getPT_lnphii(P, T, yi))
      cbrtkvi = np_cbrt(kvi)
      return kvi, 1. / kvi, *(self.upji / yi), kvpi, cbrtkvi, 1. / cbrtkvi
    else:
      raise ValueError(f'Unsupported level number: {self.kvlevel}.')

  def getPT_PID(self, P: float, T: float, yi: Vector[Float]) -> int:
    """Obtain the phase designation index for a given pressure,
    temperature, and composition.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    An integer that designates a phase (`0` = vapour, `1` = liquid).
    """
    if self.phaseid == 0:
      kvim1 = (np_exp(5.3727 * (1. + self.wi) * (1. - self.Tci / T))
               * self.Pci / P - 1.)
      if yi.dot(kvim1 / (1. + .5 * kvim1)) > 0.:
        return 0
      else:
        return 1
    elif self.phaseid == 1:
      Z = self.getPT_Z(P, T, yi)
      v = Z * R * T / P
      vpc = yi.dot(self.vci)
      Tpc = yi.dot(self.Tci)
      if v * T * T > vpc * Tpc * Tpc:
        return 0
      else:
        return 1
    elif self.phaseid == 2:
      Z, dZdT, d2ZdT2 = self.getPT_Z_dT_dT2(P, T, yi)
      if Z * T * (2. * dZdT + T * d2ZdT2) - (T * dZdT + Z)**2 > 0.:
        return 1
      else:
        return 0
    elif self.phaseid == 3:
      Z, dZdP, dZdT, d2ZdPdT = self.getPT_Z_dP_dT_dPdT(P, T, yi)
      if dZdP * dZdT - Z * d2ZdPdT > 0.:
        return 1
      else:
        return 0
    else:
      raise ValueError(
        f'Unknown phase designation method: "{self.phaseid}".'
      )

  def getPT_PIDj(
    self,
    Pj: float | Vector[Float],
    Tj: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
  ) -> Vector[Integer]:
    """Obtain phase designation indices.

    Parameters
    ----------
    Pj: float | Vector[Float], shape (Np,)
      Pressure(s) of mixtures [Pa]. It is allowed to specify different
      pressure for each mixture. In that case, `Np` is the number of
      mixtures.

    Tj: float | Vector[Float], shape (Np,)
      Temperature(s) of mixtures [K]. It is allowed to specify different
      temperature for each mixture. In that case, `Np` is the number of
      mixtures.

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components. It is allowed to specify
      different mole fraction arrays for each mixture. In that case,
      `Np` is the number of mixtures.

    Returns
    -------
    A `Vector[Integer]` of shape `(Np,)` of designation indices
    (`0` = vapour, `1` = liquid).
    """
    if self.phaseid == 0:
      Pj = np_atleast_1d(Pj)
      Tj = np_atleast_1d(Tj)
      yji = np_atleast_2d(yji)
      kvjim1 = (np_exp(5.3727 * (1. + self.wi) * (1. - self.Tci / Tj[:,None]))
                * self.Pci / Pj[:,None] - 1.)
      return np_where(np_vecdot(yji, kvjim1 / (1. + .5 * kvjim1)) > 0., 0, 1)
    elif self.phaseid == 1:
      Zj = self.getPT_Zj(Pj, Tj, yji)
      vj = Zj * R * Tj / Pj
      vpcj = yji.dot(self.vci)
      Tpcj = yji.dot(self.Tci)
      return np_where(vj * Tj * Tj > vpcj * Tpcj * Tpcj, 0, 1)
    elif self.phaseid == 2:
      Zj, dZjdTj, d2ZjdT2j = self.getPT_Zj_dT_dT2(Pj, Tj, yji)
      aj = 2. * dZjdTj + Tj * d2ZjdT2j
      bj = Tj * dZjdTj + Zj
      return np_where(Zj * Tj * aj - bj * bj > 0., 1, 0)
    elif self.phaseid == 3:
      Zj, dZjdPj, dZjdTj, d2ZjdPjdTj = self.getPT_Zj_dP_dT_dPdT(Pj, Tj, yji)
      return np_where(dZjdPj * dZjdTj - Zj * d2ZjdPjdTj > 0., 1, 0)
    else:
      raise ValueError(
        f'Unknown phase identification method: "{self.phaseid}".'
      )

  def getVT_P(self, v: float, T: float, yi: Vector[Float]) -> float:
    """Compute pressure for a given molar volume, temperature, and
    composition.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    Pressure [Pa].
    """
    v2p = v + yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    return R * T / (v2p - bm) - alpham / (v2p * v2p + 2. * bm * v2p - bm * bm)

  def getVT_P_dv(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[float, float]:
    """Compute pressure and its partial derivative with respect to
    molar volume for a given molar volume, temperature, and composition.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple containing:
    - pressure [Pa],
    - the partial derivative of pressure with respect to molar volume
      [Pa/(m³/mol)].
    """
    v2p = v + yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    d1 = R * T / (v2p - bm)
    d2 = 1. / (v2p * (v2p + 2. * bm) - bm * bm)
    return (
      d1 - alpham * d2,
      - d1 / (v2p - bm) + 2. * alpham * (v2p + bm) * d2 * d2,
    )

  def getVT_P_dT(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[float, float]:
    """Compute pressure and its partial derivative with respect to
    temperature for a given molar volume, temperature, and composition.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple containing:
    - pressure [Pa],
    - the partial derivative of pressure with respect to temperature
      [Pa/K].
    """
    v2p = v + yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    d1 = R / (v2p - bm)
    d2 = 1. / (v2p * (v2p + 2. * bm) - bm * bm)
    dv2pdT = yi.dot(self.vstibi)
    dd1dT = -d1 * d1 / R * dv2pdT
    dd2dT = -2. * d2 * d2 * (v2p + bm) * dv2pdT
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT_ = self.D.dot(yi * dsqrtalphaidT)
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * dSidT_
    dalphamdT = yi.dot(dSidT)
    return (
      T * d1 - alpham * d2,
      d1 + T * dd1dT - dalphamdT * d2 - alpham * dd2dT,
    )

  def getVT_P_dv_dv2(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[float, float, float]:
    """Compute pressure and its first and second partial derivatives
    with respect to molar volume for a given molar volume, temperature,
    and composition.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple containing:
    - pressure [Pa],
    - the first partial derivative of pressure with respect to molar
      volume [Pa/(m³/mol)],
    - the second partial derivative of pressure with respect to molar
      volume [Pa/(m³/mol)²].
    """
    v2p = v + yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si = sqrtalphai * self.D.dot(yi * sqrtalphai)
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    d0 = v2p - bm
    d1 = R * T / d0
    d2 = v2p * (v2p + 2. * bm) - bm * bm
    d3 = d2 * d2
    d4 = v2p + bm
    return (
      d1 - alpham / d2,
      2. * alpham * (v2p + bm) / d3 - d1 / d0,
      2. * (d1 / (d0 * d0) + alpham * (d2 - 4. * d4 * d4) / (d3 * d2)),
    )

  def getVT_P_dT_dT2(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[float, float, float]:
    """Compute pressure and its first and second partial derivatives
    with respect to temperature for a given molar volume, temperature,
    and composition.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple containing:
    - pressure [Pa],
    - the first partial derivative of pressure with respect to
      temperature [Pa/K],
    - the second partial derivative of pressure with respect to
      temperature [Pa/K²].
    """
    v2p = v + yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    d1 = R / (v2p - bm)
    d2 = 1. / (v2p * (v2p + 2. * bm) - bm * bm)
    dv2pdT = yi.dot(self.vstibi)
    dd1dT = -d1 * d1 / R * dv2pdT
    dd2dT = -2. * d2 * d2 * (v2p + bm) * dv2pdT
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT_ = self.D.dot(yi * dsqrtalphaidT)
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * dSidT_
    dalphamdT = yi.dot(dSidT)
    d2d1dT2 = -2. * d1 * dd1dT / R * dv2pdT
    d2d2dT2 = -2. * dv2pdT * d2 * (2. * dd2dT * (v2p + bm) + d2 * dv2pdT)
    d2sqrtalphaidT2 = dsqrtalphaidT * (-.5 / T)
    d2SidT2 = (d2sqrtalphaidT2 * Si_
               + 2. * dsqrtalphaidT * dSidT_
               + sqrtalphai * self.D.dot(yi * d2sqrtalphaidT2))
    d2alphamdT2 = yi.dot(d2SidT2)
    return (
      T * d1 - alpham * d2,
      d1 + T * dd1dT - dalphamdT * d2 - alpham * dd2dT,
      2. * (dd1dT - dalphamdT * dd2dT) + T * d2d1dT2 - d2alphamdT2 * d2
      - alpham * d2d2dT2,
    )

  def getVT_P_dv_dT_dv2_dT2_dvdT(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[float, float, float, float, float, float]:
    """Compute pressure and its first and second partial derivatives
    with respect to molar volume and temperature for a given molar
    volume, temperature, and composition.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A tuple containing:
    - pressure [Pa],
    - the first partial derivative of pressure with respect to molar
      volume [Pa/(m³/mol)],
    - the first partial derivative of pressure with respect to
      temperature [Pa/K],
    - the second partial derivative of pressure with respect to molar
      volume [Pa/(m³/mol)²],
    - the second partial derivative of pressure with respect to
      temperature [Pa/K²],
    - the second partial derivative of pressure with respect to molar
      volume and temperature [Pa/(m³/mol*K)].
    """
    v2p = v + yi.dot(self.vsibi + self.vstibi * (T - self.Trsi))
    sqrtT = sqrt(T)
    multi = 1. + self.kappai * (1. - sqrtT * self._Tci)
    sqrtalphai = self.sqrtai * multi
    Si_ = self.D.dot(yi * sqrtalphai)
    Si = sqrtalphai * Si_
    alpham = yi.dot(Si)
    bm = yi.dot(self.bi)
    dmultidT = (-.5 / sqrtT) * self.kappai * self._Tci
    dsqrtalphaidT = self.sqrtai * dmultidT
    dSidT_ = self.D.dot(yi * dsqrtalphaidT)
    dSidT = dsqrtalphaidT * Si_ + sqrtalphai * dSidT_
    dalphamdT = yi.dot(dSidT)
    d2sqrtalphaidT2 = dsqrtalphaidT * (-.5 / T)
    d2SidT2 = (d2sqrtalphaidT2 * Si_
               + 2. * dsqrtalphaidT * dSidT_
               + sqrtalphai * self.D.dot(yi * d2sqrtalphaidT2))
    d2alphamdT2 = yi.dot(d2SidT2)
    d0 = 1. / (v2p - bm)
    d1 = R * T * d0
    d2 = 1. / (v2p * (v2p + 2. * bm) - bm * bm)
    d3 = d2 * d2
    d4 = v2p + bm
    dv2pdT = yi.dot(self.vstibi)
    dPdv = 2. * alpham * (v2p + bm) * d3 - d1 * d0
    d2Pdv2 = 2. * (d1 * d0 * d0 + alpham * (1. / d2 - 4. * d4 * d4) * d3 * d2)
    d2PdvdT = 2. * dalphamdT * (v2p + bm) * d3 - R * d0 * d0 + d2Pdv2 * dv2pdT
    return (
      d1 - alpham * d2,
      dPdv,
      R * d0 - dalphamdT * d2 + dPdv * dv2pdT,
      d2Pdv2,
      (dv2pdT * (2. * dalphamdT * (v2p + bm) * d3 - R * d0 * d0 + d2PdvdT)
       - d2alphamdT2 * d2),
      d2PdvdT,
    )

  def getVT_lnfi_dnj(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
  ) -> tuple[Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacities of components and their
    partial derivatives with respect to component mole numbers for a
    given molar volume, temperature, and composition.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacities of components,
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacities with respect to mole numbers of
      components [1/mol].

    Notes
    -----
    Partial derivatives formulas were taken from the paper of M.L.
    Michelsen and R.A. Heidemann, 1981 (doi: 10.1002/aic.690270326).
    """
    d1 = 2.414213562373095
    d2 = -0.414213562373095
    RT = R * T
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
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
    F5 = 2. * log((k + d1) / (k + d2)) / (d1 - d2)
    F6 = F2 - F5
    lnfi = (np_log(yi * (RT / (v - bm)))
            + bti * (F1 - F0 * k / ((k + d2) * (k + d1)))
            - F5 / 2. * F0 * (2. * gmi - bti))
    aij = sqrtalphai[:,None] * sqrtalphai * self.D
    bij = bti[:,None] * bti
    cij = gmi[:,None] * bti
    cti = bti * F1
    Q = (F0 * (bij * F3 - F5 / alpham * aij + F6 * (bij - cij - cij.T))
         + (cti[:,None] + cti) + bij * (F1 * F1))
    np_fill_diagonal(Q, Q.diagonal() + 1. / yi)
    return lnfi, Q / n

  def getVT_d3F(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
    zti: Vector[Float],
    n: float = 1.,
  ) -> float:
    """Compute the cubic form of the Helmholtz energy Taylor series
    decomposition for a given molar volume, temperature, composition
    of a mixture, and a vector of component mole number changes.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    zti: Vector[Float], shape (Nc,)
      Component mole number changes [mol].

    n: float
      Mole number of a mixture [mol]. Default is `1.0` [mol].

    Returns
    -------
    The cubic form of the Helmholtz energy Taylor series decomposition.

    Notes
    -----
    This method is used by the critical point calculation procedure.
    Calculation formulas were taken from the paper of M.L. Michelsen and
    R.A. Heidemann, 1981 (doi: 10.1002/aic.690270326).
    """
    d1 = 2.414213562373095
    d2 = -0.414213562373095
    RT = R * T
    multi = 1. + self.kappai * (1. - sqrt(T) * self._Tci)
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
    F5 = 2. * log((k + d1) / (k + d2)) / (d1 - d2)
    F6 = F2 - F5
    zts = zti.sum()
    btm = zti.dot(bti)
    gmm = zti.dot(gmi)
    ti = sqrtalphai * self.D.dot(zti * sqrtalphai)
    tm = ti.dot(zti) / alpham
    C = (RT * (3. * zts * (btm * F1)**2 + 2. * (btm * F1)**3
               - np_power(zti, 3).dot(1. / (yi * yi)))
         + (3. * btm**2 * (2. * gmm - btm) * (F3 + F6) - 2. * btm**3 * F4
            - 3. * btm * tm * F6) * alpham / bm)
    return C / (n * n)

  def getVT_vmin(self, T: float, yi: Vector[Float]) -> float:
    """Compute the minimum molar volume for a given temperature and
    composition of a mixture.

    Parameters
    ----------
    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The minimum molar volume [m³/mol].
    """
    return yi.dot(self.bi)

  def getVT_PID(self, v: float, T: float, yi: Vector[Float]) -> int:
    """Obtain the phase designation index for a given volume,
    temperature, and composition of a mixture.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    An integer that designates a phase (`0` = vapour, `1` = liquid).
    """
    if self.phaseid == 0:
      kvim1 = (np_exp(5.3727 * (1. + self.wi) * (1. - self.Tci / T))
               * self.Pci / self.getVT_P(v, T, yi) - 1.)
      if yi.dot(kvim1 / (1. + .5 * kvim1)) > 0.:
        return 0
      else:
        return 1
    elif self.phaseid == 1:
      vpc = yi.dot(self.vci)
      Tpc = yi.dot(self.Tci)
      if v * T * T > vpc * Tpc * Tpc:
        return 0
      else:
        return 1
    elif self.phaseid == 2:
      _, _dv, _dT, _dv2, _dT2, _dvdT = self.getVT_P_dv_dT_dv2_dT2_dvdT(
        v, T, yi,
      )
      dvdT = -_dT / _dv
      d2vdT2 = -(_dT2 + 2. * _dvdT * dvdT + _dv2 * dvdT) / _dv
      if v * d2vdT2 - dvdT * dvdT > 0.:
        return 1
      else:
        return 0
    elif self.phaseid == 3:
      _, _dv, _dT, _dv2, _, _dvdT = self.getVT_P_dv_dT_dv2_dT2_dvdT(
        v, T, yi,
      )
      if _dvdT / _dT - _dv2 / _dv - 1. / v > 0.:
        return 1
      else:
        return 0
    else:
      raise ValueError(
        f'Unknown phase designation method: "{self.phaseid}".'
      )

  @staticmethod
  def fdG(Z1: float, Z2: float, A: float, B: float) -> float:
    """Compute the Gibbs energy difference between two states
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
    return (log((Z2 - B) / (Z1 - B))
            + (Z1 - Z2)
            + log((Z1 - B * 0.414213562373095)
                  * (Z2 + B * 2.414213562373095)
                  / ((Z1 + B * 2.414213562373095)
                     * (Z2 - B * 0.414213562373095)))
              * 0.3535533905932738 * A / B)

  def solve(self, A: float, B: float, pid: int | Integer = -1) -> float:
    """Solve the modified Peng-Robinson equation of state.

    Parameters
    ----------
    A: float
      The coefficient of the cubic form of the modified Peng-Robinson
      equation of state.

    B: float
      The coefficient of the cubic form of the modified Peng-Robinson
      equation of state.

    Returns
    -------
    The solution of the modified Peng-Robinson equation of state
    corresponding to the lowest Gibbs energy.

    Notes
    -----
    This method implements Cardano's method to solve the cubic form of
    the equation of state.

    When the equation of state has three real roots, the correct
    solution will be chosen based on the comparison of Gibbs energies
    corresponding to these roots.

    In cases with three real roots, the first one is calculated using
    Cardano's formula. Once the first real root is known, the cubic
    polynomial can be "deflated", i.e., divided by a linear factor.
    The other roots are then found by solving the quadratic equation.
    For details, see the paper of U.K. Deiters and R. Macias-Salinas,
    2014 (doi: 10.1021/ie4038664).

    The accuracy of Cardano's method is known to be low near the
    critical point or in the low-temperature and low-pressure region.
    Therefore, one Newton iteration will be performed to refine the
    solution if the absolute value of the equation is greater than
    `1e-12`.
    """
    b = B - 1.
    c = A - B * (2. + 3. * B)
    d = B * (B * (1. + B) - A)
    r = b * b
    p = c + r / -3.
    q = d + b * (2. * r - 9. * c) / 27.
    s = q * q * .25 + p * p * p / 27.
    if s > 0.:
      s = sqrt(s)
      x = cbrt(-.5 * q + s) + cbrt(-.5 * q - s) + b / -3.
      y = d + x * (c + x * (b + x))
      if y > 1e-12 or y < -1e-12:
        x -= y / (c + x * (2. * b + 3. * x))
      return x
    else:
      x0 = (2. * sqrt(p / -3.) * cos(acos(1.5 * q * sqrt(-3. / p) / p) / 3.)
            + b / -3.)
      y0 = d + x0 * (c + x0 * (b + x0))
      if y0 > 1e-12 or y0 < -1e-12:
        x0 -= y0 / (c + x0 * (2. * b + 3. * x0))
      r = b + x0
      D = sqrt(r * r + 4. * d / x0)
      x1 = (-r + D) * .5
      x2 = (-r - D) * .5
      if pid < 0:
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
      elif pid:
        if x2 > B:
          return x2
        elif x1 > B:
          return x1
        else:
          return x0
      else:
        return x0

  def solve_iter(self, A: float, B: float, pid: int | Integer = -1) -> float:
    """Solve the modified Peng-Robinson equation of state.

    Parameters
    ----------
    A: float
      The coefficient of the cubic form of the modified Peng-Robinson
      equation of state.

    B: float
      The coefficient of the cubic form of the modified Peng-Robinson
      equation of state.

    Returns
    -------
    The solution of the modified Peng-Robinson equation of state
    corresponding to the lowest Gibbs energy.

    Notes
    -----
    This method implements Halley's method to solve the cubic form of
    the equation of state.

    When the equation of state has three real roots, the correct
    solution will be chosen based on the comparison of Gibbs energies
    corresponding to these roots.

    This implementation starts with determining the number of real
    roots. In cases with only one real root, the sign of the equation
    at the inflection point is used to determine whether this root is
    vapour-like or liquid-like. For details on this step, see the paper
    of J. Zhao et al, 2025 (doi: 10.1016/j.fluid.2025.114466). If the
    root is vapour-like, the initial guess is calculated according to
    R. Gosset et al, 1986 (doi: 10.1016/0378-3812(86)85061-0).
    Otherwise, the Vieta initialization scheme is used (U.K. Deiters
    and R. Macias-Salinas, 2014 (doi: 10.1021/ie4038664)). To prevent
    oscillations of the iterative method, the sign of the first
    derivative is monitored as described by R. Gosset et al, 1986
    (doi: 10.1016/0378-3812(86)85061-0).

    In cases with three real roots, the initial guess for the maximum
    root is calculated using the Laguerre-Nair-Samuelson initialization
    scheme as mentioned by U.K. Deiters and R. Macias-Salinas, 2014
    (doi: 10.1021/ie4038664). This procedure ensures that the starting
    point will be greater than the root, and the iterative procedure
    can be used safely due to Darboux's theorem. Once the first real
    root is known, the cubic polynomial can be "deflated", i.e.,
    divided by a linear factor. The other roots are then found by
    solving the resulting quadratic equation.

    It should be noted that in all cases, if Halley's method does not
    converge in ten iterations, Cardano's formula will be used with
    one Newton's step to refine the solution if the absolute value
    of the equation is greater than `1e-12`.

    Based on the benchmarks, this implementation is more than twice
    as slow as solving the EOS using Cardano's formula. Therefore,
    the `solve` method is used by default to solve the cubic form of
    the EOS. However, such results of the program execution time test
    were obtained due to the overhead of the Python interpreter.
    Fortran implementation of this function is faster than using
    Cardano's formula.
    """
    b = B - 1.
    c = A - B * (2. + 3. * B)
    d = B * (B * (1. + B) - A)
    r = b * b
    p = c + r / -3.
    q = d + b * (2. * r - 9. * c) / 27.
    s = q * q * .25 + p * p * p / 27.
    if s > 0.:
      x = b / -3.
      if d + x * (c + x * (b + x)) < 0.:
        x = 1. if B < 1. else B
        for _ in range(10):
          dydx = c + x * (2. * b + 3. * x)
          if dydx < 0.:
            x *= 2.
            continue
          else:
            y = d + x * (c + x * (b + x))
            d2ydx2 = 6. * x + 2. * b
          dx = y * dydx / (dydx * dydx - .5 * y * d2ydx2)
          if dx < 1e-12 and dx > -1e-12:
            break
          x -= dx
        else:
          s = sqrt(s)
          x = cbrt(-.5 * q + s) + cbrt(-.5 * q - s) + b / -3.
          y = d + x * (c + x * (b + x))
          if y > 1e-12 or y < -1e-12:
            x -= y / (c + x * (2. * b + 3. * x))
          return x
      else:
        x = -d / c
        if x < B:
          x = B
        for _ in range(10):
          dydx = c + x * (2. * b + 3. * x)
          if dydx < 0.:
            x *= .3
            if x < B:
              x = B
          else:
            y = d + x * (c + x * (b + x))
            d2ydx2 = 6. * x + 2. * b
            dx = y * dydx / (dydx * dydx - .5 * y * d2ydx2)
            if dx > -1e-12 and dx < 1e-12:
              break
            x -= dx
        else:
          s = sqrt(s)
          x = cbrt(-.5 * q + s) + cbrt(-.5 * q - s) + b / -3.
          y = d + x * (c + x * (b + x))
          if y > 1e-12 or y < -1e-12:
            x -= y / (c + x * (2. * b + 3. * x))
          return x
      return x
    else:
      x0 = b / -3. + 2. / 3. * sqrt(-3. * p)
      for _ in range(10):
        dydx = c + x0 * (2. * b + 3. * x0)
        d2ydx2 = 6. * x0 + 2. * b
        y = d + x0 * (c + x0 * (b + x0))
        dx = y * dydx / (dydx * dydx - .5 * y * d2ydx2)
        if dx < 1e-12 and dx > -1e-12:
          break
        x0 -= dx
      else:
        x0 = (2. * sqrt(p / -3.) * cos(acos(1.5 * q * sqrt(-3. / p) / p) / 3.)
              + b / -3.)
        y0 = d + x0 * (c + x0 * (b + x0))
        if y0 > 1e-12 or y0 < -1e-12:
          x0 -= y0 / (c + x0 * (2. * b + 3. * x0))
      r = b + x0
      D = sqrt(r * r + 4. * d / x0)
      x1 = (-r + D) * .5
      x2 = (-r - D) * .5
      if pid < 0:
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
      elif pid:
        if x2 > B:
          return x2
        elif x1 > B:
          return x1
        else:
          return x0
      else:
        return x0

  def update_eos(
    self,
    Pci: Vector[Float],
    Tci: Vector[Float],
    wi: Vector[Float],
    mwi: Vector[Float],
    dij: Vector[Float],
    s0i: Vector[Float] | None = None,
    s1i: Vector[Float] | None = None,
    Trsi: Vector[Float] | None = None,
    vci: Vector[Float] | None = None,
  ) -> None:
    """Update all EOS parameters without class initialization.

    Parameters
    ----------
    Pci: Vector[Float], shape (Nc,)
      Critical pressures of `Nc` components [Pa].

    Tci: Vector[Float], shape (Nc,)
      Critical temperatures of `Nc` components [K].

    wi: Vector[Float], shape (Nc,)
      Acentric factors of `Nc` components.

    mwi: Vector[Float], shape (Nc,)
      Molar weights of `Nc` components [kg/mol].

    dij: Vector[Float], shape (Nc * (Nc - 1) // 2,)
      Binary interaction coefficients of `Nc` components as a lower
      triangle matrix.

    s0i: Vector[Float], shape (Nc,)
      Volume shift coefficients of `Nc` components.

    s1i: Vector[Float], shape (Nc,) | None
      Gradients of the linear temperature-dependent volume shift
      coefficients of `Nc` components. Default is `None` which means
      that they will not be changed.

    Trsi: Vector[Float], shape (Nc,) | None
      Reference temperatures in the linear temperature-dependent volume
      shift coefficients of `Nc` components. Default is `None` which
      means that they will not be changed.

    vci: Vector[Float], shape (Nc,) | None
      Critical molar volumes [m³/mol] of `Nc` components. Default is
      `None` which means that they will not be changed.
    """
    self.Pci = Pci
    self.Tci = Tci
    self.wi = wi
    self.mwi = mwi
    self._Tci = 1. / np_sqrt(Tci)
    self.bi = 0.07779607390388849 * R * Tci / Pci
    self.sqrtai = 0.6761919320144113 * R * Tci / np_sqrt(Pci)
    w2i = wi * wi
    w3i = w2i * wi
    self.kappai = np_where(
      wi <= 0.491,
      0.37464 + 1.54226 * wi - 0.26992 * w2i,
      0.379642 + 1.48503 * wi - 0.164423 * w2i + 0.016666 * w3i,
    )
    D = np_zeros(shape=(self.Nc, self.Nc))
    D[np_tril_indices(self.Nc, -1)] = dij
    self.D = 1. - (D + D.T)
    if s0i is not None:
      self.vsibi = s0i * self.bi
    if s1i is not None:
      self.vstibi = s1i * self.bi
    if Trsi is not None:
      self.Trsi = Trsi
    if vci is not None:
      self.vci = vci
    pass

  def update_Pci(self, Pci: Vector[Float]) -> None:
    """Update critical pressures of components and EOS parameters
    dependent on them without class initialization.

    Parameters
    ----------
    Pci: Vector[Float], shape (Nc,)
      Critical pressures of `Nc` components [Pa].
    """
    self.Pci = Pci
    self.bi = 0.07779607390388849 * R * self.Tci / Pci
    self.sqrtai = 0.6761919320144113 * R * self.Tci / np_sqrt(Pci)
    pass

  def update_Tci(self, Tci: Vector[Float]) -> None:
    """Update critical temperatures of components and EOS parameters
    dependent on them without class initialization.

    Parameters
    ----------
    Tci: Vector[Float], shape (Nc,)
      Critical temperatures of `Nc` components [K].
    """
    self.Tci = Tci
    self._Tci = 1. / np_sqrt(Tci)
    self.bi = 0.07779607390388849 * R * Tci / self.Pci
    self.sqrtai = 0.6761919320144113 * R * Tci / np_sqrt(self.Pci)
    pass

  def update_wi(self, wi: Vector[Float]) -> None:
    """Update acentric factors of components and EOS parameters
    dependent on them without class initialization.

    Parameters
    ----------
    wi: Vector[Float], shape (Nc,)
      Acentric factors of `Nc` components [K].
    """
    self.wi = wi
    w2i = wi * wi
    w3i = w2i * wi
    self.kappai = np_where(
      wi <= 0.491,
      0.37464 + 1.54226 * wi - 0.26992 * w2i,
      0.379642 + 1.48503 * wi - 0.164423 * w2i + 0.016666 * w3i,
    )
    pass

  def update_mwi(self, mwi: Vector[Float]) -> None:
    """Update molar weights of components without class initialization.

    Parameters
    ----------
    mwi: Vector[Float], shape (Nc,)
      Molar weights of `Nc` components [kg/mol].
    """
    self.mwi = mwi
    pass

  def update_vsi(
    self,
    vsi: Vector[Float],
    vsti: Vector[Float] | None = None,
    Trsi: Vector[Float] | None = None,
  ) -> None:
    """Update volume shift parameters of components and EOS parameters
    dependent on them without class initialization.

    Parameters
    ----------
    vsi: Vector[Float], shape (Nc,)
      Volume shift parameters of `Nc` components.

    vsti: Vector[Float], shape (Nc,) | None
      Gradients of the linear temperature-dependent volume shift
      parameters of `Nc` components. Default is `None` which means
      that they will not be changed.

    Trsi: Vector[Float], shape (Nc,) | None
      Reference temperatures in the linear temperature-dependent volume
      shift parameters of `Nc` components. Default is `None` which means
      that they will not be changed.
    """
    self.vsibi = vsi * self.bi
    if vsti is not None:
      self.vsti = vsti
    if Trsi is not None:
      self.Trsi = Trsi
    pass

  def update_dij(self, dij: Vector[Float]) -> None:
    """Update binary interaction coefficients and EOS parameters
    dependent on them without class initialization.

    Parameters
    ----------
    dij: Vector[Float], shape (Nc * (Nc - 1) // 2,)
      Binary interaction coefficients of `Nc` components as a lower
      triangle matrix.
    """
    D = np_zeros(shape=(self.Nc, self.Nc))
    D[self.ltridx] = dij
    self.D = 1. - (D + D.T)
    pass

  def update_vci(self, vci: Vector[Float]) -> None:
    """Update critical molar volumes of components and EOS parameters
    dependent on them without class initialization.

    Parameters
    ----------
    vci: Vector[Float], shape (Nc,)
      Critical molar volumes [m³/mol] of `Nc` components.
    """
    self.vci = vci
    pass

  def update_form(self, newform: Literal['PT', 'VT']) -> None:
    self.form = newform
    pass

  def replace_components(self, names: Iterable[str]) -> None:
    raise NotImplementedError(
      'Replacement of components by their names is not implemented yet.'
    )

  def append_components(self, names: Iterable[str]) -> None:
    raise NotImplementedError(
      'Addition of components by their names is not implemented yet.'
    )

  @classmethod
  def init_by_names(cls, names: Iterable[str]) -> Self:
    raise NotImplementedError(
      'Class initialization using names of components is not implemented yet.'
    )
