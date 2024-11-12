import numpy as np
import numpy.typing as npt

from check_inputs import check_PTyi


R = 8.3144598 # Universal gas constant [J/mol/K]


class vdw(object):
  def __init__(
    self,
    Pci: npt.NDArray[np.floating],
    Tci: npt.NDArray[np.floating],
    dij: npt.NDArray[np.floating],
  ) -> None:
    """
    Allows to calculate component fugacities and the compressibility
    factor of a mixture of `Nc` components using the van der Waals
    equation of state.

    Arguments:
      Pci : critical pressures of components [Pa].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.
      Tci : critical temperatures of components [K].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.
      dij : the lower triangle of the binary interaction coefficients
        matrix expressed as a numpy.ndarray[numpy.floating] with the
        shape of `(Nc * (Nc - 1) // 2,)`.
    """
    args = (Pci, Tci, dij)
    if not all(map(lambda a: isinstance(a, np.ndarray), args)):
      raise TypeError(
        "Types of all input arrays must be numpy.ndarray, but"
        f"\n\t{type(Pci)=},\n\t{type(Tci)=},\n\t{type(dij)=}."
      )
    if not all(map(lambda a: np.issubdtype(a.dtype, np.floating), args)):
      raise TypeError(
        "Data types of all input arrays must be numpy.floating, but"
        f"\n\t{Pci.dtype=},\n\t{Tci.dtype=},\n\t{dij.dtype=}."
      )
    if not all(map(lambda a: a.ndim == 1, args)):
      raise ValueError(
        "Dimensions of all input arrays must be equal 1, but"
        f"\n\t{Pci.ndim=},\n\t{Tci.ndim=},\n\t{dij.ndim=}."
      )
    self.Nc = Pci.shape[0]
    if not all(map(lambda a: a.shape == (self.Nc,), args[:-1])):
      raise ValueError(
        f"The shape of Pci and Tci arrays must be equal to ({self.Nc},), but"
        f"\n\t{Pci.shape=},\n\t{Tci.shape=}."
      )
    if dij.shape != (self.Nc * (self.Nc - 1) // 2,):
      raise ValueError(
        f"The shape of the dij must be ({self.Nc*(self.Nc-1)//2,}), but"
        f"\n\t{dij.shape=}."
      )
    if not np.isfinite(Pci).all():
      where = np.where(1 - np.isfinite(Pci))
      raise ValueError(
        "All input arrays must contain finite values, but"
        f"\nPci have values: {Pci[where]}\n\tat indices: {where}."
      )
    if not np.isfinite(Tci).all():
      where = np.where(1 - np.isfinite(Tci))
      raise ValueError(
        "All input arrays must contain finite values, but"
        f"\n\tTci have values: {Tci[where]}\n\tat indices: {where}."
      )
    if not np.isfinite(dij).all():
      where = np.where(1 - np.isfinite(dij))
      raise ValueError(
        "All input arrays must contain finite values, but"
        f"\n\tdij have values: {dij[where]}\n\tat indices: {where}."
      )
    if (Pci <= 0.).any():
      raise ValueError(
        "Critical pressures of all components must be greater than zero."
      )
    if (Tci <= 0.).any():
      raise ValueError(
        "Critical temperatures of all components must be greater than zero."
      )
    self.sqrtai = .649519052838329 * R * Tci / np.sqrt(Pci)
    self.bi = .125 * R * Tci / Pci
    D = np.zeros(shape=(self.Nc, self.Nc), dtype=Pci.dtype)
    D[np.tril_indices(self.Nc, -1)] = dij
    self.D = 1. - (D + D.T)
    pass


  def get_Z(
    self,
    P: np.floating,
    T: np.floating,
    yi: npt.NDArray[np.floating],
    check_input: bool = True,
  ) -> np.floating:
    """
    Calculates the compressibility factor using the van der Waals
    equation of state.

    Arguments:
      P: pressure [Pa]. Must be a scalar of numpy.floating.
      T: temperature [K]. Must be a scalar of numpy.floating.
      yi: mole fraction for each component [fr.].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.

    Returns:
      The compressibility factor.
    """
    if check_input:
      check_PTyi(P, T, yi, self.Nc)
    Si = self.sqrtai * (yi * self.sqrtai).dot(self.D)
    am = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = am * P / (R * R * T * T)
    B = bm * P / (R * T)
    Z = self.solve_eos(A, B)
    return Z

  def get_lnphii(
    self,
    P: np.floating,
    T: np.floating,
    yi: npt.NDArray[np.floating],
    check_input: bool = True,
  ) -> npt.NDArray[np.floating]:
    """
    Calculates component fugacity coefficients using the van der Waals
    equation of state.

    Arguments:
      P: pressure [Pa]. Must be a scalar of numpy.floating.
      T: temperature [K]. Must be a scalar of numpy.floating.
      yi: mole fraction for each component [fr.].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.

    Returns:
      A numpy array containing the natural logarithm of fugacity
      coefficients for each component.
    """
    if check_input:
      check_PTyi(P, T, yi, self.Nc)
    Si = self.sqrtai * (yi * self.sqrtai).dot(self.D)
    am = yi.dot(Si)
    bm = yi.dot(self.bi)
    A = am * P / R / R / T / T
    B = bm * P / R / T
    Z = self.solve_eos(A, B)
    return -np.log(Z - B) + B / bm / (Z - B) * self.bi - 2. * A / Z / am * Si

  def get_lnphii_Z(
    self,
    P: np.floating,
    T: np.floating,
    yi: npt.NDArray[np.floating],
    check_input: bool = True,
  ) -> tuple[npt.NDArray[np.floating], np.floating]:
    """
    Calculates component fugacity coefficients using the van der Waals
    equation of state.

    Arguments:
      P: pressure [Pa]. Must be a scalar of numpy.floating.
      T: temperature [K]. Must be a scalar of numpy.floating.
      yi: mole fraction for each component [fr.].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.

    Returns:
      A tuple containing a numpy array with the natural logarithm of
      fugacity coefficients for each component, and the compressibility
      factor of a mixture.
    """
    if check_input:
      check_PTyi(P, T, yi, self.Nc)
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
    P: np.floating,
    T: np.floating,
    yi: npt.NDArray[np.floating],
    check_input: bool = True,
  ) -> npt.NDArray[np.floating]:
    """
    Calculates component fugacities using the van der Waals
    equation of state.

    Arguments:
      P: pressure [Pa]. Must be a scalar of numpy.floating.
      T: temperature [K]. Must be a scalar of numpy.floating.
      yi: mole fraction for each component [fr.].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.

    Returns:
      A numpy array containing the natural logarithm of fugacities
      for each component.
    """
    if check_input:
      check_PTyi(P, T, yi, self.Nc)
    return self.get_lnphii(P, T, yi, False) + np.log(P * yi)

  @staticmethod
  def solve_eos(
    A: np.floating,
    B: np.floating,
  ) -> np.floating:
    """
    Solves the Van-der-Waals equation of state using the Cardano's
    method, and selectes an appropriate root based on the comparison
    of the Gibbs energy difference.

    Arguments:
      A: the coefficient in the cubic form of the Van-der-Waals
        equation of state. Must be a scalar of numpy.floating.
      B: the coefficient in the cubic form of the Van-der-Waals
        equation of state. Must be a scalar of numpy.floating.

    Returns:
      The solution of the Van-der-Waals equation of state.
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
  def __init__(
    self,
    Pci: npt.NDArray[np.floating],
    Tci: npt.NDArray[np.floating],
    wi: npt.NDArray[np.floating],
    vsi: npt.NDArray[np.floating],
    dij: npt.NDArray[np.floating],
  ) -> None:
    """
    Allows to calculate the fugacities of components and
    the compressibility factor of a mixture of `Nc` components
    using the Peng-Robinson (1978) equation of state.

    Arguments:
      Pci: critical pressures of components [Pa].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.
      Tci: critical temperatures of components [K].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.
      wi: acentric factors of components.
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.
      vsi: volume shift parameters of components.
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.
      dij: the lower triangle of the binary interaction coefficients
        matrix expressed as a numpy.ndarray[numpy.floating] with
        the shape of `(Nc * (Nc - 1) // 2,)`.
    """
    args = (Pci, Tci, wi, vsi, dij)
    if not all(map(lambda a: isinstance(a, np.ndarray), args)):
      raise TypeError(
        "Types of all input arrays must be numpy.ndarray, but"
        f"\n\t{type(Pci)=},"
        f"\n\t{type(Tci)=},"
        f"\n\t{type(wi)=},"
        f"\n\t{type(vsi)=},"
        f"\n\t{type(dij)=}."
      )
    if not all(map(lambda a: np.issubdtype(a.dtype, np.floating), args)):
      raise TypeError(
        "Data types of all input arrays must be numpy.floating, but"
        f"\n\t{Pci.dtype=},"
        f"\n\t{Tci.dtype=},"
        f"\n\t{wi.dtype=},"
        f"\n\t{vsi.dtype=},"
        f"\n\t{dij.dtype=}."
      )
    if not all(map(lambda a: a.ndim == 1, args)):
      raise ValueError(
        "Dimensions of all input arrays must be equal 1, but"
        f"\n\t{Pci.ndim=},"
        f"\n\t{Tci.ndim=},"
        f"\n\t{wi.ndim=},"
        f"\n\t{vsi.ndim=},"
        f"\n\t{dij.ndim=}."
      )
    self.Nc = Pci.shape[0]
    if not all(map(lambda a: a.shape == (self.Nc,), args[:-1])):
      raise ValueError(
        f"The shape of Pci, Tci, wi and vsi arrays must be ({self.Nc},), but"
        f"\n\t{Pci.shape=},"
        f"\n\t{Tci.shape=},"
        f"\n\t{wi.shape=},"
        f"\n\t{vsi.shape=}."
      )
    if dij.shape != (self.Nc * (self.Nc - 1) // 2,):
      raise ValueError(
        f"The shape of the dij array must be ({self.Nc*(self.Nc-1)//2,}), but"
        f"\n\t{dij.shape=}."
      )
    if not np.isfinite(Pci).all():
      where = np.where(1 - np.isfinite(Pci))
      raise ValueError(
        "All input arrays must contain finite values, but"
        f"\n\tPci have values: {Pci[where]}\n\tat indices: {where}."
      )
    if not np.isfinite(Tci).all():
      where = np.where(1 - np.isfinite(Tci))
      raise ValueError(
        "All input arrays must contain finite values, but"
        f"\n\tTci have values: {Tci[where]}\n\tat indices: {where}."
      )
    if not np.isfinite(wi).all():
      where = np.where(1 - np.isfinite(wi))
      raise ValueError(
        "All input arrays must contain finite values, but"
        f"\n\twi have values: {wi[where]}\n\tat indices: {where}."
      )
    if not np.isfinite(vsi).all():
      where = np.where(1 - np.isfinite(vsi))
      raise ValueError(
        "All input arrays must contain finite values, but"
        f"vsi have values: {vsi[where]}\n\tat indices: {where}."
      )
    if not np.isfinite(dij).all():
      where = np.where(1 - np.isfinite(dij))
      raise ValueError(
        "All input arrays must contain finite values, but"
        f"dij have values: {dij[where]}\n\tat indices: {where}."
      )
    if (Pci <= 0.).any():
      raise ValueError(
        "Critical pressures of all components must be greater than zero."
      )
    if (Tci <= 0.).any():
      raise ValueError(
        "Critical temperatures of all components must be greater than zero."
      )
    if (wi <= 0.).any():
      raise ValueError(
        "Acentric factors of all components must be greater than zero."
      )
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
    P: np.floating,
    T: np.floating,
    yi: npt.NDArray[np.floating],
    check_input: bool = True,
  ) -> np.floating:
    """
    Calculates the compressibility factor using
    the Peng-Robinson equation of state (1978).

    Arguments:
      P: pressure [Pa]. Must be a scalar of numpy.floating.
      T: temperature [K]. Must be a scalar of numpy.floating.
      yi: mole fraction for each component [fr.].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.

    Returns:
      The compressibility factor.
    """
    if check_input:
      check_PTyi(P, T, yi, self.Nc)
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
    P: np.floating,
    T: np.floating,
    yi: npt.NDArray[np.floating],
    check_input: bool = True,
  ) -> npt.NDArray[np.floating]:
    """
    Calculates fugacity coefficients of components using
    the Peng-Robinson equation of state.

    Arguments:
      P: pressure [Pa]. Must be a scalar of numpy.floating.
      T: temperature [K]. Must be a scalar of numpy.floating.
      yi: mole fraction for each component [fr.].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.

    Returns:
      A numpy array containing the natural logarithm of fugacity
      coefficients for each component.
    """
    if check_input:
      check_PTyi(P, T, yi, self.Nc)
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
    P: np.floating,
    T: np.floating,
    yi: npt.NDArray[np.floating],
    check_input: bool = True,
  ) -> tuple[npt.NDArray[np.floating], np.floating]:
    """
    Calculates fugacity coefficients of components and
    the compressibility factor using the Peng-Robinson
    equation of state.

    Arguments:
      P: pressure [Pa]. Must be a scalar of numpy.floating.
      T: temperature [K]. Must be a scalar of numpy.floating.
      yi: mole fraction for each component [fr.].
        Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.

    Returns:
      A tuple containing a numpy array with the natural logarithm of
      fugacity coefficients for each component, and the compressibility
      factor of a mixture.
    """
    if check_input:
      check_PTyi(P, T, yi, self.Nc)
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

  def get_lnphiji_Zj(
    self,
    P: np.floating,
    T: np.floating,
    yji: npt.NDArray[np.floating],
    check_input: bool = True,
  ) -> tuple[npt.NDArray[np.floating], np.floating]:
    """
    Calculates fugacity coefficients of components and
    the compressibility factor for each given component composition
    using the Peng-Robinson equation of state.

    Arguments:
      P: pressure [Pa]. Must be a scalar of numpy.floating.
      T: temperature [K]. Must be a scalar of numpy.floating.
      yji: mole fraction for each component [fr.].
        Must be a numpy.ndarray[numpy.floating] with the shape
        equals to `(Np, Nc)`, where Np is a number of phases and
        Nc is a number of components.

    Returns:
      A tuple containing a numpy array of the natural logarithm of
      fugacity coefficients for each component in each phase, and
      a numpy array of compressibility factors for each phase.
    """
    # if check_input:
    #   check_PTyi(P, T, yji, self.Nc)
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
    P: np.floating,
    T: np.floating,
    yi: npt.NDArray[np.floating],
    check_input: bool = True,
  ) -> npt.NDArray[np.floating]:
    """
    Calculates fugacities of components using the Peng-Robinson
    equation of state.

    Arguments:
      P: pressure [Pa]. Must be a scalar of numpy.floating.
      T: temperature [K]. Must be a scalar of numpy.floating.
      yi: mole fraction for each component [fr.].
      Must be a numpy.ndarray[numpy.floating] with the shape `(Nc,)`.

    Returns:
      A numpy array containing natural logarithms of fugacities
      of each component.
    """
    if check_input:
      check_PTyi(P, T, yi, self.Nc)
    return self.get_lnphii(P, T, yi, False) + np.log(P * yi)

  @staticmethod
  def solve_eos(
    A: np.floating,
    B: np.floating,
  ) -> np.floating:
    """
    Solves the Peng-Robinson equation of state using the Cardano's
    method, and selectes an appropriate root based on the comparison
    of the Gibbs energy difference.

    Arguments:
      A: the coefficient in the cubic form of the Peng-Robinson
        equation of state. Must be a scalar of numpy.floating.
      B: the coefficient in the cubic form of the Peng-Robinson
        equation of state. Must be a scalar of numpy.floating.

    Returns:
      The solution of the Peng-Robinson equation of state.
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



