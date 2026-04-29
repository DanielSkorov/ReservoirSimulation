from typing import (
  Protocol,
)

from resim.pvt.datatypes import (
  Integer,
  Float,
  Vector,
  Matrix,
)


class Eos(Protocol):
  """Any instance of any equation of state (Eos) must have the following
  attribures:

  +-----------+------+-------------------------------------------------+
  | Attribute | Type | Description                                     |
  +===========+======+=================================================+
  | name      | str  | The name of an EOS (for logging).               |
  +-----------+------+-------------------------------------------------+
  | Nc        | int  | The number of components in a mixture.          |
  +-----------+------+-------------------------------------------------+
  """
  name: str
  Nc: int


class State2pPTEos(Eos, Protocol):
  """A protocol for an initialized instance of a PT-based equation
  of state that can be used to characterize one- and two-phase
  thermodynamic states of mixtures. It must have the following
  attributes:

  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components in a mixture. |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molecular weights of components        |
  |           |               | [kg/mol] as a vector of shape `(Nc,)`. |
  +-----------+---------------+----------------------------------------+

  Any class that implements this protocol must also have methods:

  +-----------+--------------------------------------------------------+
  | Method    | Result                                                 |
  +===========+========================================================+
  | getPT_PID | The phase designation index of a mixture.              |
  +-----------+--------------------------------------------------------+
  | getPT_Z   | The compressibility factor of a mixture.               |
  +-----------+--------------------------------------------------------+
  """
  mwi: Vector[Float]

  def getPT_PID(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> int:
    """This method is used to obtain the phase designation index of
    a mixture (`0` = vapour, `1` = liquid, etc).

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
    The phase designation index of a mixture.
    """
    pass

  def getPT_Z(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> float:
    """This method is used to calculate the compressibility factor of
    a mixture.

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
    The compressibility factor of a mixture.
    """
    pass


class StateNpPTEos(State2pPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state that can be used to characterize a multiphase thermodynamic
  state of a mixture. It must have the following attributes:

  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components in a mixture. |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molecular weights of components        |
  |           |               | [kg/mol] as a vector of shape `(Nc,)`. |
  +-----------+---------------+----------------------------------------+

  Any class that implements this protocol must also have methods:

  +------------+-------------------------------------------------------+
  | Method     | Result                                                |
  +============+=======================================================+
  | getPT_PID  | The phase designation index of a mixture.             |
  +------------+-------------------------------------------------------+
  | getPT_Z    | The compressibility factor of a mixture.              |
  +------------+-------------------------------------------------------+
  | getPT_PIDj | A vector of shape `(Np,)` of phase designation        |
  |            | indices. `Np` is the number of phases.                |
  +------------+-------------------------------------------------------+
  | getPT_Zj   | A vector of shape `(Np,)` of compressibility factors. |
  +------------+-------------------------------------------------------+
  """
  def getPT_PIDj(
    self,
    P: float | Vector[Float],
    T: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
  ) -> Vector[Integer]:
    """This method is used to obtain the designation index for
    each phase (`0` = vapour, `1` = liquid, etc).

    Parameters
    ----------
    P: float | Vector[Float], shape (Np,)
      Pressure or a vector of pressures [Pa].

    T: float | Vector[Float], shape (Np,)
      Temperature or a vector of temperatures [K].

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      A vector of mole fractions of `Nc` components or a matrix of mole
      fractions of `Nc` components in each of `Np` phases.

    Returns
    -------
    A `Vector[Integer]` of shape `(Np,)` of phase designation indices.
    """
    pass

  def getPT_Zj(
    self,
    P: float | Vector[Float],
    T: float | Vector[Float],
    yji: Vector[Float] | Matrix[Float],
  ) -> Vector[Float]:
    """This method is used to obtain the compressibility factor for
    each phase.

    Parameters
    ----------
    P: float | Vector[Float], shape (Np,)
      Pressure or a vector of pressures [Pa].

    T: float | Vector[Float], shape (Np,)
      Temperature or a vector of temperatures [K].

    yji: Vector[Float], shape (Nc,) | Matrix[Float], shape (Np, Nc)
      A vector of mole fractions of `Nc` components or a matrix of mole
      fractions of `Nc` components in each of `Np` phases.

    Returns
    -------
    A `Vector[Float]` of shape `(Np,)` of compressibility factors.
    """
    pass


class State2pVTEos(Eos, Protocol):
  """A protocol for an initialized instance of a VT-based equation
  of state that can be used to characterize one- and two-phase
  thermodynamic states of mixtures. It must have the following
  attributes:

  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components in a mixture. |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molecular weights of components        |
  |           |               | [kg/mol] as a vector of shape `(Nc,)`. |
  +-----------+---------------+----------------------------------------+

  Any class that implements this protocol must also have methods:

  +-----------+--------------------------------------------------------+
  | Method    | Result                                                 |
  +===========+========================================================+
  | getVT_P   | The compressibility factor of a mixture.               |
  +-----------+--------------------------------------------------------+
  | getVT_PID | The phase designation index of a mixture.              |
  +-----------+--------------------------------------------------------+
  """
  mwi: Vector[Float]

  def getVT_P(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
  ) -> float:
    """Compute pressure for a given volume, temperature, and composition
    of a mixture.

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
    pass

  def getVT_PID(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> int:
    """This method is used to obtain the phase designation index of
    a mixture (`0` = vapour, `1` = liquid, etc).

    Parameters
    ----------
    V: float
      Volume [m³].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    The phase designation index of a mixture.
    """
    pass
