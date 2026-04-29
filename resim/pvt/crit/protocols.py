from typing import (
  Literal,
  Protocol,
)

from resim.pvt.datatypes import (
  Float,
  Matrix,
  OnePhaseState,
  State,
  Vector,
)

from resim.pvt.eos import (
  Eos,
  State2pVTEos,
)


class SpinVTEos(Eos, Protocol):
  """A protocol of an initialized instance of a VT-based equation of
  state that can be used to calculate the spinodal temperature of a
  mixture. It must have the following attributes:
  +-----------+------+-------------------------------------------------+
  | Attribute | Type | Description                                     |
  +===========+======+=================================================+
  | name      | str  | The name of an EOS (for logging).               |
  +-----------+------+-------------------------------------------------+
  | Nc        | int  | The number of components in a system.           |
  +-----------+------+-------------------------------------------------+

  Any class that implements this protocol must also have methods:
  +----------------+---------------------------------------------------+
  | Method         | Result                                            |
  +================+===================================================+
  | getVT_lnfi_dnj | A vector of shape `(Nc,)` of natural logarithms   |
  |                | of fugacities of components and a matrix of shape |
  |                | `(Nc, Nc)` of their partial derivatives with      |
  |                | respect to mole numbers of components.            |
  +----------------+---------------------------------------------------+
  """
  def getVT_lnfi_dnj(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
    n: float,
  ) -> tuple[Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacities of components and their
    partial derivatives with respect to mole numbers of components.

    Parameters
    ----------
    v: float
      Molar volume [m³/mol].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a mixture [mol].

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacities of components,
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacities with respect to mole numbers of
      components.
    """
    pass


class CritSolverVTEos(SpinVTEos, Protocol):
  """A protocol for an initialized instance of an equation of state
  that can be used to solve the critical state problem formulated
  for the VT-thermodynamics. It must have the following attributes:
  +-----------+------+-------------------------------------------------+
  | Attribute | Type | Description                                     |
  +===========+======+=================================================+
  | name      | str  | The name of an EOS (for logging).               |
  +-----------+------+-------------------------------------------------+
  | Nc        | int  | The number of components in a system.           |
  +-----------+------+-------------------------------------------------+

  Any class that implements this protocol must also have methods:
  +----------------+---------------------------------------------------+
  | Method         | Result                                            |
  +================+===================================================+
  | getVT_lnfi_dnj | A vector of shape `(Nc,)` of natural logarithms   |
  |                | of fugacities of components and a matrix of shape |
  |                | `(Nc, Nc)` of their partial derivatives with      |
  |                | respect to mole numbers of components.            |
  +----------------+---------------------------------------------------+
  | getVT_d3F      | The cubic form of the Helmholtz energy Taylor     |
  |                | series decomposition.                             |
  +----------------+---------------------------------------------------+
  | getVT_vmin     | The minimum molar volume [m³/mol].                |
  +----------------+---------------------------------------------------+
  """
  def getVT_d3F(
    self,
    v: float,
    T: float,
    yi: Vector[Float],
    zti: Vector[Float],
    n: float,
  ) -> float:
    """Compute the cubic form of the Helmholtz energy Taylor series
    decomposition for a given vector of component mole number changes.

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
      Mole number of a mixture [mol].

    Returns
    -------
    The cubic form of the Helmholtz energy Taylor series decomposition.
    """
    pass

  def getVT_vmin(
    self,
    T: float,
    yi: Vector[Float],
  ) -> float:
    """Calculate the minimum molar volume.

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
    pass


class CritVTEos(CritSolverVTEos, State2pVTEos, Protocol):
  """A protocol of an initialized instance of a VT-based equation of
  state that can be used to calculate the critical state of a mixture.
  It must have the following attributes:
  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | form      | Literal['VT'] | The formalism of an EOS.               |
  +-----------+---------------+----------------------------------------+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components in a system.  |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molecular weights of components        |
  |           |               | [kg/mol] as a vector of shape `(Nc,)`. |
  +-----------+---------------+----------------------------------------+
  | Tci       | Vector[Float] | Critical temperatures of components    |
  |           |               | [K] as a vector of shape `(Nc,)`.      |
  +-----------+---------------+----------------------------------------+
  | vci       | Vector[Float] | Critical molar volumes of components   |
  |           |               | [m³/mol] as a vector of shape `(Nc,)`. |
  +-----------+---------------+----------------------------------------+

  Any class that implements this protocol must also have methods:
  +------------------+-------------------------------------------------+
  | Method           | Result                                          |
  +==================+=================================================+
  | getVT_lnfi_dnj   | A vector of shape `(Nc,)` of natural logarithms |
  |                  | of fugacities of components and a matrix of     |
  |                  | shape `(Nc, Nc)` of their partial derivatives   |
  |                  | with respect to mole numbers of components.     |
  +------------------+-------------------------------------------------+
  | getVT_d3F        | The cubic form of the Helmholtz energy Taylor   |
  |                  | series decomposition.                           |
  +------------------+-------------------------------------------------+
  | getVT_vmin       | The minimum molar volume [m³/mol].              |
  +------------------+-------------------------------------------------+
  | getVT_P          | Pressure [Pa].                                  |
  +------------------+-------------------------------------------------+
  | getVT_PID        | The phase designation index of a mixture.       |
  +------------------+-------------------------------------------------+
  """
  form: Literal['VT']
  Tci: Vector[Float]
  vci: Vector[Float]


class SpinSolver[Eos](Protocol):
  """A protocol for callable objects that can be used to find a spinodal
  temperature for the VT-thermodynamics.

  Parameters
  ----------
  eos: CritVTEos
    An initialized instance of an EOS of the type `CritVTEos`.

  V: float
    Volume [m³].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  T0: float
    An initial guess of temperature [K].

  zetai0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  Returns
  -------
  A tuple containing:
  - spinodal temperature [K],
  - an eigenvector as a `Vector[Float]` of shape `(Nc,)`.
  """
  def __call__(
    self,
    eos: Eos,
    V: float,
    yi: Vector[Float],
    T0: float,
    zetai0: Vector[Float],
    /,
  ) -> tuple[float, float, Vector[Float]]:
    pass


class CritSolver[Eos](Protocol):
  """A protocol for callable objects that can solve the critical state
  problems.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  P1: float
    An initial guess for the first thermodynamic parameter in SI units.

  P2: float
    An initial guess for the second thermodynamic parameter in SI units.

  Returns
  -------
  A tuple containing two thermodynamic paramters that characterize the
  critical state of a mixture.
  """
  def __call__(
    self,
    eos: Eos,
    yi: Vector[Float],
    P1: float,
    P2: float,
    /,
  ) -> tuple[float, float]:
    pass


class CritRoutine[Eos](Protocol):
  """A protocol for callable objects that can perform the critical state
  calculation procedure for a given mole composition of a mixture.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    The mole number of a mixture [mol].

  init: tuple[float, float] | State | None
    This parameter is used to initialize a critical state calculation
    procedure. The detailed explanation of the logic behind possible
    types of `init` is given in the following table:

    +---------------------+--------------------------------------------+
    | Type                | Description                                |
    +=====================+============================================+
    | tuple[float, float] | Initial guesses for two parameters in SI   |
    |                     | units.                                     |
    +---------------------+--------------------------------------------+
    | State               | A state of a mixture, parameters of which  |
    |                     | can be used to initialize a critical state |
    |                     | calculation procedure.                     |
    +---------------------+--------------------------------------------+
    | None                | An internal routine should be used to      |
    |                     | obtain initial guesses.                    |
    +---------------------+--------------------------------------------+

  Returns
  -------
  The critical state of a mixture.
  """
  def __call__(
    self,
    eos: Eos,
    P1: float,
    P2: float,
    yi: Vector[Float],
    n: float,
    init: tuple[float, float] | State | None,
    /,
  ) -> OnePhaseState:
    pass
