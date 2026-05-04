from typing import (
  Iterable,
  Literal,
  Protocol,
)

from resim.pvt.datatypes import (
  Envelope,
  Float,
  Integer,
  Matrix,
  State,
  Tensor,
  Vector,
)

from resim.pvt.eos import (
  Eos,
  StateNpPTEos,
)

from resim.pvt.psat import (
  PsatSolverPTEos,
)


class Env2pSolverPTEos(Eos, Protocol):
  """A protocol for an initialized instance of a PT-based equation
  of state (PTEos) that can be used to solve the two-phase envelope
  problem for a mixture. It must have the following attributes:
  +-----------+------+-------------------------------------------------+
  | Attribute | Type | Description                                     |
  +===========+======+=================================================+
  | name      | str  | The name of an EOS (for logging).               |
  +-----------+------+-------------------------------------------------+
  | Nc        | int  | The number of components in a system.           |
  +-----------+------+-------------------------------------------------+

  Any class that implements this protocol must also have methods:
  +------------------------+-------------------------------------------+
  | Method                 | Result                                    |
  +========================+===========================================+
  | getPT_lnphii_dP_dT_dyj | - A vector of shape `(Nc,)` of logarithms |
  |                        |   of fugacity coefficients of components. |
  |                        | - A vector of the same shape of their     |
  |                        |   partial derivatives with respect to     |
  |                        |   pressure.                               |
  |                        | - A vector of the same shape of their     |
  |                        |   partial derivatives with respect to     |
  |                        |   temperature.                            |
  |                        | - A matrix of shape `(Nc, Nc)` of their   |
  |                        |   partial derivatives with respect to     |
  |                        |   mole fractions of components.           |
  +------------------------+-------------------------------------------+
  """
  def getPT_lnphii_dP_dT_dyj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to pressure,
    temperature, and mole fractions of components.

    The mole fraction constraint should not be taken into account.

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
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to temperature,
    - a `Matrix[Float]` of shape `(Nc, Nc)` partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to mole fractions of components.
    """
    pass


class EnvNpSolverPTEos(Env2pSolverPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation
  of state (PTEos) that can be used to solve the multiphase envelope
  problem for a mixture. It must have the following attributes:
  +-----------+------+-------------------------------------------------+
  | Attribute | Type | Description                                     |
  +===========+======+=================================================+
  | name      | str  | The name of an EOS (for logging).               |
  +-----------+------+-------------------------------------------------+
  | Nc        | int  | The number of components in a system.           |
  +-----------+------+-------------------------------------------------+

  Any class that implements this protocol must also have methods:
  +-------------------------+------------------------------------------+
  | Method                  | Result                                   |
  +=========================+==========================================+
  | getPT_lnphii_dP_dT_dyj  | - A vector of shape `(Nc,)` of natural   |
  |                         |   logarithms of fugacity coefficients of |
  |                         |   components.                            |
  |                         | - A vector of the same shape of their    |
  |                         |   partial derivatives with respect to    |
  |                         |   pressure.                              |
  |                         | - A vector of the same shape of their    |
  |                         |   partial derivatives with respect to    |
  |                         |   temperature.                           |
  |                         | - A matrix of shape `(Nc, Nc)` of their  |
  |                         |   partial derivatives with respect to    |
  |                         |   mole fractions of components.          |
  +-------------------------+------------------------------------------+
  | getPT_lnphiji_dP_dT_dyk | - A matrix of shape `(Np, Nc)` of        |
  |                         |   logarithms of fugacity coefficients of |
  |                         |   componentsfor each mixture.            |
  |                         | - A matrix of the same shape of their    |
  |                         |   partial derivatives with respect to    |
  |                         |   pressure.                              |
  |                         | - A matrix of the same shape of their    |
  |                         |   partial derivatives with respect to    |
  |                         |   temperature.                           |
  |                         | - A tensor of shape `(Np, Nc, Nc)` of    |
  |                         |   their partial derivatives with respect |
  |                         |   to mole fractions of components.       |
  +-------------------------+------------------------------------------+
  """
  def getPT_lnphiji_dP_dT_dyk(
    self,
    P: float,
    T: float,
    yji: Matrix[Float],
  ) -> tuple[Matrix[Float], Matrix[Float], Matrix[Float], Tensor[Float]]:
    """Compute natural logarithms of fugacity coefficients of components
    and their partial derivatives with respect to pressure, tenperature,
    and component mole fractions for each mixture.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yji: Matrix[Float], shape (Np, Nc)
      Mole fractions of `Nc` components for each of `Np` mixtures.

    Returns
    -------
    A tuple containing:
    - a `Matrix[Float]` of shape `(Np, Nc)` of natural logarithms of
      fugacity coefficients of components in mixtures,
    - a `Matrix[Float]` of shape `(Np, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure in mixtures,
    - a `Matrix[Float]` of shape `(Np, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to temperature in mixtures,
    - a `Tensor[Float]` of shape `(Np, Nc, Nc)` of partial derivatives
      of natural logarithms of fugacity coefficients of components with
      respect to mole fractions of components in mixtures.
    """
    pass


class Env2pPTEos(PsatSolverPTEos, Env2pSolverPTEos, StateNpPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation
  of state that can be used to construct the two-phase envelope of
  a mixture. It must have the following attributes:
  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | form      | Literal['PT'] | The formalism of an EOS.               |
  +-----------+---------------+----------------------------------------+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components in a system.  |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molecular weights of components        |
  |           |               | [kg/mol] as a vector of shape `(Nc,)`. |
  +-----------+---------------+----------------------------------------+

  Any class that implements this protocol must also have methods:
  +------------------------+-------------------------------------------+
  | Method                 | Result                                    |
  +========================+===========================================+
  | getPT_kvguess          | A sequence of initial guesses of          |
  |                        | k-values.                                 |
  +------------------------+-------------------------------------------+
  | getPT_lnphii           | A vector of shape `(Nc,)` of logarithms   |
  |                        | of fugacity coefficients of components.   |
  +------------------------+-------------------------------------------+
  | getPT_lnphii_dnj       | Previous + a matrix of shape `(Nc, Nc)`   |
  |                        | of their partial derivatives with respect |
  |                        | to mole numbers of components.            |
  +------------------------+-------------------------------------------+
  | getPT_lnphii_dP        | A vector of shape `(Nc,)` of logarithms   |
  |                        | of fugacity coefficients of components    |
  |                        | and a vector of the same shape of their   |
  |                        | partial derivatives with respect to       |
  |                        | pressure.                                 |
  +------------------------+-------------------------------------------+
  | getPT_lnphii_dP_dnj    | Previous + a matrix of shape `(Nc, Nc)`   |
  |                        | of their partial derivatives with respect |
  |                        | to mole numbers of components.            |
  +------------------------+-------------------------------------------+
  | getPT_lnphii_dP_dT_dyj | - A vector of shape `(Nc,)` of logarithms |
  |                        |   of fugacity coefficients of components. |
  |                        | - A vector of the same shape of their     |
  |                        |   partial derivatives with respect to     |
  |                        |   pressure.                               |
  |                        | - A vector of the same shape of their     |
  |                        |   partial derivatives with respect to     |
  |                        |   temperature.                            |
  |                        | - A matrix of shape `(Nc, Nc)` of their   |
  |                        |   partial derivatives with respect to     |
  |                        |   mole fractions of components.           |
  +------------------------+-------------------------------------------+
  | get_PIDj               | A vector of shape `(Np,)` of phase        |
  |                        | designation indices. `Np` is the number   |
  |                        | of phases.                                |
  +------------------------+-------------------------------------------+
  | get_Zj                 | A vector of shape `(Np,)` of              |
  |                        | compressibility factors.                  |
  +------------------------+-------------------------------------------+
  """
  form: Literal['PT']

  def getPT_kvguess(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> Iterable[Vector[Float]]:
    """Create an iterable object of initial guesses of k-values.

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
    An iterable object of vectors (with the shape `(Nc,)`) of initial
    guesses of k-values.
    """
    pass


class Env2pSolver[Eos](Protocol):
  """A protocol for callable objects that can be used to solve the two-
  phase envelope problem for a mixture.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  x0: Vector[Float], shape (Nc + 2,)
    An initial guess for the basic variables of the two-phase envelope
    problem.

  sidx: int | Integer
    An index of the fixed (known) variable.

  sval: float
    A value of the fixed (known) variable.

  phf: float
    Mole fraction of the non-reference phase for which the phase
    envelope should be constructed.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  Returns
  -------
  A tuple that containing:
  - the solution of the phase envelope equations as a `Vector[Float]`
    of shape `(Nc + 2,)`,
  - jacobian (at the solution) as a `Matrix[Float]` of the shape
    `(Nc + 2, Nc + 2)`,
  - the number of iterations to converge.
  """
  def __call__(
    self,
    eos: Eos,
    x0: Vector[Float],
    sidx: int | Integer,
    sval: float,
    phfr: float,
    yi: Vector[Float],
    /,
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float], Matrix[Float], int]:
    pass


class Env2pRoutine[Eos](Protocol):
  """A protocol for callable objects that can be used to construct the
  two-phase envelope of a mixture.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  phf: float
    Mole fraction of the non-reference phase for which the phase
    envelope should be constructed.

  init: tuple[float, float, Vector[Float]] | State | float
    This parameter is used to initialize the phase envelope routine.
    The detailed explanation of the logic behind different types of
    `init` is given in the following table:

    +------------------+-----------------------------------------------+
    | Type             | Description                                   |
    +==================+===============================================+
    | tuple[           | A tuple containing an initial guess of:       |
    |   float,         | - presure [Pa],                               |
    |   float,         | - temperature [K],                            |
    |   Vector[Float], | - k-values of components as a vector of shape |
    | ]                | `(Nc,)`.                                      |
    +------------------+-----------------------------------------------+
    | State            | A state of a mixture, k-values, pressure,     |
    |                  | and temperature of which can be used as an    |
    |                  | initial guess for calculation of the first    |
    |                  | point of the phase envelope.                  |
    +------------------+-----------------------------------------------+
    | float            | A temperature [K], from which the phase       |
    |                  | envelope calculation procedure starts. An     |
    |                  | internal initialization procedure should be   |
    |                  | used to obtain initial pressure and k-values. |
    +------------------+-----------------------------------------------+

  Returns
  -------
  The two-phase envelope as an instance of `Envelope`.
  """
  def run(
    self,
    eos: Eos,
    yi: Vector[Float],
    phf: float,
    init: tuple[float, float, Vector[Float]] | State | float,
    /,
  ) -> Envelope:
    pass
