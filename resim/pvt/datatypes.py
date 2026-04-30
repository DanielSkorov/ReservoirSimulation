from __future__ import (
  annotations,
)

from typing import (
  Protocol,
  Self,
  TypeAlias,
  TypeVar,
)

from dataclasses import (
  dataclass,
  field,
)

from numpy import (
  argsort as np_argsort,
  array2string as np_array2string,
  bool as np_bool,
  dtype as np_dtype,
  float64 as np_float64,
  generic as np_generic,
  integer as np_integer,
  ndarray as np_ndarray,
)


Logical = np_bool

Integer = np_integer

Float = np_float64


T = TypeVar('T', bound=np_generic)


Vector: TypeAlias = np_ndarray[tuple[int], np_dtype[T]]

Matrix: TypeAlias = np_ndarray[tuple[int, int], np_dtype[T]]

Tensor: TypeAlias = np_ndarray[tuple[int, int, int], np_dtype[T]]

Array: TypeAlias = np_ndarray[tuple[int, ...], np_dtype[T]]


def arr2str(arr: Array[Float], prec: int) -> str:
  return np_array2string(arr, precision=prec, floatmode='maxprec_equal')


@dataclass(eq=False, slots=True)
class State(object):
  """A dataclass to store parameters (properties) that characterize
  a thermodynamic state of a mixture.

  Attributes
  ----------
  Nc: int
    The number of components.

  Np: int
    The number of phases.

  P: float
    Pressure [Pa].

  T: float
    Temperature [K]

  V: float
    Volume [m³].

  n: float
    Number of moles of a mixture [mol].

  ni: Vector[Float], shape (Nc,)
    Mole numbers of `Nc` components [mol].

  nj: Vector[Float], shape (Np,)
    Mole numbers of `Np` phases [mol].

  nji: Matrix[Float], shape (Np, Nc)
    Mole numbers of `Nc` components in each of `Np` phases [mol].

  fj: Vector[Float], shape (Np,)
    Mole fractions of `Np` phases.

  yji: Matrix[Float], shape (Np, Nc)
    Mole fractions of `Nc` components in each of `Np` phases.

  Zj: Vector[Float], shape (Np,)
    Compressibility factors of `Np` phases.

  vj: Vector[Float], shape (Np,)
    Molar volumes of `Np` phases [m³/mol].

  Vj: Vector[Float], shape (Np,)
    Volumes [m³] of `Np` phases.

  sj: Vector[Float], shape (Np,)
    Volume fractions of `Np` phases.

  dj: Vector[Float], shape (Np,)
    Mass densities of `Np` phases [kg/m³].

  pidj: Vector[Integer], shape (Np,)
    Phase designation indices of `Np` phases (`0` = vapour,
    `1` = liquid, etc.).

  kvji: Matrix[Float], shape (Np - 1, Nc) | None
    K-values of `Nc` components in each of `(Np - 1)` non-reference
    phases. If a one-phase state is stable, this attribute is `None`.

  tmpl: str
    A template for the table representation of this dataclass. This
    attribute is valued internally after class initialization.
  """
  Nc: int
  Np: int
  P: float
  T: float
  V: float
  n: float
  ni: Vector[Float]
  nj: Vector[Float]
  nji: Matrix[Float]
  fj: Vector[Float]
  yji: Matrix[Float]
  Zj: Vector[Float]
  vj: Vector[Float]
  Vj: Vector[Float]
  sj: Vector[Float]
  dj: Vector[Float]
  pidj: Vector[Integer]
  kvji: Matrix[Float] | None
  tmpl: str = field(init=False, default='')

  def __post_init__(self):
    Np = self.Np
    self.tmpl = (
      '{0:7.3f}{1:8.2f}{2:9.1e}{3:9.4f}'
      + ''.join(
        [
          f'{{{i}:5d}}{{{i+Np}:9.4f}}{{{i+2*Np}:9.4f}}{{{i+3*Np}:11.5g}}'
          for i in range(4, 4 + Np)
        ]
      )
    )
    pass

  def __str__(self, prefix: str = '', shift: str = '') -> str:
    """Generate the text representation of the dataclass.

    Parameters
    ----------
    prefix: str
      The prefix for the text representation. Default is `''`.

    shift: str
      The shift for each row of the text representation.
      Default is `''`.

    Returns
    -------
    The text representation of the dataclass.
    """
    return prefix + (
      f'{shift}Pressure: {self.P / 1e6:.3f} [MPa].\n'
      f'{shift}Temperature: {self.T - 273.15:.2f} [°C].\n'
      f'{shift}Volume: {self.V:.3e} [m³].\n'
      f'{shift}Mole number: {self.n:.4f} [mol].\n'
      f'{shift}Phase IDs: {self.pidj}.\n'
      f'{shift}Phase mole fractions: {arr2str(self.fj, 4)}.\n'
      f'{shift}Phase volume fractions: {arr2str(self.sj, 4)}.\n'
      f'{shift}Phase mass densities: {arr2str(self.dj, 3)} [kg/m³].'
    )

  def __repr__(
    self,
    phases: int | None = None,
    head_prefix: str = '',
    row_prefix: str = '',
  ) -> str:
    """Generate the table representation of the dataclass.

    Parameters
    ----------
    phases: int | None
      The number of phases for which a table header must be generated.
      If it is `0`, the table header is omitted. If it is `None`, the
      number of phases for the header is equal to the `Np` attribute of
      the dataclass. Default is `None`.

    head_prefix: str
      The prefix for the table header. Default is `''`.

    row_prefix: str
      The prefix for the row of the table. Default is `''`.

    Returns
    -------
    The table representation of the dataclass.
    """
    if phases is None:
      head = (
        head_prefix
        + 'P [MPa]  T [°C]   V [m³]  n [mol]'
        + self.Np * '  PID  f [fr.]  s [fr.]  ρ [kg/m³]'
        + '\n'
      )
    elif phases == 0:
      head = ''
    else:
      head = (
        head_prefix
        + 'P [MPa]  T [°C]   V [m³]  n [mol]'
        + phases * '  PID  f [fr.]  s [fr.]  ρ [kg/m³]'
        + '\n'
      )
    return head + row_prefix + self.tmpl.format(
      self.P / 1e6, self.T - 273.15, self.V, self.n,
      *self.pidj, *self.fj, *self.sj, *self.dj,
    )

  def __format__(self, fmt: str) -> str:
    """Generate a representation of the dataclass for a given format.

    Parameters
    ----------
    fmt: str
      A given format. Available options are listed in the following
      table.

      +---------------+------------------------------------------------+
      | Format        | Description                                    |
      +===============+================================================+
      | `'s'` or `''` | The text representation of the dataclass.      |
      +---------------+------------------------------------------------+
      | `'r'`         | The table representation of the dataclass.     |
      +---------------+------------------------------------------------+

    Returns
    -------
    A representation of the dataclass.

    Raises
    -----
    ValueError
      This exceptioon is raised if an unknown format is passed.
    """
    if not fmt or fmt == 's':
      return self.__str__()
    elif fmt == 'r':
      return self.__repr__()
    else:
      raise ValueError(f'Unsupported format: "{fmt}".')

  def densort(self) -> None:
    """Sort the array-like attributes of the dataclass in ascending
    order based on phase densities.
    """
    if self.kvji is not None:
      idx = np_argsort(self.dj)
      self.nj = self.nj[idx]
      self.nji = self.nji[idx]
      self.fj = self.fj[idx]
      self.yji = self.yji[idx]
      self.Zj = self.Zj[idx]
      self.vj = self.vj[idx]
      self.Vj = self.Vj[idx]
      self.sj = self.sj[idx]
      self.dj = self.dj[idx]
      self.pidj = self.pidj[idx]
      self.kvji = self.yji[:-1] / self.yji[-1]
    pass

  @classmethod
  def from_state(cls, s: State, *args) -> Self:
    """A class method used to create a new instance of the dataclass
    from a given one.

    Parameters
    ----------
    s: State
      A given instance of the dataclass.

    *args
      Other arguments that will be ignored.

    Returns
    -------
    A new instance of the dataclass whose attributes are identical to
    those of a given instance.
    """
    return cls(
      s.Nc, s.Np, s.P, s.T, s.V, s.n, s.ni, s.nj, s.nji, s.fj, s.yji, s.Zj,
      s.vj, s.Vj, s.sj, s.dj, s.pidj, s.kvji,
    )


@dataclass(eq=False, slots=True, repr=False)
class OnePhaseState(State):
  """A container for parameters that characterize a one-phase state
  of any thermodynamic multicomponent system. It inherits attributes
  and methods from `State`, except:

  Attributes
  ----------
  kvji: None
    For any one-phase state of any mixture k-values are `None`.
  """
  kvji: None


@dataclass(eq=False, slots=True, repr=False)
class MultiPhaseState(State):
  """A container for parameters that characterize a multiphase state
  of any thermodynamic multicomponent system. It inherits attributes
  and methods from `State`, except:

  Attributes
  ----------
  kvji: Matrix[Float], shape (Np - 1, Nc)
    K-values of `Nc` components in each of `(Np - 1)` non-reference
    phases.
  """
  kvji: Matrix[Float]


@dataclass(eq=False, slots=True)
class Envelope(object):
  """A dataclass for phase envelope calculation outputs with pretty-
  printing.

  Attributes
  ----------
  Ns: int
    The number of states along a phase envelope.

  Nc: int
    The number of components. It is the same for all states.

  Np: int
    The number of phases. It is the same for all states.

  P: Vector[Float], shape (Ns,)
    Pressure [Pa].

  T: Vector[Float], shape (Ns,)
    Temperature [K]

  V: Vector[Float], shape (Ns,)
    Volume [m³].

  n: Vector[Float], shape (Ns,)
    Number of moles of a mixture [mol].

  ni: Matrix[Float], shape (Ns, Nc)
    Mole numbers of `Nc` components [mol].

  nj: Matrix[Float], shape (Ns, Np)
    Mole numbers of `Np` phases [mol].

  nji: Tensor[Float], shape (Ns, Np, Nc)
    Mole numbers of `Nc` components in each of `Np` phases [mol].

  fj: Matrix[Float], shape (Ns, Np)
    Mole fractions of `Np` phases.

  yji: Tensor[Float], shape (Ns, Np, Nc)
    Mole fractions of `Nc` components in each of `Np` phases.

  Zj: Matrix[Float], shape (Ns, Np)
    Compressibility factors of `Np` phases.

  vj: Matrix[Float], shape (Ns, Np)
    Molar volumes of `Np` phases [m³/mol].

  Vj: Matrix[Float], shape (Ns, Np)
    Volumes of `Np` phases [m³].

  sj: Matrix[Float], shape (Ns, Np)
    Volume fractions of `Np` phases.

  dj: Matrix[Float], shape (Ns, Np)
    Mass densities of `Np` phases [kg/m³].

  pidj: Matrix[Integer], shape (Ns, Np)
    Phase designation indices of `Np` phases (`0` = vapour,
    `1` = liquid, etc.).

  kvji: Tensor[Float], shape (Ns, Np - 1, Nc)
    K-values of `Nc` components in `(Np - 1)` non-reference phases.

  Pc: Vector[Float], shape (Ncp,) | None
    Pressures of critical points [Pa]. This attribute is `None` if no
    critical points were detected for a phase envelope.

  Tc: Vector[Float], shape (Ncp,) | None
    Temperatures of critical points [K]. This attribute is `None` if no
    critical points were detected for a phase envelope.

  Pcb: Vector[Float] | None
    Pressures of cricondenbar points [Pa]. This attribute is `None` if
    no cricondenbar points were detected for a phase envelope.

  Tcb: Vector[Float] | None
    Temperatures of cricondenbar points [K]. This attribute is `None` if
    no cricondenbar points were detected for a phase envelope.

  Pct: Vector[Float] | None
    Pressures of cricondentherm points [Pa]. This attribute is `None` if
    no cricondentherm points were detected for a phase envelope.

  Tct: Vector[Float] | None
    Temperatures of cricondentherm points [K]. This attribute is `None`
    if no cricondentherm points were detected for a phase envelope.

  Methods
  -------
  __str__(self) -> str
    Return the text representation of this class.

  __repr__(self) -> str
    Return the table representation of this class.
  """
  Ns: int
  Nc: int
  Np: int
  P: Vector[Float]
  T: Vector[Float]
  V: Vector[Float]
  n: Vector[Float]
  ni: Matrix[Float]
  nj: Matrix[Float]
  nji: Tensor[Float]
  fj: Matrix[Float]
  yji: Tensor[Float]
  Zj: Matrix[Float]
  vj: Matrix[Float]
  Vj: Vector[Float]
  sj: Matrix[Float]
  dj: Matrix[Float]
  pidj: Matrix[Integer]
  kvji: Tensor[Float]
  Pc: Vector[Float] | None = None
  Tc: Vector[Float] | None = None
  Pcb: Vector[Float] | None = None
  Tcb: Vector[Float] | None = None
  Pct: Vector[Float] | None = None
  Tct: Vector[Float] | None = None

  def __str__(self) -> str:
    out = ''
    if self.Pc is not None and self.Tc is not None:
      out += ('Critical points:\n\t'
              f'{self.Pc / 1e6} [MPa]\n\t'
              f'{self.Tc - 273.15} [°C]\n')
    if self.Pcb is not None and self.Tcb is not None:
      out += ('Cricondenbars:\n\t'
              f'{self.Pcb / 1e6} [MPa]\n\t'
              f'{self.Tcb - 273.15} [°C]\n')
    if self.Pct is not None and self.Tct is not None:
      out += ('Cricondentherms:\n\t'
              f'{self.Pct / 1e6} [MPa]\n\t'
               f'{self.Tct - 273.15} [°C]\n')
    return out

  def __repr__(self) -> str:
    raise NotImplementedError(
      'The table representation of the phase envelope is not implemented yet.'
    )


class ReportState(Protocol):
  """A protocol for dataclasses that can be used to calculate, store,
  and print thermodynamic parameters of a mixture.

  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | Np        | int           | The number of phases.                  |
  +-----------+---------------+----------------------------------------+

  +------------+-------------------------------------------------------+
  | Method     | Result                                                |
  +============+=======================================================+
  | __str__    | The text representation of a dataclass.               |
  +------------+-------------------------------------------------------+
  | __repr__   | The table representation of a dataclass.              |
  +------------+-------------------------------------------------------+
  | __format__ | A representation of a dataclass for a given format.   |
  +------------+-------------------------------------------------------+

  +--------------+-----------------------------------------------------+
  | Class method | Result                                              |
  +==============+=====================================================+
  | from_state   | A new instance of a dataclass obtained from a given |
  |              | state as `State`.                                   |
  +--------------+-----------------------------------------------------+
  """
  Np: int

  def __str__(self, prefix: str = '', shift: str = '') -> str:
    """Generate the text representation of a dataclass.

    Parameters
    ----------
    prefix: str
      The prefix for the text representation. Default is `''`.

    shift: str
      The shift for each row of the text representation.
      Default is `''`.

    Returns
    -------
    The text representation of a dataclass.
    """
    pass

  def __repr__(
    self,
    phases: int | None = None,
    head_prefix: str = '',
    row_prefix: str = '',
  ) -> str:
    """Generate the table representation of a dataclass.

    Parameters
    ----------
    phases: int | None
      The number of phases for which a table header must be generated.
      If it is `0`, the table header is expected to be omitted. If it is
      `None`, the number of phases for the header should be considered
      equal to the `Np` attribute of the dataclass. Default is `None`.

    head_prefix: str
      The prefix for the table header. Default is `''`.

    row_prefix: str
      The prefix for the row of the table. Default is `''`.

    Returns
    -------
    The table representation of a dataclass.
    """
    pass

  def __format__(self, fmt: str) -> str:
    """Generate a representation of a dataclass for a given format.

    Parameters
    ----------
    fmt: str
      A given format.

    Returns
    -------
    A representation of a dataclass.
    """
    pass

  @classmethod
  def from_state(cls, s: State, *args) -> Self:
    """A class method used to create a new dataclass instance from a
    given instance of the `State`.

    Parameters
    ----------
    s: State
      A given instance of the `State` dataclass.

    *args
      Other arguments.

    Returns
    -------
    A new datatclass instance.
    """
    pass


class States[S: ReportState](list[S]):
  """A list of states. It inherits methods from the `builtins.list`
  except the following:

  +------------+-------------------------------------------------------+
  | Method     | Result                                                |
  +============+=======================================================+
  | __str__    | The text representation of the class.                 |
  +------------+-------------------------------------------------------+
  | __repr__   | The table representation of the class.                |
  +------------+-------------------------------------------------------+
  | __format__ | A representation of the class for a given format.     |
  +------------+-------------------------------------------------------+
  """
  def __str__(self, prefix: str = '', shift='') -> str:
    """Generate the text representation of the class.

    Parameters
    ----------
    prefix: str
      The prefix for the text representation. Default is `''`.

    shift: str
      The shift for each row of the text representation.
      Default is `''`.

    Returns
    -------
    The text representation of the class.
    """
    return prefix + '\n'.join([
      s.__str__(f'{shift}Stage #{i}:\n', f'{shift}\t')
      for i, s in enumerate(self)
    ])

  def __repr__(
    self,
    phases: int | None = None,
    head_prefix: str = '',
    row_prefix: str = '',
  ) -> str:
    """Generate the table representation of the class.

    Parameters
    ----------
    phases: int | None
      The number of phases for which a table header must be generated.
      If it is `None`, the number of phases for the header is equal to
      the maximum among all `Np` attributes of all items of the class.
      Default is `None`.

    head_prefix: str
      The prefix for the table header. Default is `''`.

    row_prefix: str
      The prefix for the row of the table. Default is `''`.

    Returns
    -------
    The table representation of the dataclass.
    """
    if phases is None:
      phases = max([s.Np for s in self])
    rows = [
      self[0].__repr__(
        phases, head_prefix + 'Stage  ', row_prefix + f'{0:5d}  ',
      )
    ]
    for i, s in enumerate(self[1:], 1):
      rows.append(s.__repr__(0, '', row_prefix + f'{i:5d}  '))
    return '\n'.join(rows)

  def __format__(self, fmt: str):
    """Generate a representation of the class for a given format.

    Parameters
    ----------
    fmt: str
      A given format. Available options are listed in the following
      table.

      +---------------+------------------------------------------------+
      | Format        | Description                                    |
      +===============+================================================+
      | `'s'` or `''` | The text representation of the class.          |
      +---------------+------------------------------------------------+
      | `'r'`         | The table representation of the class.         |
      +---------------+------------------------------------------------+

    Returns
    -------
    A representation of the class.

    Raises
    -----
    ValueError
      This exceptioon is raised if an unknown format is passed.
    """
    if not fmt or fmt == 's':
      return self.__str__()
    elif fmt == 'r':
      return self.__repr__()
    else:
      raise ValueError(f'Unsupported format: "{fmt}".')


class SepResult[K: str, V: States[State]](dict[K, V]):
  """A container for separator outputs with pretty-printing. It inherits
  methods from `builtins.dict` except the following attributes and
  methods:

  +------------+------+------------------------------------------------+
  | Attribute  | Type | Description                                    |
  +============+======+================================================+
  | _liqbranch | str  | The name of the separator branch where the     |
  |            |      | final state of the liquid phase separation     |
  |            |      | is stored.                                     |
  +------------+------+------------------------------------------------+
  | _liqstage  | int  | The index of a stage in the liquid phase       |
  |            |      | separation branch that contains the final      |
  |            |      | state.                                         |
  +------------+------+------------------------------------------------+
  | _gasbrach  | str  | The name of the separator branch where the     |
  |            |      | final state of the gas phase separation is     |
  |            |      | stored.                                        |
  +------------+------+------------------------------------------------+
  | _gasstage  | int  | The index of a stage in the gas phase          |
  |            |      | separation branch that contains the final      |
  |            |      | state.                                         |
  +------------+------+------------------------------------------------+

  +--------------+-----------------------------------------------------+
  | Method       | Result                                              |
  +==============+=====================================================+
  | __str__      | The text representation of the class.               |
  +--------------+-----------------------------------------------------+
  | __repr__     | The table representation of the class.              |
  +--------------+-----------------------------------------------------+
  | __format__   | A representation of the class for a given format.   |
  +--------------+-----------------------------------------------------+
  | set_liqstate | Mark a specific stage of a given separator branch   |
  |              | as final state of the liquid phase.                 |
  +--------------+-----------------------------------------------------+
  | set_gasstate | Mark a specific stage of a given separator branch   |
  |              | as final state of the gas phase.                    |
  +--------------+-----------------------------------------------------+
  | get_liqstate | Return a state corresponding to the final stage of  |
  |              | the liquid phase separation procedure.              |
  +--------------+-----------------------------------------------------+
  | get_gasstate | Return a state corresponding to the final stage of  |
  |              | the gas phase separation procedure.                 |
  +--------------+-----------------------------------------------------+

  Notes
  -----
  Any separation process can be divided into streams (branches), each
  of which consists of several stages characterized by corresponding
  states of a multicomponent thermodynamic system. Each stream should
  have a name (e.g., `'main'`, `'gas'`, `'oil'`, etc.) and be described
  with a corresponding resultes as an instance of `States`. Therefore,
  any separator can be represented by a dictionary, in which keys are
  stream (branch) names as strings and values are corresponding lists
  of separation stages (states), that the stream is going through.
  """
  _liqbranch: K
  _liqstage: int
  _gasbrach: K
  _gasstage: int

  def __str__(self, prefix: str = '', shift: str = '') -> str:
    """Generate the text representation of the class.

    Parameters
    ----------
    prefix: str
      The prefix for the text representation. Default is `''`.

    shift: str
      The shift for each row of the text representation.
      Default is `''`.

    Returns
    -------
    The text representation of the class.
    """
    return prefix + '\n'.join([
      self[b].__str__(f'{shift}Branch "{b}":\n', f'{shift}\t') for b in self
    ])

  def __repr__(
    self,
    phases: int | None = None,
    head_prefix: str = '',
    row_prefix: str = '',
  ) -> str:
    """Generate the table representation of the class.

    Parameters
    ----------
    phases: int | None
      The number of phases for which a table header must be generated.
      If it is `None`, the number of phases for the header is equal to
      the maximum among all `Np` attributes of all items in each branch
      of the class. Default is `None`.

    head_prefix: str
      The prefix for the table header. Default is `''`.

    row_prefix: str
      The prefix for the row of the table. Default is `''`.

    Returns
    -------
    The table representation of the dataclass.
    """
    branches = list(self.keys())
    if phases is None:
      phases = max([s.Np for b in branches for s in self[b]])
    b = branches[0]
    rows = [
      self[b].__repr__(
        phases, head_prefix + 'Branch  ', row_prefix + f'{b:>6s}  ',
      )
    ]
    for b in branches[1:]:
      rows.append(self[b].__repr__(0, '', row_prefix + f'{b:>6s}  '))
    return '\n'.join(rows)

  def __format__(self, fmt: str) -> str:
    """Generate a representation of the class for a given format.

    Parameters
    ----------
    fmt: str
      A given format. Available options are listed in the following
      table.

      +---------------+------------------------------------------------+
      | Format        | Description                                    |
      +===============+================================================+
      | `'s'` or `''` | The text representation of the class.          |
      +---------------+------------------------------------------------+
      | `'r'`         | The table representation of the class.         |
      +---------------+------------------------------------------------+

    Returns
    -------
    A representation of the class.

    Raises
    -----
    ValueError
      This exceptioon is raised if an unknown format is passed.
    """
    if not fmt or fmt == 's':
      return self.__str__()
    elif fmt == 'r':
      return self.__repr__()
    else:
      raise ValueError(f'Unsupported format: "{fmt}".')

  def set_liqstate(self, branch: K, stage: int) -> None:
    """Mark a specific stage of a given separator branch as final state
    of the liquid phase.

    Parameters
    ----------
    branch: str
      The name of a branch.

    stage: int
      The stage in a given branch.

    Notes
    -----
    The marked stage can be used to compute parameters of the liquid
    phase of a mixture. Given `branch` and `stage` are stored as
    `_liqbranch` and `_liqstage` attributes of this class.
    """
    self._liqbranch = branch
    self._liqstage = stage
    pass

  def set_gasstate(self, branch: K, stage: int) -> None:
    """Mark a specific stage of a given separator branch as final state
    of the gas phase.

    Parameters
    ----------
    branch: str
      The name of a branch.

    stage: int
      The stage in a given branch.

    Notes
    -----
    The marked stage can be used to compute parameters of the gas
    phase of a mixture. Given `branch` and `stage` are stored as
    `_gasbranch` and `_gasstage` attributes of this class.
    """
    self._gasbrach = branch
    self._gasstage = stage
    pass

  def get_liqstate(self) -> State:
    """Return a state corresponding to the final stage of the liquid
    phase separation procedure.
    """
    return self[self._liqbranch][self._liqstage]

  def get_gasstate(self) -> State:
    """Return a state corresponding to the final stage of the gas
    phase separation procedure.
    """
    return self[self._gasbrach][self._gasstage]
