from typing import (
  Protocol,
)

from resim.pvt.datatypes import (
  Float,
  Vector,
  SepResult,
  State,
  States,
)


class Separator[Eos](Protocol):
  """A callable object that can be used to perform a separation
  procedure.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  Returns
  -------
  Separation procedure results as an instance of `SepResult`.
  """
  def __call__(
    self,
    eos: Eos,
    yi: Vector[Float],
    /,
  ) -> SepResult[str, States[State]]:
    pass
