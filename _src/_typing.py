from typing import (
    # Never,
    Protocol,
)

from numpy import (
    typing as npt,
    float64,
)


class Eos(Protocol):
    def run(self, P):
        pass

