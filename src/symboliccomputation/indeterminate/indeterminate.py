"""A way to store indeterminates, or the atoms of this package."""

class Indeterminate:
    """An indeterminate.

    Like $x$, to be used in a mathematical expression, where it doesn't
    represent a specific value.
    """

    name: str = "x"

    def __init__(self, name: str) -> None:
        """Initialize an instance of the Indeterminate class.

        Args:
            name (str): _description_
        """
        self.name = name

    def __hash__(self) -> int:
        """To use this class as a dict key, need to be able to compute a hash."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """To test equality of two indeterminates."""
        if not isinstance(other, Indeterminate):
            return NotImplemented
        return self.name == other.name
