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
