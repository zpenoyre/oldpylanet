# This is a submodule to test sphinx's autodoc feature.
__all__ = ["BigFrog", "BigMole", "is_frog", "is_mole", "what_is"]


class BigFrog:
    """
    A big frog object.

    Parameters
    ---------
    iceBaskt : bool
        A bool to determine whether the big frog has an ice basketball.
    """

    def __init__(self,ice_basketball=True):
        """
        Initialize big frog.
        """
        self.ice_basketball = ice_basketball

    def about_self(self):
        """
        Return information about a big frog instance created from BigFrog.
        """
        return "Am big frog. Is {0} that am have ice basktball".format(self.ice_basketball)

class BigMole:
    """
    A big mole object.

    Parameters
    ---------
    iceBaskt : bool
            A bool to determine whether the big mole has a big worm.
    """

    def __init__(self,worm=True):
        """
        Initialize big mole.
        """
        self.worm = worm

    def about_self(self):
        """
        Return information about a big mole instance created from BigMole.
        """
        return "Am big mole. Is {0} that am eat worm".format(self.worm)


def is_frog(big_animal):
    """
    Decide if the big animal is a BigFrog object.

    Parameters
    ----------
    big_animal : obj
        An object.
    """
    if isinstance(big_animal, BigFrog):
        return True
    else:
        return False

def is_mole(big_animal):
    """
    Decide if the big animal is a BigMole object.

    Parameters
    ----------
    big_animal : obj
        An object.
    """
    if isinstance(big_animal, BigMole):
        return True
    else:
        return False

def what_is(big_animal):
    """
    Decide what the big animal is.

    Parameters
    ----------
    big_animal : str
        A string indicating the name of the animal.
    """
    if isinstance(big_animal, BigFrog):
        animal = "frog"
    elif isinstance(big_animal, BigMole):
        animal = "mole"
    else:
        animal = None
    return animal


