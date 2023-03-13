"""Misc functions."""
import numpy as np


def smape_np(Mleft, Mright):
    """Return smape scores.

    Parameters:
    -----------
    Mleft, Mright (nd.array)
        the 2 tables of floats
    """
    num = 2 * np.abs(Mright - Mleft)
    denom = np.abs(Mleft) + np.abs(Mright) + np.finfo(float).eps
    return 100 / Mleft.size * np.sum(num / denom)
