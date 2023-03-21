"""Misc functions."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


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


@dataclass
class VizualizeResults:
    """This class returns zoomed signal for easy vizualization."""

    dataset: pd.DataFrame
    page: str = 'Acier_inoxydable_fr.wikipedia.org_desktop_all-agents'
    nsamp_before: int = 50
    nsamp_after: int = 62

    def get_aligned_results(self, Y_test, pred):
        """Return 3 aligned Series (initial datasets, targets, predictions)."""
        iloc = self.dataset.columns.get_loc(Y_test.columns[0])
        window = np.arange(-self.nsamp_before, self.nsamp_after)+iloc
        all_samples = self.dataset.loc[self.page].iloc[window]
        y_true = Y_test.loc[self.page].rename("y_true")
        y_pred = pd.DataFrame(pred,
                              index=Y_test.index,
                              columns=Y_test.columns)\
                   .loc[self.page].rename("y_pred")
        for ii in [all_samples, y_true, y_pred]:
            ii.index = pd.to_datetime(ii.index)
        return all_samples, y_true, y_pred
