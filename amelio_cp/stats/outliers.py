import pandas as pd
import numpy as np


class Outliers:
    def __init__(self):
        pass

    @staticmethod
    def finding_outliers(data:pd.DataFrame, feature:str, k:str = 1.5):
        """finds the outliers for a specific feature

        Parameters
        ----------
        data : pd.DataFrame
            dataframe with all the data
        feature : str
            feature (i.e., column) to consider
        k : str, optional
            IQR multiplier, by default 1.5 according to Tuckey's IQR method (q1-1.5*iqr ≈ mu-2.7*sigma)

        Returns
        -------
        pd.dataframe
            returns a df with the identified outliers and their value (+ low and high boundaries defined according to Tuckey's method)
        """

        s = data[feature]
        s.dropna(axis=0)

        q1, q3 = s.quantile([0.25,0.75])
        iqr = q3 - q1

        low_boundary, high_boundary = (q1 - k * iqr), (q3 + k * iqr)
        range = (data[feature] < low_boundary) | (data[feature] > high_boundary)

        out = data.loc[range, [feature]].copy()
        out["lower_bound"] = low_boundary
        out["upper_bound"] = high_boundary
        out["direction"] = np.where(out[feature] < low_boundary, "low", "high")

        return out