

import pandas as pd
import numpy as np



def clean_outage_data(df):
    clean_data = df.copy()
    #get rid of second header (just contains units)
    clean_data.columns = clean_data.columns.droplevel(-1)
    #the variables column has all na values and 'OBS' is the same as the index so we will drop these columns
    clean_data = clean_data.drop(['variables', 'OBS'], 1)
    #combine the 'OUTAGE.START.DATE' and  'OUTAGE.START.TIME' columns to be one 'OUTAGE.START' column
    clean_data['OUTAGE.START'] = clean_data['OUTAGE.START.DATE'].dt.strftime('%A/%Y/%m/%d')+ " " +clean_data['OUTAGE.START.TIME'].transform(str)
    clean_data = clean_data.drop(['OUTAGE.START.DATE', 'OUTAGE.START.TIME'], 1)
    #combine the OUTAGE.RESTORATION.DATE and OUTAGE.RESTORATION.TIME columns to be one 'OUTAGE.RESTORATION' column
    clean_data['OUTAGE.RESTORATION'] = clean_data['OUTAGE.RESTORATION.DATE'].dt.strftime('%A/%Y/%m/%d')+ " " +clean_data['OUTAGE.RESTORATION.TIME'].transform(str)
    clean_data = clean_data.drop(['OUTAGE.RESTORATION.DATE', 'OUTAGE.RESTORATION.TIME'], 1)
    clean_data['OUTAGE.START'] = pd.to_datetime(clean_data['OUTAGE.START'])
    clean_data['OUTAGE.RESTORATION'] = pd.to_datetime(clean_data['OUTAGE.RESTORATION'])
    return clean_data


from sklearn.base import BaseEstimator, TransformerMixin


class StdScalerByGroup(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 2, 2], 'c2': [3, 1, 2, 0]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> std.grps_ is not None
        True
        """
        # X may not be a pandas dataframe (e.g. a np.array)
        df = pd.DataFrame(X)

        # A dictionary of means/standard-deviations for each column, for each group.        
        self.grps_ = (X.groupby(X.columns[0]).mean(), X.groupby(X.columns[0]).std())

        return self

    def transform(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 3, 4], 'c2': [1, 2, 3, 4]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> out = std.transform(X)
        >>> out.shape == (4, 2)
        True
        >>> np.isclose(out.abs(), 0.707107, atol=0.001).all().all()
        True
        """

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")
        
        # Define a helper function here?
        def calc_zscore(x):
            return (x-x.mean())/x.std()

        # X may not be a dataframe (e.g. np.array)
        df = pd.DataFrame(X)
        
        
        return df.groupby(df.columns[0]).transform(calc_zscore)

