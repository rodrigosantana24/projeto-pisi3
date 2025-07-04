from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# === Definições de classes customizadas usadas no pipeline ===
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')
        X['year'] = X[self.date_column].dt.year
        X['month'] = X[self.date_column].dt.month
        return X.drop(columns=[self.date_column])

class CapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            lower, upper = self.bounds_[col]
            X[col] = X[col].clip(lower, upper)
        return X
    
class LogCapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            lower, upper = self.bounds_[col]
            X[col] = X[col].clip(lower, upper)
            X[col] = np.log1p(X[col])
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, lower_percentile=0.05, upper_percentile=0.95):
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for c in self.columns:
            low = X[c].quantile(self.lower_percentile)
            high = X[c].quantile(self.upper_percentile)
            self.bounds_[c] = (low, high)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (low, high) in self.bounds_.items():
            X[c] = X[c].clip(low, high)
        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, column=None, top_n=10, other_label='Other'):
        self.column = column
        self.top_n = top_n
        self.other_label = other_label
        self.top_categories_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            col_data = X[self.column]
        else:
            col_data = pd.Series(X[:, 0])

        vc = col_data.value_counts()
        self.top_categories_ = vc.nlargest(self.top_n).index
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            col_data = X[self.column]
        else:
            col_data = pd.Series(X[:, 0])

        transformed = col_data.apply(lambda x: x if x in self.top_categories_ else self.other_label)

        return pd.DataFrame(transformed, columns=[self.column])