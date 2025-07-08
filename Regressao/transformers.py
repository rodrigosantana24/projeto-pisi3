from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

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
    
    def get_feature_names_out(self, input_features=None):
        output_features = list(input_features)
        if self.date_column in output_features:
            output_features.remove(self.date_column)
        output_features.extend(['year', 'month'])
        return output_features

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
    def get_feature_names_out(self, input_features=None):
        # Retorna os nomes das colunas que não mudam
        return self.columns if self.columns is not None else input_features
    
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
    def __init__(self, column, top_n=10, other_label='Other'):
        self.column = column
        self.top_n = top_n
        self.other_label = other_label

    def fit(self, X, y=None):
        vc = X[self.column].value_counts()
        self.top_ = vc.nlargest(self.top_n).index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].where(
            X[self.column].isin(self.top_),
            other=self.other_label
        )
        return X
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return [self.column]
    


class TopNMultiLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=10, other_label='Outros', prefix=''):
        self.top_n = top_n
        self.other_label = other_label
        self.prefix = prefix
        self.mlb = None
        self.top_items = None

    def fit(self, X, y=None):
        # Remove nulos e explode os valores multilabel
        all_items = X.dropna().str.split('-').explode().str.strip()
        all_items = all_items[all_items != '']  # remove vazios

        self.top_items = all_items.value_counts().nlargest(self.top_n).index.tolist()

        if not self.top_items:
            self.top_items = [self.other_label]

        classes = list(self.top_items)
        if self.other_label not in classes:
            classes.append(self.other_label)

        self.mlb = MultiLabelBinarizer(classes=classes)
        self.mlb.fit([classes])  # precisa de uma amostra com todas as classes
        return self

    def transform(self, X):
        X_processed = X.fillna('').str.split('-').apply(self._filter_top_items)

        encoded = self.mlb.transform(X_processed)
        columns = [f'{self.prefix}_{cls}' for cls in self.mlb.classes_]
        return pd.DataFrame(encoded, columns=columns, index=X.index)

    def _filter_top_items(self, items):
        if not isinstance(items, list):
            return [self.other_label]

        return [
            item.strip() if item.strip() in self.top_items else self.other_label
            for item in items if isinstance(item, str) and item.strip()
        ]
    
    def get_feature_names_out(self, input_features=None):
        return [f'{self.prefix}_{cls}' for cls in self.mlb.classes_]
    
class BudgetRuntimeRatioTransformer(BaseEstimator, TransformerMixin):
    """
    Calcula a razão entre o orçamento (budget) e a duração (runtime) de um filme.
    Trata divisões por zero e valores ausentes.
    """
    def __init__(self, budget_col='budget', runtime_col='runtime', new_col_name='budget_runtime_ratio'):
        self.budget_col = budget_col
        self.runtime_col = runtime_col
        self.new_col_name = new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Calcula a razão
        X[self.new_col_name] = X[self.budget_col] / X[self.runtime_col]
        # Substitui infinitos (resultantes da divisão por zero) por NaN
        X[self.new_col_name] = X[self.new_col_name].replace([np.inf, -np.inf], np.nan)
        # Preenche todos os NaNs com 0
        X[self.new_col_name] = X[self.new_col_name].fillna(0)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return [self.new_col_name]
        return list(input_features) + [self.new_col_name]