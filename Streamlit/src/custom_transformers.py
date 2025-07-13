# ==============================================================================
# ARQUIVO CORRIGIDO: src/custom_transformers.py
# ==============================================================================

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')
        # Cria as novas features
        new_features = pd.DataFrame({
            'year': X[self.date_column].dt.year,
            'month': X[self.date_column].dt.month
        }, index=X.index)
        
        # Remove a coluna original e adiciona as novas
        X = X.drop(columns=[self.date_column])
        X = pd.concat([X, new_features], axis=1)
        return X
    
    def get_feature_names_out(self, input_features=None):
        # Este método já estava bom, mas um pequeno ajuste para robustez
        output_features = [f for f in input_features if f != self.date_column]
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
        # --- CORREÇÃO ---
        # Este transformador não muda os nomes nem o número de colunas.
        # Ele deve simplesmente retornar a lista de features de entrada.
        return list(input_features)
    
class LogCapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            # Garante que os valores sejam positivos antes de aplicar o log no transform
            temp_col = X[col][X[col] > 0]
            Q1 = temp_col.quantile(0.25)
            Q3 = temp_col.quantile(0.75)
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
        
    def get_feature_names_out(self, input_features=None):
        # --- ADICIONADO ---
        # Método que estava faltando. Não altera nomes de colunas.
        return list(input_features)

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

    def get_feature_names_out(self, input_features=None):
        # --- ADICIONADO ---
        # Método que estava faltando. Não altera nomes de colunas.
        return list(input_features)

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
        # Este método já estava correto.
        return list(input_features)

class TopNMultiLabelTransformer(BaseEstimator, TransformerMixin):
    # Este transformador já estava correto, nenhuma mudança necessária.
    def __init__(self, top_n=10, other_label='Outros', prefix=''):
        self.top_n = top_n
        self.other_label = other_label
        self.prefix = prefix
        self.mlb = None
        self.top_items = None

    def fit(self, X, y=None):
        all_items = X.dropna().str.split('-').explode().str.strip()
        all_items = all_items[all_items != '']
        self.top_items = all_items.value_counts().nlargest(self.top_n).index.tolist()
        if not self.top_items:
            self.top_items = [self.other_label]
        classes = list(self.top_items)
        if self.other_label not in classes:
            classes.append(self.other_label)
        self.mlb = MultiLabelBinarizer(classes=classes)
        self.mlb.fit([classes])
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
    def __init__(self, budget_col='budget', runtime_col='runtime', new_col_name='budget_runtime_ratio'):
        self.budget_col = budget_col
        self.runtime_col = runtime_col
        self.new_col_name = new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.new_col_name] = X[self.budget_col] / X[self.runtime_col]
        X[self.new_col_name] = X[self.new_col_name].replace([np.inf, -np.inf], np.nan)
        X[self.new_col_name] = X[self.new_col_name].fillna(0)
        return X

    def get_feature_names_out(self, input_features=None):
        output_features = list(input_features)
        if self.new_col_name not in output_features:
            output_features.append(self.new_col_name)
        return output_features