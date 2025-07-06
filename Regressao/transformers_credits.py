from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extrai o ano e o mês de uma coluna de data."""
    def __init__(self, date_column='release_date'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.date_column] = pd.to_datetime(X_copy[self.date_column], errors='coerce')
        X_copy['year'] = X_copy[self.date_column].dt.year
        X_copy['month'] = X_copy[self.date_column].dt.month
        X_copy = X_copy.drop(columns=[self.date_column])
        X_copy['year'].fillna(X_copy['year'].median(), inplace=True)
        X_copy['month'].fillna(X_copy['month'].median(), inplace=True)
        return X_copy

class CreditsCounter(BaseEstimator, TransformerMixin):
    """Conta o número de créditos e adiciona como uma nova feature."""
    def __init__(self, credits_column='credits'):
        self.credits_column = credits_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['credits_count'] = X_copy[self.credits_column].astype(str).apply(lambda x: len(x.split('-')))
        X_copy = X_copy.drop(columns=[self.credits_column])
        return X_copy

class MultiLabelProcessor(BaseEstimator, TransformerMixin):
    """Processa colunas com múltiplos rótulos usando MultiLabelBinarizer."""
    def __init__(self, multilabel_cols_config):
        self.multilabel_cols_config = multilabel_cols_config
        self.mlbs = {}

    def fit(self, X, y=None):
        for col, config in self.multilabel_cols_config.items():
            exploded = X[col].str.split(config['sep']).explode().str.strip()
            top = exploded.value_counts().nlargest(config['top_n']).index
            
            def filter_top(vals):
                vals = [v.strip() for v in str(vals).split(config['sep'])]
                return [v if v in top else 'Outros' for v in vals]

            processed_col = X[col].apply(filter_top)
            mlb = MultiLabelBinarizer()
            mlb.fit(processed_col)
            self.mlbs[col] = (mlb, top) 
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, (mlb, top) in self.mlbs.items():
            config = self.multilabel_cols_config[col]
            def filter_top(vals):
                vals = [v.strip() for v in str(vals).split(config['sep'])]
                return [v if v in top else 'Outros' for v in vals]

            processed_col = X_copy[col].apply(filter_top)
            encoded_df = pd.DataFrame(
                mlb.transform(processed_col),
                columns=[f"{config['prefix']}_{cls}" for cls in mlb.classes_],
                index=X_copy.index
            )
            X_copy = X_copy.drop(columns=[col]).join(encoded_df)
        return X_copy

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Agrupa categorias raras em uma coluna."""
    def __init__(self, column, top_n=10, other_label='Other'):
        self.column = column
        self.top_n = top_n
        self.other_label = other_label
        self.top_categories_ = []

    def fit(self, X, y=None):
        self.top_categories_ = X[self.column].value_counts().nlargest(self.top_n).index
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].where(X_copy[self.column].isin(self.top_categories_), self.other_label)
        return X_copy