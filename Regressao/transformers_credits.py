import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class CapTransformer(BaseEstimator, TransformerMixin):
    """
    Trata outliers em colunas numéricas, limitando os valores com base no
    Intervalo Interquartil (IQR).
    """
    def __init__(self, columns):
        self.columns = columns
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_copy = X.copy()
        for col in self.columns:
            Q1 = X_copy[col].quantile(0.25)
            Q3 = X_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.bounds_[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            lower, upper = self.bounds_[col]
            X_copy[col] = X_copy[col].clip(lower, upper)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return list(input_features)


class CreditsCounter(BaseEstimator, TransformerMixin):
    """Conta o número de créditos (atores/diretores) e adiciona como feature."""
    def __init__(self, credits_column='credits'):
        self.credits_column = credits_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Garante que a coluna seja string antes de usar .split()
        X_copy['credits_count'] = X_copy[self.credits_column].astype(str).apply(lambda x: len(x.split('-')) if x != 'nan' else 0)
        X_copy = X_copy.drop(columns=[self.credits_column])
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return [f for f in input_features if f != self.credits_column] + ['credits_count']


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
        
        # Preenche NaNs que podem surgir de datas inválidas
        X_copy['year'] = X_copy['year'].fillna(X_copy['year'].median())
        X_copy['month'] = X_copy['month'].fillna(X_copy['month'].median())

        X_copy = X_copy.drop(columns=[self.date_column])
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return [f for f in input_features if f != self.date_column] + ['year', 'month']


class MultiLabelProcessor(BaseEstimator, TransformerMixin):
    """Processa colunas com múltiplos rótulos usando MultiLabelBinarizer."""
    def __init__(self, multilabel_cols_config):
        self.multilabel_cols_config = multilabel_cols_config
        self.mlbs_ = {}
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        X_copy = X.copy()
        output_features = list(X.columns)

        for col, config in self.multilabel_cols_config.items():
            # Cria a lista de categorias mais frequentes a partir do treino
            exploded = X_copy[col].astype(str).str.split(config['sep']).explode().str.strip()
            top_categories = exploded.value_counts().nlargest(config['top_n']).index.tolist()
            
            # Função para transformar os dados
            def filter_top(vals):
                vals_list = [v.strip() for v in str(vals).split(config['sep'])]
                return [v if v in top_categories else 'Outros' for v in vals_list]

            processed_col = X_copy[col].apply(filter_top)
            
            # Treina o binarizador
            mlb = MultiLabelBinarizer()
            mlb.fit(processed_col)
            
            self.mlbs_[col] = (mlb, top_categories)
            
            # Atualiza a lista de nomes de features de saída
            if col in output_features:
                output_features.remove(col)
            output_features.extend([f"{config['prefix']}_{cls}" for cls in mlb.classes_])

        self.feature_names_out_ = output_features
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, (mlb, top_categories) in self.mlbs_.items():
            config = self.multilabel_cols_config[col]

            def filter_top(vals):
                vals_list = [v.strip() for v in str(vals).split(config['sep'])]
                return [v if v in top_categories else 'Outros' for v in vals_list]

            processed_col = X_copy[col].apply(filter_top)
            
            encoded_df = pd.DataFrame(
                mlb.transform(processed_col),
                columns=[f"{config['prefix']}_{cls}" for cls in mlb.classes_],
                index=X_copy.index
            )
            X_copy = X_copy.drop(columns=[col]).join(encoded_df)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Agrupa categorias raras em uma única coluna."""
    def __init__(self, column, top_n=10, other_label='Other'):
        self.column = column
        self.top_n = top_n
        self.other_label = other_label
        self.top_categories_ = []

    def fit(self, X, y=None):
        # O ColumnTransformer pode passar um array com uma coluna
        df = pd.DataFrame(X, columns=[self.column])
        self.top_categories_ = df[self.column].value_counts().nlargest(self.top_n).index
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=[self.column])
        X_df[self.column] = X_df[self.column].where(X_df[self.column].isin(self.top_categories_), self.other_label)
        # Retorna no mesmo formato que recebeu
        return X_df.to_numpy() if isinstance(X, np.ndarray) else X_df

    def get_feature_names_out(self, input_features=None):
        return list(input_features)