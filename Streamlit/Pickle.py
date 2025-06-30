import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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

# === Função para definir instância de exemplo ===
def get_instancia():
    return {
        'popularity': 55.3,
        'budget': 150000000,
        'runtime': 120,
        'original_language': 'en',
        'release_date': '2015-07-16',
        'genres': ['Action', 'Adventure', 'Science Fiction'],
        'production_companies': ['Warner Bros. Pictures', 'Outros']
    }


def main():
    # 2. Carrega objetos pickled
    with open('Streamlit/models/mlb_genres.pkl', 'rb') as f:
        mlb_genres = pickle.load(f)
    with open('Streamlit/models/mlb_production.pkl', 'rb') as f:
        mlb_prod = pickle.load(f)
    with open('Streamlit/models/xgb_pipeline.pkl', 'rb') as f:
        modelo = pickle.load(f)

    # 3. Monta DataFrame inicial
    instancia = get_instancia()
    df = pd.DataFrame([instancia])

    # 4. Aplica MultiLabelBinarizer para gêneros e produtoras
    df_gen = pd.DataFrame(
    mlb_genres.transform(df['genres']),
    columns=[f'genre_{cls}' for cls in mlb_genres.classes_],
    index=df.index
    )
    df_prod = pd.DataFrame(
    mlb_prod.transform(df['production_companies']),
    columns=[f'production_{cls}' for cls in mlb_prod.classes_],
    index=df.index
    )

    # 5. Concatena e remove colunas originais
    df_final = pd.concat([
    df.drop(columns=['genres', 'production_companies']),
    df_gen, df_prod
    ], axis=1)

    # Reconstrua as colunas esperadas manualmente:
    colunas_esperadas = list(df.drop(columns=['genres', 'production_companies']).columns)
    colunas_esperadas += [f'genre_{cls}' for cls in mlb_genres.classes_]
    colunas_esperadas += [f'production_{cls}' for cls in mlb_prod.classes_]

    # Reindexa para garantir a ordem e presença
    df_final = df_final.reindex(columns=colunas_esperadas, fill_value=0)

    # 6. Faz a predição
    y_pred = modelo.predict(df_final)
    print(f"Predição do modelo: {y_pred[0]}")


if __name__ == '__main__':
    main()