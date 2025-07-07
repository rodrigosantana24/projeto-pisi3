from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

class CreditsTransformer(BaseEstimator, TransformerMixin):
    """ Transforma a coluna 'credits' em features binárias para os N atores mais frequentes.  """
    def __init__(self, top_n=10, other_label='Outros'):
        self.top_n = top_n
        self.other_label = other_label
        self.mlb = None
        self.top_actors = None

    def fit(self, X, y=None):
        # Concatena todos os créditos, separa por '-' e conta as ocorrências
        all_credits = X.str.split('-').explode()
        top_actors = all_credits.value_counts().nlargest(self.top_n).index
        self.top_actors = top_actors
        
        # Prepara o MultiLabelBinarizer com as classes (top atores + 'Outros')
        self.mlb = MultiLabelBinarizer(classes=list(top_actors) + [self.other_label])
        self.mlb.fit([list(top_actors) + [self.other_label]])  # Fit com todas as classes possíveis
        return self

    def transform(self, X):
        X_processed = X.str.split('-').apply(self._filter_top_actors)
        
        # Transforma os dados usando o MultiLabelBinarizer
        encoded_data = self.mlb.transform(X_processed)
        
        # Cria um DataFrame com os nomes das colunas
        columns = [f'actor_{cls}' for cls in self.mlb.classes_]
        return pd.DataFrame(encoded_data, columns=columns, index=X.index)

    def _filter_top_actors(self, actor_list):
        # Retorna uma lista com os atores do top N ou 'Outros'
        return [actor if actor in self.top_actors else self.other_label for actor in actor_list]