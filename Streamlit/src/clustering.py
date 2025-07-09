import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from matplotlib import cm

from streamlit_searchbox import st_searchbox

features = [
    # Variáveis numéricas principais
    'imdb_rating', 'imdb_votes',
    'release_year', 'release_month',

    # Gêneros (ajuste conforme seus one-hot)
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama',
    'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Outros',
    'Romance', 'Science Fiction', 'Thriller', 'War', 'Western',

    # Principais produtoras (se quiser incluir)
    '20th Century Fox', 'Columbia Pictures', 'Goldwyn', 'Mayer', 'Metro',
    'Paramount Pictures', 'RKO Radio Pictures', 'Universal Pictures',
    'Walt Disney Productions', 'Warner Bros. Pictures',

    # Variáveis derivadas
    'popularity_per_budget', 'vote_weighted', 'runtime_per_budget',
    'vote_popularity_weighted', 'budget_log', 'vote_count_log',

    # Continentes (one-hot)
    'continent_Africa', 'continent_America_do_Norte', 'continent_Asia',
    'continent_Europa', 'continent_Outros'
]

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Removendo a função get_clustered_data_for_recommendation
# pois os arquivos já virão clusterizados e serão carregados por load_df_analisado

@st.cache_data
def load_df_analisado(data_path_df, expected_cluster_col_name):
    """
    Carrega o DataFrame e renomeia a coluna 'cluster' para o nome esperado
    (ex: 'cluster_5' ou 'cluster_33'). Também adiciona 'title_lower'.
    """
    st.write(f"Carregando dados com clusters prontos de: {data_path_df}...")
    df = pd.read_csv(data_path_df)

    # Verifica se a coluna 'cluster' genérica existe e a renomeia
    if 'cluster' in df.columns:
        df = df.rename(columns={'cluster': expected_cluster_col_name})
    else:
        st.error(f"Erro: A coluna 'cluster' não foi encontrada no DataFrame '{data_path_df}'. "
                 f"Certifique-se de que o arquivo contém uma coluna de cluster genérica.")
        return None

    # Adiciona a coluna title_lower
    if 'title' in df.columns:
        df['title_lower'] = df['title'].str.lower()
    else:
        st.error("Erro: Coluna 'title' não encontrada no DataFrame carregado. "
                 "Verifique se o arquivo CSV possui uma coluna 'title'.")
        return None

    # Verifica se a coluna renomeada existe (deve existir após a renomeação)
    if expected_cluster_col_name not in df.columns:
        st.error(f"Erro interno: A coluna '{expected_cluster_col_name}' não foi corretamente estabelecida no DataFrame.")
        return None

    return df

def plot_elbow_method(X, features):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[features])
    wss = []
    k_values = list(range(1, 16))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wss.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, wss, 'bo-')
    ax.set_title("Método do Cotovelo")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Soma dos Quadrados Intra-cluster (WSS)")
    ax.set_xticks(k_values)
    plt.tight_layout()
    return fig

def plot_calinski_harabasz_method(X_scaled):
    chi_scores = []
    k_values = list(range(2, 16))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        if len(np.unique(labels)) > 1:
            chi_scores.append(calinski_harabasz_score(X_scaled, labels))
        else:
            chi_scores.append(np.nan)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, chi_scores, 'co-')
    ax.set_title("Calinski-Harabasz Index por K")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Calinski-Harabasz Index")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle='--', alpha=0.6)
    if not all(pd.isna(chi_scores)):
        max_chi_k = k_values[np.nanargmax(chi_scores)]
        max_chi_val = np.nanmax(chi_scores)
        ax.axvline(x=max_chi_k, color='purple', linestyle=':', lw=2, label=f'Melhor K (CHI): {max_chi_k}')
        ax.legend()
    plt.tight_layout()
    return fig

def plot_davies_bouldin_method(X_scaled):
    dbi_scores = []
    k_values = list(range(2, 16))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        if len(np.unique(labels)) > 1:
            dbi_scores.append(davies_bouldin_score(X_scaled, labels))
        else:
            dbi_scores.append(np.nan)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, dbi_scores, 'go-')
    ax.set_title("Davies-Bouldin Index por K")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Davies-Bouldin Index")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle='--', alpha=0.6)
    if not all(pd.isna(dbi_scores)):
        min_dbi_k = k_values[np.nanargmin(dbi_scores)]
        min_dbi_val = np.nanmin(dbi_scores)
        ax.axvline(x=min_dbi_k, color='red', linestyle=':', lw=2, label=f'Melhor K (DBI): {min_dbi_k}')
        ax.legend()
    plt.tight_layout()
    return fig

def recomendar_filmes_por_cluster(df_analisado, nome_filme, cluster_col_name='cluster', coluna_nome='title', n_recomendacoes=5, cluster_count=None):
    nome_filme = nome_filme.strip().lower()
    if cluster_col_name not in df_analisado.columns:
        st.error(f"Erro: A coluna de cluster '{cluster_col_name}' não foi encontrada no DataFrame.")
        return
    if coluna_nome not in df_analisado.columns:
        st.error(f"Erro: A coluna '{coluna_nome}' (título do filme) não foi encontrada no DataFrame.")
        return
    filme_linha = df_analisado[df_analisado[coluna_nome + '_lower'] == nome_filme]
    if filme_linha.empty:
        st.warning("Filme não encontrado no dataset.")
        return
    cluster_id = filme_linha[cluster_col_name].values[0]
    cluster_info = f"{cluster_count}" if cluster_count else "desconhecido"
    st.success(f"Filme '{filme_linha[coluna_nome].values[0]}' está no cluster {cluster_id} (usando {cluster_info} clusters).")
    possible_recommendations = df_analisado[
        (df_analisado[cluster_col_name] == cluster_id) &
        (df_analisado[coluna_nome + '_lower'] != nome_filme)
    ]
    num_available_recs = len(possible_recommendations)
    if num_available_recs == 0:
        st.info(f"Não há outros filmes no cluster {cluster_id} para recomendar.")
        return
    n_to_sample = min(n_recomendacoes, num_available_recs)
    recomendacoes = possible_recommendations.sample(n=n_to_sample)
    st.subheader("🎬 Recomendações:")
    st.table(recomendacoes[[coluna_nome]])

def plot_silhouette(X_scaled, labels, title="Gráfico de Silhueta"):
    n_clusters = len(np.unique(labels))
    silhouette_vals = silhouette_samples(X_scaled, labels)
    silhouette_avg = np.mean(silhouette_vals)
    y_lower = 10
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X_scaled) + (n_clusters + 1) * 10])
    for i in range(n_clusters):
        ith_vals = silhouette_vals[labels == i]
        ith_vals.sort()
        size_cluster_i = ith_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_title(title)
    ax1.set_xlabel("Coeficiente de Silhueta")
    ax1.set_ylabel("Clusters")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(f"**Silhouette Score médio:** `{silhouette_avg:.3f}`")

def resumo_clusters(df, features):
    st.write("### Média das variáveis por cluster:")
    st.dataframe(df.groupby("cluster")[features].mean().round(2))

def plot_kmeans_single(df):
    st.subheader("Análise de KMeans")
    st.write(df.head())

def clustering_elbow():
    st.info("Carregando o dataset `X.csv` para análise de clusters.")
    X = load_data('data/X.csv')
    st.write("Pré-visualização dos dados:", X.head())
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[features])
    st.header("1. Análise para Determinar o Número Ideal de Clusters (K)")
    st.markdown("---")
    st.subheader("1.1. Método do Cotovelo")
    col_elbow_plot, col_elbow_text = st.columns([0.6, 0.4])
    with col_elbow_plot:
        elbow_fig = plot_elbow_method(X, features)
        st.pyplot(elbow_fig)
        plt.close(elbow_fig)
    st.markdown("---")
    st.subheader("1.2. Davies-Bouldin Index (DBI) por K")
    col_dbi_plot, col_dbi_text = st.columns([0.6, 0.4])
    with col_dbi_plot:
        dbi_fig = plot_davies_bouldin_method(X_scaled)
        st.pyplot(dbi_fig)
        plt.close(dbi_fig)
        dbi_scores_suggestion = []
        k_values_for_suggestion = list(range(2, 26))
        for k_val in k_values_for_suggestion:
            kmeans_temp = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X_scaled)
            if len(np.unique(labels_temp)) > 1:
                dbi_scores_suggestion.append(davies_bouldin_score(X_scaled, labels_temp))
            else:
                dbi_scores_suggestion.append(np.nan)
        if not all(pd.isna(dbi_scores_suggestion)):
            min_dbi_k_val = k_values_for_suggestion[np.nanargmin(dbi_scores_suggestion)]
            st.info(f"O K com o menor Davies-Bouldin Index sugerido é: `{min_dbi_k_val}`.")
        else:
            st.warning("Não foi possível determinar o K ideal pelo DBI com os dados atuais.")
    st.markdown("---")
    st.subheader("1.3. Calinski-Harabasz Index (CHI) por K")
    col_chi_plot, col_chi_text = st.columns([0.6, 0.4])
    with col_chi_plot:
        chi_fig = plot_calinski_harabasz_method(X_scaled)
        st.pyplot(chi_fig)
        plt.close(chi_fig)

def clustering_pca_kmeans():
    st.info("Carregando o dataset `X.csv` para análise PCA + KMeans + Silhueta.")
    X = load_data('data/X.csv')
    st.write("Pré-visualização dos dados:", X.head())
    st.write("Colunas usadas:", features)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[features])
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    st.write(f"**Número de componentes PCA escolhidos:** `{X_pca.shape[1]}`")
    n_clusters = st.slider("Escolha o número de clusters (K)", 2, 25, 6, key="pca_kmeans")
    if st.button("Rodar PCA + KMeans"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
        labels = kmeans.fit_predict(X_pca)
        st.success(f"Silhouette Score médio com PCA: {silhouette_score(X_pca, labels):.3f}")
        plot_silhouette(X_pca, labels, title="Silhueta com PCA + KMeans")
        df_analisado = X.copy()
        df_analisado['cluster'] = labels
        for i in range(min(3, X_pca.shape[1])):
            df_analisado[f'pca_{i}'] = X_pca[:, i]
        resumo_clusters(df_analisado, features)

# Função síncrona para buscar as sugestões (sem async)
def get_movie_suggestions(search_term: str, all_titles: list) -> list:
    if not search_term:
        return []
    search_term_lower = search_term.strip().lower()
    filtered_titles = [
        title for title in all_titles
        if search_term_lower in title.lower()
    ]
    return sorted(filtered_titles)[:10]

def clustering_recommendation():
    st.title("🎬 Recomendação de Filmes por Cluster")

    recommendation_type = st.radio(
        "Escolha o tipo de recomendação:",
        ("Recomendação Genérica (5 clusters)", "Recomendação Específica (33 clusters)")
    )

    df_analisado_clustered = None
    cluster_col_name = None
    cluster_info = None

    if recommendation_type == "Recomendação Genérica (5 clusters)":
        cluster_col_name = 'cluster_5' # Nome da coluna que esperamos APÓS a renomeação
        cluster_info = 5
        # Carrega df_analisado_5.csv e renomeia 'cluster' para 'cluster_5'
        df_analisado_clustered = load_df_analisado('data/df_analisado_5.csv', cluster_col_name)
    else: # "Recomendação Específica (33 clusters)"
        cluster_col_name = 'cluster_33' # Nome da coluna que esperamos APÓS a renomeação
        cluster_info = 33
        # Carrega df_analisado_33.csv e renomeia 'cluster' para 'cluster_33'
        df_analisado_clustered = load_df_analisado('data/df_analisado_33.csv', cluster_col_name)
    
    if df_analisado_clustered is None:
        st.error("Não foi possível carregar os dados para recomendação. Verifique os arquivos CSV e a presença da coluna 'cluster'.")
        return

    # Extrai todos os títulos da coluna 'title' do DataFrame carregado e processado
    all_titles = df_analisado_clustered['title'].unique().tolist()
    
    st.markdown("---")
    st.subheader("🔍 Encontre um Filme")

    selected_movie = st_searchbox(
        label="Digite o nome do filme para buscar:",
        search_function=lambda search_term: get_movie_suggestions(search_term, all_titles),
        placeholder="Digite o título de um filme",
        key="movie_search_box"
    )

    num_recommendations = st.slider("Quantos filmes recomendar?", 1, 10, 5)

    if st.button("Recomendar filmes", key="recommend_button"):
        if selected_movie is None or selected_movie.strip() == "":
            st.warning("Por favor, digite ou selecione um filme para continuar.")
            return

        nome_normalizado = selected_movie.strip().lower()
        
        filme_no_dataset = df_analisado_clustered[df_analisado_clustered['title_lower'] == nome_normalizado]

        if filme_no_dataset.empty:
            st.error(f"O filme '{selected_movie}' não foi encontrado no nosso dataset.")
            return

        recomendar_filmes_por_cluster(
            df_analisado_clustered,
            filme_no_dataset['title'].iloc[0],
            cluster_col_name=cluster_col_name,
            coluna_nome='title',
            n_recomendacoes=num_recommendations,
            cluster_count=cluster_info
        )

# Função principal para ser chamada pela main
def run_clustering(selected_clustering_topic):
    st.title("🔍 Clusterização de Filmes")
    if selected_clustering_topic == "Método do Cotovelo":
        clustering_elbow()
    elif selected_clustering_topic == "PCA + KMeans + Silhueta":
        clustering_pca_kmeans()
    elif selected_clustering_topic == "Recomendação de Filmes":
        clustering_recommendation()