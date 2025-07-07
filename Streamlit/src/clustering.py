import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from matplotlib import cm

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

@st.cache_data
def get_clustered_data_for_recommendation(data_path_df, data_path_X, features):
    st.write("Processando dados e realizando clusterização (isso pode demorar na primeira vez)...")
    
    # Carrega o dataframe principal (com títulos, etc.)
    df_analisado = pd.read_csv(data_path_df)

    # Carrega o dataframe APENAS com as features para clusterização
    X_clustering = pd.read_csv(data_path_X)
    
    # Garante que X_clustering contém as features corretas e está alinhado com df_analisado
    X_scaled = RobustScaler().fit_transform(X_clustering[features])

    # Realiza a clusterização para 5 clusters
    kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_analisado['cluster_5'] = kmeans_5.fit_predict(X_scaled)

    # Realiza a clusterização para 33 clusters
    kmeans_33 = KMeans(n_clusters=33, random_state=42, n_init=10)
    df_analisado['cluster_33'] = kmeans_33.fit_predict(X_scaled)

    st.success("Clusterização concluída e dados prontos para recomendação!")
    return df_analisado

def plot_elbow_method(X, features): # Recebe features como argumento
    # st.subheader("Método do Cotovelo") # Subheader será adicionado na função chamadora
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[features])

    wss = []
    k_values = list(range(1, 11))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(6, 4)) # Tamanho ajustado
    ax.plot(k_values, wss, 'bo-')
    ax.set_title("Método do Cotovelo")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Soma dos Quadrados Intra-cluster (WSS)")
    ax.set_xticks(k_values) # Para garantir que todos os K's sejam mostrados

    plt.tight_layout()
    return fig # Retorna a figura

def plot_calinski_harabasz_method(X_scaled):
    # st.subheader("Calinski-Harabasz Index (CHI) por K") # Subheader será adicionado na função chamadora

    chi_scores = []
    # CHI exige no mínimo 2 clusters
    k_values = list(range(2, 11)) # De 2 a 25 clusters, igual ao DBI
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        # CHI também exige que haja mais de um cluster formado
        if len(np.unique(labels)) > 1:
            chi_scores.append(calinski_harabasz_score(X_scaled, labels))
        else:
            chi_scores.append(np.nan) # Adiciona NaN se o cálculo não for possível

    fig, ax = plt.subplots(figsize=(6, 4)) # Tamanho ajustado, consistente com os outros gráficos
    ax.plot(k_values, chi_scores, 'co-') # 'c' para ciano, 'o' para bolinhas
    ax.set_title("Calinski-Harabasz Index por K")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Calinski-Harabasz Index")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Opcional: Adicionar anotação para a melhor K
    # A maior pontuação de CHI é a melhor
    if not all(pd.isna(chi_scores)): # Verifica se não são todos NaN
        max_chi_k = k_values[np.nanargmax(chi_scores)] # np.nanargmax para ignorar NaNs
        max_chi_val = np.nanmax(chi_scores) # np.nanmax para ignorar NaNs
        ax.axvline(x=max_chi_k, color='purple', linestyle=':', lw=2, label=f'Melhor K (CHI): {max_chi_k}')
        ax.legend()
        # A mensagem de info será adicionada na função clustering_elbow

    plt.tight_layout()
    return fig # Retorne a figura para ser plotada no Streamlit

def plot_davies_bouldin_method(X_scaled):
    # st.subheader("Davies-Bouldin Index (DBI)") # Subheader será adicionado na função chamadora

    dbi_scores = []
    k_values = list(range(2, 11)) # DBI exige no mínimo 2 clusters
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        if len(np.unique(labels)) > 1: # Verifica se há mais de um cluster formado
            dbi_scores.append(davies_bouldin_score(X_scaled, labels))
        else:
            dbi_scores.append(np.nan) # Se apenas um cluster, adicione NaN

    fig, ax = plt.subplots(figsize=(6, 4)) # Tamanho ajustado
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
    return fig # Retorna a figura

# MUDANÇAS AQUI: adicionei 'cluster_col_name' como argumento e removi 'random_state'
def recomendar_filmes_por_cluster(df_analisado, nome_filme, cluster_col_name='cluster', coluna_nome='title', n_recomendacoes=5, cluster_count=None):
    nome_filme = nome_filme.strip().lower()

    if cluster_col_name not in df_analisado.columns:
        st.error(f"Erro: A coluna de cluster '{cluster_col_name}' não foi encontrada no DataFrame.")
        return

    if coluna_nome not in df_analisado.columns:
        st.error(f"Erro: A coluna '{coluna_nome}' (título do filme) não foi encontrada no DataFrame.")
        return

    df_analisado[coluna_nome + '_lower'] = df_analisado[coluna_nome].str.lower()
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
    fig, ax1 = plt.subplots(figsize=(8, 4)) # Tamanho ajustado para silhueta
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
    st.pyplot(fig) # Plota diretamente aqui
    plt.close(fig) # Fecha a figura
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

    # Escalonamento dos dados (feito uma vez para todos os plots e cálculos)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[features])

    st.header("1. Análise para Determinar o Número Ideal de Clusters (K)")
    st.markdown("---")

    # --- Seção do Método do Cotovelo ---
    st.subheader("1.1. Método do Cotovelo")
    col_elbow_plot, col_elbow_text = st.columns([0.6, 0.4])

    with col_elbow_plot:
        elbow_fig = plot_elbow_method(X, features)
        st.pyplot(elbow_fig)
        plt.close(elbow_fig)
    st.markdown("---")
    # --- Seção do Davies-Bouldin Index (DBI) por K ---
    st.subheader("1.2. Davies-Bouldin Index (DBI) por K")
    col_dbi_plot, col_dbi_text = st.columns([0.6, 0.4])

    with col_dbi_plot:
        dbi_fig = plot_davies_bouldin_method(X_scaled)
        st.pyplot(dbi_fig)
        plt.close(dbi_fig)
       
        # Recalcula dbi_scores para pegar o K sugerido
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

    # --- NOVA SEÇÃO: Calinski-Harabasz Index (CHI) por K ---
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
    X_scaled = scaler.fit_transform(X)

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

def clustering_recommendation():
    st.title("🎬 Recomendação de Filmes por Cluster")

    # Escolha entre 5 ou 33 clusters
    recommendation_type = st.radio(
        "Escolha o tipo de recomendação:",
        ("Recomendação Genérica (5 clusters)", "Recomendação Específica (33 clusters)")
    )

    # Carregamento de dados com base na escolha
    if recommendation_type == "Recomendação Genérica (5 clusters)":
        df_analisado = load_df_analisado('data/df_analisado_5.csv')
        cluster_info = 5
    else:
        df_analisado = load_df_analisado('data/df_analisado_33.csv')
        cluster_info = 33

    # Entrada do usuário
    nome_input = st.text_input("Digite o nome do filme:")
    num_recommendations = st.slider("Quantos filmes recomendar?", 1, 10, 5)

    if st.button("Recomendar filmes"):
        if nome_input.strip() == "":
            st.warning("Digite um nome de filme.")
            return

        nome_normalizado = nome_input.strip().lower()
        df_analisado['title_lower'] = df_analisado['title'].str.lower()

        filme_linha = df_analisado[df_analisado['title_lower'] == nome_normalizado]

        if filme_linha.empty:
            st.error("Filme não encontrado no dataset.")
            return

        nome_original = filme_linha['title'].values[0]
        cluster_id = filme_linha['cluster'].values[0]

        st.success(f"🎥 O filme **'{nome_original}'** pertence ao cluster **{cluster_id}** (de {cluster_info} clusters).")

        # Filtra filmes do mesmo cluster, exceto o original
        filmes_recomendados = df_analisado[
            (df_analisado['cluster'] == cluster_id) &
            (df_analisado['title_lower'] != nome_normalizado)
        ]

        if filmes_recomendados.empty:
            st.info("Nenhum outro filme encontrado nesse cluster para recomendar.")
            return

        n = min(num_recommendations, len(filmes_recomendados))
        recomendacoes = filmes_recomendados.sample(n=n)

        st.subheader("🔁 Recomendações aleatórias:")
        st.table(recomendacoes[['title']])


@st.cache_data
def load_df_analisado(data_path_df):
    st.write("Carregando dados com clusters prontos...")
    df = pd.read_csv(data_path_df)

    if 'cluster' not in df.columns:
        st.error("Erro: A coluna 'cluster' não foi encontrada no DataFrame.")
        return None

    return df


# Função principal para ser chamada pela main
def run_clustering(selected_clustering_topic):
    st.title("🔍 Clusterização de Filmes")
    if selected_clustering_topic == "Método do Cotovelo":
        clustering_elbow()
    elif selected_clustering_topic == "PCA + KMeans + Silhueta":
        clustering_pca_kmeans()
    elif selected_clustering_topic == "Recomendação de Filmes":
        clustering_recommendation()

