import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
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

def plot_elbow_method(X, features): # Recebe features como argumento
    # st.subheader("Método do Cotovelo") # Subheader será adicionado na função chamadora
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[features])

    wss = []
    k_values = list(range(1, 26))
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

def plot_davies_bouldin_method(X_scaled):
    # st.subheader("Davies-Bouldin Index (DBI)") # Subheader será adicionado na função chamadora

    dbi_scores = []
    k_values = list(range(2, 26)) # DBI exige no mínimo 2 clusters
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

def recomendar_filmes_por_cluster(df_analisado, nome_filme, coluna_nome='title', n_recomendacoes=5):
    nome_filme = nome_filme.strip().lower()
    if coluna_nome not in df_analisado.columns or 'cluster' not in df_analisado.columns:
        st.error("O DataFrame precisa ter as colunas 'title' e 'cluster'.")
        return
    df_analisado[coluna_nome + '_lower'] = df_analisado[coluna_nome].str.lower()
    filme_linha = df_analisado[df_analisado[coluna_nome + '_lower'] == nome_filme]
    if filme_linha.empty:
        st.warning("Filme não encontrado no dataset.")
        return
    cluster_id = filme_linha['cluster'].values[0]
    st.success(f"Filme '{filme_linha[coluna_nome].values[0]}' está no cluster {cluster_id}.")
    recomendacoes = df_analisado[
        (df_analisado['cluster'] == cluster_id) &
        (df_analisado[coluna_nome + '_lower'] != nome_filme)
    ].sample(n=min(n_recomendacoes, len(df_analisado[df_analisado['cluster'] == cluster_id]) - 1), random_state=42)
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[features])

    st.header("1. Análise para Determinar o Número Ideal de Clusters (K)")
    st.markdown("---")

    # --- Seção do Método do Cotovelo ---
    st.subheader("1.1. Método do Cotovelo")
    col_elbow_plot, col_elbow_text = st.columns([0.6, 0.4])

    with col_elbow_plot:
        # Passa 'features' para a função plot_elbow_method
        elbow_fig = plot_elbow_method(X, features)
        st.pyplot(elbow_fig)
        plt.close(elbow_fig)

    with col_elbow_text:
        st.markdown("### Interpretação do Método do Cotovelo")
        st.write("Procure um 'cotovelo' (ponto de inflexão) no gráfico. Este ponto indica onde a diminuição da Soma dos Quadrados Intra-cluster (WSS) começa a desacelerar, sugerindo um número ideal de clusters.")

    st.markdown("---")

    # --- Seção do Davies-Bouldin Index (DBI) por K ---
    st.subheader("1.2. Davies-Bouldin Index (DBI) por K")
    col_dbi_plot, col_dbi_text = st.columns([0.6, 0.4])

    with col_dbi_plot:
        dbi_fig = plot_davies_bouldin_method(X_scaled)
        st.pyplot(dbi_fig)
        plt.close(dbi_fig)

    with col_dbi_text:
        st.markdown("### Interpretação do Davies-Bouldin Index (DBI)")
        st.write("O DBI mede a similaridade dentro dos clusters e a separação entre eles. **Valores menores de DBI indicam uma melhor clusterização** (clusters mais compactos e bem separados). Procure o ponto de mínimo no gráfico.")
        if not all(pd.isna(plot_davies_bouldin_method(X_scaled).axes[0].lines[0].get_ydata())): # Checa se há dados válidos
            min_dbi_k_val = plot_davies_bouldin_method(X_scaled).axes[0].lines[0].get_xdata()[np.nanargmin(plot_davies_bouldin_method(X_scaled).axes[0].lines[0].get_ydata())]
            st.info(f"O K com o menor Davies-Bouldin Index sugerido é: `{min_dbi_k_val}`.")
        else:
            st.warning("Não foi possível determinar o K ideal pelo DBI com os dados atuais.")


    st.markdown("---")

    # --- Seção de Análise Detalhada (Silhueta e DBI para K selecionado) ---
    st.header("2. Análise Detalhada para um Número de Clusters (K) Selecionado")
    st.write("Escolha um valor de K para ver as métricas de Silhouette e Davies-Bouldin Index em detalhes.")

    k_selected = st.slider("Selecione o Número de Clusters (K)", 2, 25, 3) # K a partir de 2 para métricas
    
    if st.button("Analisar Clusters para K Selecionado"):
        if k_selected < 2:
            st.warning("O Coeficiente de Silhueta e o Davies-Bouldin Index exigem K >= 2.")
        else:
            st.subheader(f"Resultados para K = {k_selected}")
            kmeans_selected = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
            labels_selected = kmeans_selected.fit_predict(X_scaled)

            # --- Coeficiente de Silhueta ---
            st.markdown("#### 2.1. Coeficiente de Silhueta")
            try:
                plot_silhouette(X_scaled, labels_selected, title=f"Gráfico de Silhueta para K={k_selected}")
            except ValueError as e:
                st.error(f"Erro ao calcular Coeficiente de Silhueta: {e}. Verifique se há clusters com um único ponto ou menos.")

            st.markdown("---")

            # --- Davies-Bouldin Index (DBI) para o K selecionado ---
            st.markdown("#### 2.2. Davies-Bouldin Index (DBI)")
            try:
                if len(np.unique(labels_selected)) > 1:
                    davies_bouldin_single_k = davies_bouldin_score(X_scaled, labels_selected)
                    st.markdown(f"**Davies-Bouldin Index (DBI) para K={k_selected}:** `{davies_bouldin_single_k:.3f}`")
                    st.info("Valores menores de DBI indicam melhor clusterização (clusters mais densos e separados).")
                else:
                    st.warning("Não foi possível calcular o DBI. Apenas um cluster foi formado para o K selecionado.")
            except Exception as e:
                st.error(f"Erro ao calcular Davies-Bouldin Index para K={k_selected}: {e}")

def clustering_kmeans_single():
    st.info("Utilize o dataset df_analisado.csv para análise dos clusters.")
    data_path = 'data/df_analisado.csv'
    df = load_data(data_path)
    plot_kmeans_single(df)

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
    st.info("Carregando o arquivo `df_analisado.csv` para recomendações.")
    
    df_analisado = load_data('data/df_analisado.csv')
    st.write("Pré-visualização dos dados:", df_analisado.head())
    nome_input = st.text_input("Digite o nome de um filme para receber recomendações:")
    if st.button("Recomendar filmes"):
        if 'title' not in df_analisado.columns or 'cluster' not in df_analisado.columns:
            st.error("O DataFrame precisa ter as colunas 'title' e 'cluster'.")
        else:
            recomendar_filmes_por_cluster(df_analisado, nome_input, coluna_nome='title')

# Função principal para ser chamada pela main
def run_clustering(selected_clustering_topic):
    st.title("🔍 Clusterização de Filmes")
    if selected_clustering_topic == "Método do Cotovelo":
        clustering_elbow()
    elif selected_clustering_topic == "K-Means (1 valor de K)":
        clustering_kmeans_single()
    elif selected_clustering_topic == "PCA + KMeans + Silhueta":
        clustering_pca_kmeans()
    elif selected_clustering_topic == "Recomendação de Filmes":
        clustering_recommendation()

