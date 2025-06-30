def run_clustering():
    
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_samples, silhouette_score
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


    def plot_elbow_method(X):
        st.subheader("Método do Cotovelo")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[features])

        wss = []
        k_values = list(range(1, 11))
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            wss.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(k_values, wss, 'bo-')
        ax.set_title("Método do Cotovelo")
        ax.set_xlabel("Número de Clusters (K)")
        ax.set_ylabel("Soma dos Quadrados Intra-cluster (WSS)")
        st.pyplot(fig)

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
        st.markdown(f"**Silhouette Score médio:** `{silhouette_avg:.3f}`")


    def resumo_clusters(df, features):
        st.write("### Média das variáveis por cluster:")
        st.dataframe(df.groupby("cluster")[features].mean().round(2))


    def plot_kmeans_single(df):
        st.subheader("Análise de KMeans")
        st.write(df.head())


    # --------- Lógica da Página ---------

    st.title("🔍 Clusterização de Filmes")

    st.sidebar.subheader("Tópicos de Clusterização")
    topics = [
        "Método do Cotovelo",
        "K-Means (1 valor de K)",
        "PCA + KMeans + Silhueta",
        "Recomendação de Filmes"
    ]
    selected_topic = st.sidebar.radio("Escolha um tópico:", topics)

    if selected_topic == "Método do Cotovelo":
        st.info("Utilize o dataset X.csv para o método do cotovelo.")
        uploaded_file = st.file_uploader("Selecione o arquivo X.csv", type="csv")

        if uploaded_file:
            X = pd.read_csv(uploaded_file)
            st.write("Pré-visualização dos dados:", X.head())
            plot_elbow_method(X)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            k_sil = st.slider("Escolha o número de clusters (K) para a silhueta", 2, 10, 3)
            if st.button("Mostrar Silhueta"):
                kmeans = KMeans(n_clusters=k_sil, random_state=42)
                labels = kmeans.fit_predict(X_scaled)
                st.subheader("Gráfico de Silhueta")
                plot_silhouette(X_scaled, labels)

    elif selected_topic == "K-Means (1 valor de K)":
        st.info("Utilize o dataset df_analisado.csv para análise dos clusters.")
        data_path = 'data/df_analisado.csv'
        df = load_data(data_path)
        plot_kmeans_single(df)

    elif selected_topic == "PCA + KMeans + Silhueta":
        st.info("Utilize o dataset X.csv para análise PCA + KMeans + Silhueta.")
        uploaded_file = st.file_uploader("Selecione o arquivo X.csv", type="csv", key="pca_file")

        if uploaded_file:
            X = pd.read_csv(uploaded_file)
            st.write("Pré-visualização dos dados:", X.head())
            features = list(features)
            st.write("Colunas usadas:", features)

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=0.95, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            st.write(f"**Número de componentes PCA escolhidos:** `{X_pca.shape[1]}`")

            n_clusters = st.slider("Escolha o número de clusters (K)", 2, 10, 6, key="pca_kmeans")
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

    elif selected_topic == "Recomendação de Filmes":
        st.info("Faça upload do arquivo df_analisado.csv para recomendações.")
        uploaded_file = st.file_uploader("Selecione o arquivo df_analisado.csv", type="csv", key="recom_file")
        if uploaded_file:
            df_analisado = pd.read_csv(uploaded_file)
            st.write("Pré-visualização dos dados:", df_analisado.head())
            nome_input = st.text_input("Digite o nome de um filme para receber recomendações:")
            if st.button("Recomendar filmes"):
                if 'title' not in df_analisado.columns or 'cluster' not in df_analisado.columns:
                    st.error("O DataFrame precisa ter as colunas 'title' e 'cluster'.")
                else:
                    recomendar_filmes_por_cluster(df_analisado, nome_input, coluna_nome='title')

