import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

@st.cache_data # -> Cache the data loading function to avoid reloading on every interaction
def load_data(data_path):
    df = pd.read_csv(data_path)
    df_exploded = explode_genre_names(df)
    media_global = df_exploded['vote_average'].mean()
    media_por_genero = calculate_average_rating_by_genre(df_exploded)
    return df, df_exploded, media_global, media_por_genero

def explode_genre_names(df):
    df['genre_names'] = df['genres'].apply(lambda x: x.split('-'))
    df_exploded = df.explode('genre_names')
    return df_exploded

def calculate_average_rating_by_genre(df_exploded):
    return df_exploded.groupby('genre_names')['vote_average'].mean()

df, df_exploded, media_global, media_por_genero = load_data('filmes_filtrados_sem_nulos.csv')

st.write("# Analises Exploratórias")

st.write("### BoxPlot das notas médias por gênero cinematográfico")
def plot_boxplot_by_genre(df_exploded, media_global):
    generos_unicos = sorted(df_exploded['genre_names'].unique())
    generos_selecionados = st.multiselect(
        "Selecione os gêneros para exibir no BoxPlot:",
        generos_unicos,
        default=generos_unicos
    )

    df_filtrado = filter_by_selected_genres(df_exploded, generos_selecionados)
    fig = create_boxplot(df_filtrado, generos_selecionados, media_global)
    st.pyplot(fig)

def filter_by_selected_genres(df_exploded, generos_selecionados):
    df_filtrado = df_exploded[df_exploded['genre_names'].isin(generos_selecionados)].copy()
    generos_ordenados = sorted(generos_selecionados)
    df_filtrado['genre_names'] = pd.Categorical(df_filtrado['genre_names'], categories=generos_ordenados, ordered=True)
    df_filtrado = df_filtrado.sort_values('genre_names')
    return df_filtrado

def create_boxplot(df_filtrado, generos_selecionados, media_global):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=df_filtrado,
        x='genre_names',
        y='vote_average',
        ax=ax,
        palette="Set2",
        showmeans=True,
        meanprops={"marker":"o",
                   "markerfacecolor":"white", 
                   "markeredgecolor":"black",
                   "markersize":"7"}
    )
    ax.axhline(media_global, color='red', linestyle='--', linewidth=2, label=f'Média Global: {media_global:.2f}')
    ax.set_xlabel("Gênero", fontsize=12)
    ax.set_ylabel("Nota Média", fontsize=12)
    ax.set_title("BoxPlot das Notas Médias por Gênero Cinematográfico", fontsize=15, fontweight='bold')
    plt.xticks(rotation=35, ha='right')
    ax.legend()
    plt.tight_layout()
    return fig

plot_boxplot_by_genre(df_exploded, media_global)

st.write("### Histograma da distribuição da nota média")
def plot_histogram(df):
    st.write("""
            ##### Ajuste o número de bins do histograma:
            Bins: itervalos em que os dados são agrupados. 
            Aumentar o número de bins pode revelar mais detalhes na distribuição, enquanto diminuir pode suavizar a visualização.
             """)
    bins = st.slider("Número de bins", min_value=5, max_value=100, value=30, step=1)

    st.write("Escolha se deseja exibir a curva de densidade:")
    show_kde = st.checkbox("Exibir curva de densidade (KDE)", value=True)

    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.histplot(df['vote_average'], kde=show_kde, bins=bins, color='blue', ax=ax_hist)
    ax_hist.set_title('Distribuição da nota média', fontsize=14)
    ax_hist.set_xlabel('Nota média (vote_average)')
    ax_hist.set_ylabel('Frequência')
    ax_hist.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_hist)

plot_histogram(df)

st.write("### Matriz de Correlação (Spearman)")

def get_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    return numeric_cols

def plot_correlation_matrix(df):
    numeric_cols = get_numeric_columns(df)
    corr_matrix = df[numeric_cols].corr(method='spearman')
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='RdYlGn',
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
        ax=ax_corr
    )
    ax_corr.set_title('Matriz de Correlação (Spearman)', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig_corr)

plot_correlation_matrix(df)

st.write("### Receita média por idioma original")
def plot_revenue_by_language(df):
    receita_por_idioma = df.groupby('original_language')['revenue'].mean().sort_values(ascending=False).reset_index()
    fig_lang = px.bar(receita_por_idioma, 
                     x='original_language', 
                     y='revenue', 
                     title='Receita Média por Idioma Original',
                     labels={'original_language': 'Idioma Original', 'revenue': 'Receita Média'},
                     color='revenue',
                     color_continuous_scale='Blues')
    st.plotly_chart(fig_lang)

plot_revenue_by_language(df)

st.write("### Receita média por gênero cinematográfico")
def plot_revenue_by_genre(df_exploded):
    receita_por_genero = df_exploded.groupby('genre_names')['revenue'].mean().sort_values(ascending=False).reset_index()
    fig_genre = px.bar(receita_por_genero, 
                      x='genre_names', 
                      y='revenue', 
                      title='Receita Média por Gênero Cinematográfico',
                      labels={'genre_names': 'Gênero', 'revenue': 'Receita Média'},
                      color='revenue',
                      color_continuous_scale='Teal')
    fig_genre.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_genre)

plot_revenue_by_genre(df_exploded)


st.write("### Filtros por Nota e Genero (Spearman)")
def filtrar_filmes_por_nota_genero(df, nota_minima, generos):
    """
    Retorna filmes com nota igual ou superior à nota_minima e pertencentes aos gêneros informados.
    Parâmetros:
        df: DataFrame original (não explode)
        nota_minima: float
        generos: lista de strings (gêneros desejados)
    Retorno:
        DataFrame filtrado
    """
    # Explode os gêneros para facilitar o filtro
    df_exploded = df.copy()
    df_exploded['genre_names'] = df_exploded['genres'].apply(lambda x: x.split('-'))
    df_exploded = df_exploded.explode('genre_names')
    # Filtra por nota e gênero
    filtrado = df_exploded[
        (df_exploded['vote_average'] >= nota_minima) &
        (df_exploded['genre_names'].isin(generos))
    ]
    # Remove duplicatas caso um filme tenha mais de um gênero selecionado
    return filtrado.drop_duplicates(subset=['id'])

nota_minima = st.slider("Nota mínima", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
generos = st.multiselect("Gêneros", sorted(df_exploded['genre_names'].unique()), default=[])
if generos:
    filmes_filtrados = filtrar_filmes_por_nota_genero(df, nota_minima, generos)
    st.write(f"Filmes encontrados: {len(filmes_filtrados)}")
    st.dataframe(filmes_filtrados[['title', 'vote_average', 'genre_names']])
else:
    st.write("Selecione pelo menos um gênero.")


def busca_titulos(df):
    st.write("### Busca de Filmes por Título")
    termo = st.text_input("Digite parte do título do filme:")
    if termo:
        resultados = df[df['title'].str.contains(termo, case=False, na=False)]
        st.write(f"Filmes encontrados: {len(resultados)}")
        st.dataframe(resultados[['title', 'release_date', 'vote_average']])
busca_titulos(df)

st.write("## Análise com K-Means")

st.write("### 🔧 Configurações do K-Means")
k = st.slider("Número de clusters (K)", min_value=2, max_value=10, value=4)

if st.button("Rodar K-Means"):
    # Pré-processamento e Clusterização
    features = ['popularity', 'vote_average', 'revenue']
    df_kmeans = df[features].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_kmeans)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df_kmeans['cluster'] = labels

    st.success("K-Means aplicado com sucesso!")

    # Popularidade por Cluster
    st.write("###  Popularidade média por Cluster")
    pop_means = df_kmeans.groupby('cluster')['popularity'].mean()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=pop_means.index, y=pop_means.values, ax=ax1, palette="muted")
    ax1.set_ylabel("Popularidade média")
    ax1.set_title("Popularidade por Cluster")
    st.pyplot(fig1)

    # Gêneros por Cluster
    st.write("### Distribuição de Gêneros por Cluster")
    df_with_id = df.copy()
    df_with_id['cluster'] = labels
    df_exploded_k = explode_genre_names(df_with_id)
    dist = df_exploded_k.groupby(['cluster','genre_names']).size().unstack(fill_value=0)
    prop = dist.div(dist.sum(axis=1), axis=0)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(prop, annot=True, fmt=".2f", cmap="viridis", ax=ax2)
    ax2.set_title("Proporção de Gêneros por Cluster")
    st.pyplot(fig2)

    # Silhueta
    st.write("### Análise de Silhueta")
    silhouette_avg = silhouette_score(X_scaled, labels)
    st.write(f"**Score médio da silhueta:** {silhouette_avg:.3f}")

    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / k)
        ax3.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax3.set_title("Gráfico de Silhueta para cada Cluster")
    ax3.set_xlabel("Coeficiente de Silhueta")
    ax3.set_ylabel("Clusters")
    ax3.axvline(x=silhouette_avg, color="red", linestyle="--")
    st.pyplot(fig3)

st.write("## Comparação de K-Means para múltiplos valores de K")

k_values = st.slider("Selecione o valor máximo de K", min_value=2, max_value=10, value=6)

features = ['popularity', 'vote_average', 'revenue']
df_kmeans = df[features].dropna().copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_kmeans)

results = []
genre_props = {}

for k in range(2, k_values + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df_kmeans['cluster'] = labels

    # Popularidade média por cluster
    pop_means = df_kmeans.groupby('cluster')['popularity'].mean()
    pop_mean_overall = pop_means.mean()

    # Silhueta média
    sil_score = silhouette_score(X_scaled, labels)

    # Para gêneros: mantenha alinhamento de índices e explode
    df_with_id = df.copy()
    df_with_id = df_with_id.loc[df_kmeans.index]  # garante alinhamento correto
    df_with_id['cluster'] = labels
    df_exploded_k = explode_genre_names(df_with_id)

    # Cluster de maior popularidade
    top_cluster = pop_means.idxmax()
    dist = df_exploded_k[df_exploded_k['cluster'] == top_cluster].groupby('genre_names').size()
    dist_prop = dist / dist.sum()
    genre_props[k] = dist_prop.sort_values(ascending=False).head(5)  # top 5 gêneros

    results.append({
        'K': k,
        'silhouette': sil_score,
        'popularity_mean': pop_mean_overall,
        'top_cluster': top_cluster
    })

results_df = pd.DataFrame(results)

# Gráfico 1: Silhueta média por K
fig1, ax1 = plt.subplots()
sns.lineplot(x='K', y='silhouette', data=results_df, marker='o', ax=ax1)
ax1.set_title('Silhueta média por K')
ax1.set_ylim(0,1)
st.pyplot(fig1)

# Gráfico 2: Popularidade média geral por K
fig2, ax2 = plt.subplots()
sns.lineplot(x='K', y='popularity_mean', data=results_df, marker='o', ax=ax2)
ax2.set_title('Popularidade média geral por K')
st.pyplot(fig2)

# Mostrar gêneros top por K para o cluster mais popular
st.write("### 🎭 Top 5 gêneros do cluster mais popular por K")
for k in range(2, k_values + 1):
    st.write(f"**K = {k} | Cluster mais popular: {results_df.loc[results_df['K']==k, 'top_cluster'].values[0]}**")
    top_genres = genre_props[k]
    st.bar_chart(top_genres)


# Seleção das variáveis numéricas
features = ['popularity', 'vote_average', 'revenue']
df_kmeans = df[features].dropna().copy()

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_kmeans)

# Calcular WSS para diferentes valores de K
wss = []
k_values = list(range(1, 11))
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wss.append(kmeans.inertia_)  # inertia_ é o WSS

# Plot do Elbow Method
fig, ax = plt.subplots()
ax.plot(k_values, wss, marker='o')
ax.set_title('Método do Cotovelo')
ax.set_xlabel('Número de Clusters (K)')
ax.set_ylabel('Soma dos Quadrados Intra-cluster (WSS)')
plt.grid(True)
st.pyplot(fig)

