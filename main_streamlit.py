import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import plotly.express as px

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
