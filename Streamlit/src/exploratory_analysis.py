import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

@st.cache_data
def load_data(data_path):
    df = pd.read_csv(data_path)
    df['genre_names'] = df['genres'].apply(lambda x: x.split('-'))
    df_exploded = df.explode('genre_names')
    return df, df_exploded

def run_exploratory_analysis(selected_subtopic):
    df, df_exploded = load_data('data/filmes_filtrados_sem_nulos.csv')
    media_global = df['vote_average'].mean()

    st.title("Análises Exploratórias")

    if selected_subtopic == "BoxPlot por Gênero":
        st.write("### BoxPlot das notas médias por gênero cinematográfico")
        generos_unicos = sorted(df_exploded['genre_names'].unique())
        generos_selecionados = st.multiselect(
            "Selecione os gêneros para exibir no BoxPlot:",
            generos_unicos,
            default=generos_unicos,
            key="boxplot_generos"
        )

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
                hue='genre_names',
                palette="Set2",
                showmeans=True,
                meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"7"},
                legend=False
            )
            ax.axhline(media_global, color='red', linestyle='--', linewidth=2, label=f'Média Global: {media_global:.2f}')
            ax.set_xlabel("Gênero", fontsize=12)
            ax.set_ylabel("Nota Média", fontsize=12)
            ax.set_title("BoxPlot das Notas Médias por Gênero Cinematográfico", fontsize=15, fontweight='bold')
            plt.xticks(rotation=35, ha='right')
            ax.legend()
            plt.tight_layout()
            return fig

        df_boxplot = filter_by_selected_genres(df_exploded, generos_selecionados)
        fig_boxplot = create_boxplot(df_boxplot, generos_selecionados, media_global)
        st.pyplot(fig_boxplot)

    elif selected_subtopic == "Histograma das Notas":
        st.write("### Histograma da distribuição da nota média")
        bins = st.slider("Número de bins", min_value=5, max_value=100, value=30, step=1, key="hist_bins")
        show_kde = st.checkbox("Exibir curva de densidade (KDE)", value=True, key="hist_kde")

        def plot_histogram(df, bins, show_kde):
            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            sns.histplot(df['vote_average'], kde=show_kde, bins=bins, color='blue', ax=ax_hist)
            ax_hist.set_title('Distribuição da nota média', fontsize=14)
            ax_hist.set_xlabel('Nota média (vote_average)')
            ax_hist.set_ylabel('Frequência')
            ax_hist.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_hist)

        plot_histogram(df, bins, show_kde)

    elif selected_subtopic == "Matriz de Correlação":
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

    elif selected_subtopic == "Receita por Idioma":
        st.write("### Receita média por idioma original")
        def plot_revenue_by_language(df):
            receita_por_idioma = df.groupby('original_language')['revenue'].mean().sort_values(ascending=False).reset_index()
            fig_lang = px.bar(receita_por_idioma, 
                            x='original_language', 
                            y='revenue', 
                            labels={'original_language': 'Idioma Original', 'revenue': 'Receita Média'},
                            color='revenue',
                            color_continuous_scale='Blues')
            st.plotly_chart(fig_lang, use_container_width=True, config={"scrollZoom": True})

        plot_revenue_by_language(df)

    elif selected_subtopic == "Receita por Gênero":
        st.write("### Receita média por gênero cinematográfico")
        def plot_revenue_by_genre(df_exploded):
            receita_por_genero = df_exploded.groupby('genre_names')['revenue'].mean().sort_values(ascending=False).reset_index()
            fig_genre = px.bar(receita_por_genero, 
                            x='genre_names', 
                            y='revenue', 
                            labels={'genre_names': 'Gênero', 'revenue': 'Receita Média'},
                            color='revenue',
                            color_continuous_scale='Teal')
            fig_genre.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_genre, use_container_width=True, config={"scrollZoom": True})

        plot_revenue_by_genre(df_exploded)

    elif selected_subtopic == "Filtro por Nota e Gênero":
        st.write("### Filtros por Nota e Gênero (Spearman)")
        nota_minima = st.slider("Nota mínima", min_value=0.0, max_value=10.0, value=7.0, step=0.1, key="nota_minima")
        generos_filtro = st.multiselect("Gêneros", sorted(df_exploded['genre_names'].unique()), default=[], key="generos_filtro")

        def filtrar_filmes_por_nota_genero(df, nota_minima, generos):
            df_exploded = df.copy()
            df_exploded['genre_names'] = df_exploded['genres'].apply(lambda x: x.split('-'))
            df_exploded = df_exploded.explode('genre_names')
            filtrado = df_exploded[
                (df_exploded['vote_average'] >= nota_minima) &
                (df_exploded['genre_names'].isin(generos))
            ]
            return filtrado.drop_duplicates(subset=['id'])

        if generos_filtro:
            filmes_filtrados = filtrar_filmes_por_nota_genero(df, nota_minima, generos_filtro)
            st.write(f"Filmes encontrados: {len(filmes_filtrados)}")
            st.dataframe(filmes_filtrados[['title', 'vote_average', 'genre_names']])
        else:
            st.write("Selecione pelo menos um gênero.")