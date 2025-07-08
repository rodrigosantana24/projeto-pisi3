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
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    return df, df_exploded

def carregarTraducaoGeneros():
    TRADUCOES_GENEROS = {
                'Action': 'Ação',
                'Adventure': 'Aventura',
                'Animation': 'Animação',
                'Comedy': 'Comédia',
                'Crime': 'Crime',
                'Documentary': 'Documentário',
                'Drama': 'Drama',
                'Family': 'Família',
                'Fantasy': 'Fantasia',
                'History': 'História',
                'Horror': 'Terror',
                'Music': 'Música',
                'Mystery': 'Mistério',
                'Romance': 'Romance',
                'Science Fiction': 'Ficção Científica',
                'TV Movie': 'Filme de TV',
                'Thriller': 'Suspense',
                'War': 'Guerra',
                'Western': 'Faroeste'
            }
    return TRADUCOES_GENEROS

def run_exploratory_analysis(selected_subtopic):
    df, df_exploded = load_data('Streamlit/data/filmes_filtrados_sem_nulos.csv')
    media_global = df['vote_average'].mean()

    st.title("Análises Exploratórias")

    if selected_subtopic == "BoxPlot por Gênero":
        st.write("### BoxPlot das Notas Médias por Gênero")

        def plot_boxplot_notas_por_genero(df):
            df_exploded = df.dropna(subset=['genres']).copy()
            df_exploded['genres'] = df_exploded['genres'].str.split(', ')
            df_exploded = df_exploded.explode('genres')
            df_exploded['genres_translated'] = df_exploded['genres'].map(carregarTraducaoGeneros())

            generos_unicos = sorted(df_exploded['genres_translated'].dropna().unique())
            generos_selecionados = st.multiselect("Selecione os gêneros para o BoxPlot:", generos_unicos, default=generos_unicos)

            df_filtrado = df_exploded[df_exploded['genres_translated'].isin(generos_selecionados)]

            if not df_filtrado.empty:
                media_global = df['vote_average'].mean()
                df_filtrado['genres_translated'] = pd.Categorical(
                    df_filtrado['genres_translated'],
                    categories=generos_selecionados,
                    ordered=True
                )
                df_filtrado = df_filtrado.sort_values('genres_translated')

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.boxplot(
                    data=df_filtrado,
                    x='vote_average',
                    y='genres_translated',
                    color='steelblue',
                    showmeans=True,
                    meanprops={
                        "marker": "o",
                        "markerfacecolor": "white",
                        "markeredgecolor": "black",
                        "markersize": 7
                    }
                )
                ax.axvline(media_global, color='red', linestyle='--', linewidth=2, label=f'Média Global: {media_global:.2f}')
                ax.set_xlabel("Nota Média", fontsize=12)
                ax.set_ylabel("Gênero", fontsize=12)
                ax.set_title("BoxPlot das Notas Médias por Gênero", fontsize=15, fontweight='bold')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Nenhum dado disponível para os gêneros selecionados.")

        plot_boxplot_notas_por_genero(df)


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

    elif selected_subtopic == "Nota média por Idioma":
        st.write("### nota média por idioma original")
        def plot_vote_average_by_language(df):
            receita_por_idioma = df.groupby('original_language')['vote_average'].mean().sort_values(ascending=False).reset_index()
            fig_lang = px.bar(receita_por_idioma, 
                            x='original_language', 
                            y='vote_average', 
                            labels={'original_language': 'Idioma Original', 'vote_average': 'Receita Média'},
                            color='vote_average',
                            color_continuous_scale='Blues')
            st.plotly_chart(fig_lang, use_container_width=True, config={"scrollZoom": True})

        plot_vote_average_by_language(df)

    elif selected_subtopic == "Nota média por Gênero":
        st.write("### Nota média por gênero cinematográfico")
        def plot_vote_average_by_genre(df_exploded):
            receita_por_genero = df_exploded.groupby('genre_names')['vote_average'].mean().sort_values(ascending=False).reset_index()
            fig_genre = px.bar(receita_por_genero, 
                            x='vote_average', 
                            y='genre_names', 
                            labels={'genre_names': 'Gênero', 'vote_average': 'Receita Média'},
                            color='vote_average',
                            color_continuous_scale='Teal')
            st.plotly_chart(fig_genre, use_container_width=True, config={"scrollZoom": True})

        plot_vote_average_by_genre(df_exploded)

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
    
    elif selected_subtopic == "Evolução da Nota Média ao Longo dos Anos":
        st.write("### Evolução da Nota Média ao Longo dos Anos")

        def plot_evolucao_nota_por_ano(df):
            df_ano = df.dropna(subset=['release_year', 'vote_average'])
            df_ano = df_ano.groupby('release_year')['vote_average'].mean().reset_index()
            fig = px.line(df_ano, x='release_year', y='vote_average', 
                        labels={'release_year': 'Ano de Lançamento', 'vote_average': 'Nota Média'},
                        markers=True)
            fig.update_layout(title='Média de Notas por Ano de Lançamento', xaxis_title='Ano', yaxis_title='Nota Média')
            st.plotly_chart(fig, use_container_width=True)
        
        plot_evolucao_nota_por_ano(df)

    elif selected_subtopic == "Nota Média por Faixa de Orçamento":
        st.write("### Nota Média por Faixa de Orçamento")

        # Definindo mais faixas
        bins = [0, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 3e8, df['budget'].max()]
        labels = ['Até 5M', '5M–10M', '10M–20M', '20M–50M', '50M–100M', '100M–200M', '200M–300M', '300M+']
        
        df['budget_range'] = pd.cut(df['budget'], bins=bins, labels=labels, include_lowest=True)

        faixa_df = df[df['budget_range'].notna()].groupby('budget_range')['vote_average'].mean().reset_index()
        faixa_df = faixa_df.sort_values('vote_average', ascending=False)

        fig_faixa = px.bar(
            faixa_df,
            x='budget_range',
            y='vote_average',
            color='vote_average',
            color_continuous_scale='Blues',
            labels={
                'vote_average': 'Nota Média',
                'budget_range': 'Faixa de Orçamento (USD)'
            }
        )
        fig_faixa.update_layout(xaxis_title='Faixa de Orçamento', yaxis_title='Nota Média')
        st.plotly_chart(fig_faixa, use_container_width=True)