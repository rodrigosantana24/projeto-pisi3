# pip install streamlit pandas scikit-learn seaborn matplotlib numpy plotly statsmodels spicy
# streamlit run Streamlit/main_streamlit.py  

import streamlit as st
from src.exploratory_analysis import run_exploratory_analysis
from src.regression import run_regression
from src.clustering import run_clustering

def main():
    st.set_page_config(page_title="Movie App", layout="wide")
    st.sidebar.title("Navegação")
    selected_tab = st.sidebar.radio(
        "Escolha uma aba",
        ["Análises Exploratórias", "Regressão", "Clusterização"]
    )

    if selected_tab == "Análises Exploratórias":
        # Subtópicos para análises exploratórias
        subtopics = [
            "Nota média por Gênero",
            "BoxPlot por Gênero",
            "Filtro por Nota e Gênero",
            "Histograma das Notas",
            "Nota média por Idioma",
            "Matriz de Correlação"
        ]
        selected_subtopic = st.sidebar.radio("Escolha um gráfico:", subtopics)
        run_exploratory_analysis(selected_subtopic)
    elif selected_tab == "Regressão":
        regression_topics = [
            "Resultados dos Modelos",
            "Teste dos modelos"
        ]
        selected_regression_topic = st.sidebar.radio("Escolha um tópico:", regression_topics, key="regression_topic")
        run_regression(selected_regression_topic)
    elif selected_tab == "Clusterização":
        clustering_topics = [
            "Método do Cotovelo",
            "K-Means (1 valor de K)",
            "PCA + KMeans + Silhueta",
            "Recomendação de Filmes"
        ]
        selected_clustering_topic = st.sidebar.radio("Escolha um tópico:", clustering_topics, key="clustering_topic")
        run_clustering(selected_clustering_topic)

if __name__ == "__main__":
    main()