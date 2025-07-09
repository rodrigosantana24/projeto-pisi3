# ==============================================================================
# ARQUIVO PRINCIPAL: main_streamlit.py
# pip install streamlit pandas numpy joblib scikit-learn xgboost scipy matplotlib seaborn plotly statsmodels shap streamlit-tags streamlit_searchbox
# ==============================================================================

import streamlit as st
from src.exploratory_analysis import run_exploratory_analysis
from src.regression import run_regression
from src.clustering import run_clustering

def main():
    """
    Função principal que define a estrutura de navegação e executa a aba selecionada.
    """
    st.set_page_config(page_title="Movie App", layout="wide")
    
    st.sidebar.title("Navegação")
    selected_tab = st.sidebar.radio(
        "Escolha uma aba",
        ["Análises Exploratórias", "Regressão", "Clusterização"]
    )

    # --- Aba de Análises Exploratórias ---
    if selected_tab == "Análises Exploratórias":
        subtopics = [
            "Nota média por Gênero",
            "BoxPlot por Gênero",
            "Filtro por Nota e Gênero",
            "Histograma das Notas",
            "Nota média por Idioma",
            "Matriz de Correlação",
            "Evolução da Nota Média ao Longo dos Anos",
            "Nota Média por Faixa de Orçamento"
        ]
        selected_subtopic = st.sidebar.radio("Escolha um gráfico:", subtopics, key="exploratory_topic")
        run_exploratory_analysis(selected_subtopic)

    # --- Aba de Regressão ---
    elif selected_tab == "Regressão":
        regression_topics = [
            "Resultados dos Modelos",
            "Predição com popularidade",
            "Predição sem popularidade",
        ]
        selected_regression_topic = st.sidebar.radio("Escolha um tópico:", regression_topics, key="regression_topic")
        run_regression(selected_regression_topic)

    # --- Aba de Clusterização ---
    elif selected_tab == "Clusterização":
        clustering_topics = [
            "Método do Cotovelo",
            "PCA + KMeans + Silhueta",
            "Recomendação de Filmes"
        ]
        selected_clustering_topic = st.sidebar.radio("Escolha um tópico:", clustering_topics, key="clustering_topic")
        run_clustering(selected_clustering_topic)

if __name__ == "__main__":
    main()