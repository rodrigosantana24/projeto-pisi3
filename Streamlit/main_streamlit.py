# pip install streamlit pandas scikit-learn seaborn matplotlib numpy plotly statsmodels spicy
# streamlit run main_streamlit.py

import streamlit as st
from src.exploratory_analysis import run_exploratory_analysis
from src.regression import run_classification
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
            "BoxPlot por Gênero",
            "Histograma das Notas",
            "Matriz de Correlação",
            "Receita por Idioma",
            "Receita por Gênero",
            "Filtro por Nota e Gênero"
        ]
        selected_subtopic = st.sidebar.radio("Escolha um gráfico:", subtopics)
        run_exploratory_analysis(selected_subtopic)
    elif selected_tab == "Clusterização":
        run_clustering()

if __name__ == "__main__":
    main()