# ==============================================================================
# ARQUIVO FINAL v4: src/regression.py
# (Removida a função duplicada que causava o erro de 'key')
# ==============================================================================

# --- 1. IMPORTAÇÕES NECESSÁRIAS ---
import streamlit as st
import pandas as pd
import joblib
import sys
import shap
from streamlit_tags import st_tags
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback 

import src.custom_transformers


# --- CONSTANTES DE COLUNAS DE TREINO ---
TRAINING_COLUMNS_POP = ['genres', 'original_language', 'popularity', 'production_companies', 'release_date', 'budget', 'runtime', 'credits']
TRAINING_COLUMNS_NO_POP = ['genres', 'original_language', 'production_companies', 'release_date', 'budget', 'runtime', 'credits']


# --- 2. FUNÇÃO AUXILIAR PARA CONSTRUIR A INTERFACE DE PREDIÇÃO E ANÁLISE ---
def _build_prediction_interface(modelo, show_popularity_input, training_columns):
    """
    Constrói a interface de predição e gera o gráfico SHAP de ranking (waterfall).
    """
    st.write("Preencha os campos abaixo para prever a nota de avaliação de um filme.")
    key_prefix = "pop" if show_popularity_input else "no_pop"
    
    # Definições de opções para os widgets
    generos_map = {"Ação": "Action", "Animação": "Animation", "Aventura": "Adventure", "Comédia": "Comedy", "Crime": "Crime", "Drama": "Drama", "Família": "Family", "Fantasia": "Fantasy", "Faroeste": "Western", "Ficção Científica": "Science Fiction", "Guerra": "War", "História": "History", "Mistério": "Mystery", "Música": "Music", "Romance": "Romance", "Terror": "Horror", "Thriller": "Thriller"}
    genre_options_pt = sorted(list(generos_map.keys()))
    prod_options = sorted(['Warner Bros. Pictures', 'Universal Pictures', '20th Century Fox', 'Columbia Pictures', 'Paramount', 'Walt Disney Pictures', 'New Line Cinema', 'Lionsgate', 'Miramax', 'StudioCanal'])
    lang_options = sorted(['en', 'fr', 'hi', 'es', 'it', 'ru', 'ja', 'ko', 'zh', 'de', 'cn', 'pt', 'ta', 'da', 'fi', 'sv', 'no', 'pl', 'nl', 'tr'])
    credits_options = sorted(['Samuel L. Jackson', 'Robert De Niro', 'Nicolas Cage', 'Bruce Willis', 'Morgan Freeman', 'Matt Damon', 'Liam Neeson', 'Johnny Depp', 'Tom Hanks'])
    
    # Criação da interface
    st.subheader("Insira as Informações do Filme")
    col1, col2 = st.columns(2)
    
    with col1:
        if show_popularity_input:
            popularity = st.number_input("Popularidade", min_value=0.0, value=None, placeholder="Ex: 120.3", key=f"{key_prefix}_popularity")
        budget = st.number_input("Orçamento (em $)", min_value=0, value=None, placeholder="Ex: 150000000", key=f"{key_prefix}_budget")
        runtime = st.number_input("Duração (em minutos)", min_value=1, value=None, placeholder="Ex: 140", key=f"{key_prefix}_runtime")
        original_language = st.selectbox("Idioma Original", options=lang_options, index=None, placeholder="Selecione o idioma", key=f"{key_prefix}_language")
        release_date = st.date_input("Data de Lançamento", value=None, format="DD/MM/YYYY", key=f"{key_prefix}_release_date")

    with col2:
        genres_selecionados_pt = st.multiselect("Gêneros", options=genre_options_pt, placeholder="Selecione os gêneros", key=f"{key_prefix}_genres")
        final_production_companies = st_tags(
            label='Companhias de Produção:', text='Pressione Enter para adicionar', value=[],
            suggestions=prod_options, key=f'{key_prefix}_prod_tags'
        )
        final_credits = st_tags(
            label='Créditos Principais (Elenco/Diretor):', text='Pressione Enter para adicionar', value=[],
            suggestions=credits_options, key=f'{key_prefix}_credits_tags'
        )

    if st.button("📊 Prever Nota do Filme", type="primary", use_container_width=True, key=f"{key_prefix}_predict_button"):
        
        # Coleta de dados
        input_row = {'budget': budget, 'runtime': runtime, 'original_language': original_language, 'release_date': pd.to_datetime(release_date, errors='coerce') if release_date else None, 'credits': final_credits, 'genres': [generos_map.get(g, g) for g in genres_selecionados_pt], 'production_companies': final_production_companies}
        if show_popularity_input:
            # A variável 'popularity' só existe neste escopo, então a coletamos aqui
            input_row['popularity'] = popularity
        
        input_df = pd.DataFrame([input_row])
        input_df = input_df.reindex(columns=training_columns)

        try:
            pipeline_final = modelo
            with st.spinner('Analisando dados e fazendo a previsão...'):
                prediction = pipeline_final.predict(input_df)
                predicted_score = round(prediction[0], 2)
                st.success(f"## A nota média prevista para o filme é: **{predicted_score}** ⭐")

            st.subheader("Ranking das Características Mais Impactantes (SHAP)")
            st.info("Em **vermelho**, as features que **aumentaram** a nota. Em **azul**, as que **diminuíram**.")

            with st.spinner('Gerando análise de importância...'):
                plt.style.use('dark_background')
                regressor = pipeline_final.named_steps['regressor']
                preprocessing_pipeline = Pipeline(pipeline_final.steps[:-1])
                feature_names = preprocessing_pipeline.get_feature_names_out(input_features=input_df.columns)
                transformed_df = preprocessing_pipeline.transform(input_df)
                transformed_df_pandas = pd.DataFrame(transformed_df, columns=feature_names)
                explainer = shap.TreeExplainer(regressor)
                shap_values = explainer.shap_values(transformed_df_pandas)
                
                # Plot Waterfall (Ranking)
                fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 8), dpi=150)
                shap_explanation = shap.Explanation(
                    values=shap_values[0], base_values=explainer.expected_value,
                    data=transformed_df_pandas.iloc[0], feature_names=transformed_df_pandas.columns
                )
                shap.plots.waterfall(shap_explanation, max_display=15, show=False)
                st.pyplot(fig_waterfall, bbox_inches='tight', use_container_width=True)
                plt.close(fig_waterfall)

        except Exception as e:
            # Mantemos o bloco try-except para capturar outros possíveis erros no futuro
            st.error(f"Ocorreu um erro durante a predição ou análise SHAP.")
            st.error(f"**Detalhe do Erro:** {e}")
            st.write("---")
            st.write("**Traceback completo do erro (para depuração):**")
            tb_str = traceback.format_exc()
            st.code(tb_str, language='text')

# --- 3. FUNÇÃO PRINCIPAL PARA A ABA DE REGRESSÃO ---
def run_regression(selected_topic):
    st.title("Modelos de Regressão")
    sys.modules["custom_transformers"] = src.custom_transformers

    if selected_topic == "Resultados dos Modelos":
        st.header("Comparativo de Performance dos Modelos")
        # (código das tabelas de resultados, que já está correto)
        resultados_com_pop = [{"Modelo": "XGBRegressor", "R²": 0.497353, "MAE": 0.491978, "RMSE": 0.646522, "MedAE": 0.394464}, {"Modelo": "SVR", "R²": 0.454993, "MAE": 0.512859, "RMSE": 0.673214, "MedAE": 0.408498}, {"Modelo": "RandomForestRegressor", "R²": 0.450294, "MAE": 0.512854, "RMSE": 0.676110, "MedAE": 0.403060}]
        resultados_sem_pop = [{"Modelo": "XGBRegressor", "R²": 0.385363, "MAE": 0.549527, "RMSE": 0.714926, "MedAE": 0.444526}, {"Modelo": "SVR", "R²": 0.347775, "MAE": 0.567254, "RMSE": 0.736463, "MedAE": 0.464352}, {"Modelo": "RandomForestRegressor", "R²": 0.351069, "MAE": 0.564488, "RMSE": 0.734601, "MedAE": 0.455613}]
        df_resultados1 = pd.DataFrame(resultados_com_pop)
        df_resultados2 = pd.DataFrame(resultados_sem_pop)
        numeric_formatter = {'R²': '{:.2f}', 'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'MedAE': '{:.2f}'}
        styled1 = df_resultados1.style.background_gradient(subset=['R²'], cmap='RdYlGn').background_gradient(subset=['MAE', 'RMSE', 'MedAE'], cmap='RdYlGn_r').format(numeric_formatter)
        styled2 = df_resultados2.style.background_gradient(subset=['R²'], cmap='RdYlGn').background_gradient(subset=['MAE', 'RMSE', 'MedAE'], cmap='RdYlGn_r').format(numeric_formatter)
        st.write("### Tabela de Resultados (COM popularidade)")
        st.dataframe(styled1, use_container_width=True)
        st.write("### Tabela de Resultados (SEM popularidade)")
        st.dataframe(styled2, use_container_width=True)
        st.info("**Métricas:** R² (quanto maior, melhor). MAE, RMSE, MedAE (quanto menor, melhor).")
    
    elif selected_topic in ["Predição com popularidade", "Predição sem popularidade"]:
        training_cols = None
        if selected_topic == "Predição com popularidade":
            header_text = "Predição de Nota (Usando Modelo com Popularidade)"
            model_path = "Streamlit/models/best_model.pkl"
            show_pop = True
            training_cols = TRAINING_COLUMNS_POP
        
        elif selected_topic == "Predição sem popularidade":
            header_text = "Predição de Nota (Usando Modelo sem Popularidade)"
            model_path = "Streamlit/models/best_model_sem_popularidade.pkl"
            show_pop = False
            training_cols = TRAINING_COLUMNS_NO_POP

        st.header(header_text)
        try:
            modelo = joblib.load(model_path)
            # Passa todos os parâmetros necessários para a função da interface
            _build_prediction_interface(modelo, show_pop, training_cols)
        except FileNotFoundError:
            st.error(f"**Erro:** Modelo não encontrado em `{model_path}`.")
        except Exception as e:
            # O erro de 'key' duplicada acontecia aqui, ao tentar carregar a interface
            st.error(f"Ocorreu um erro inesperado ao carregar o modelo: {e}")
            st.code(traceback.format_exc(), language='text')