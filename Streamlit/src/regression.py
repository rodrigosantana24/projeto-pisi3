import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

def show_regression_report_table(y_true, y_pred):
    """Exibe métricas de regressão como tabela colorida no Streamlit."""
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    df_report = pd.DataFrame({
        "MSE": [mse],
        "R²": [r2]
    })
    styled = df_report.style.background_gradient(
        subset=['R²'], cmap='RdYlGn'
    ).background_gradient(
        subset=['MSE'], cmap='RdYlGn_r'
    ).format("{:.2f}")
    st.write("### Relatório de Regressão")
    st.dataframe(styled, use_container_width=True)

def run_regression(selected_topic):
    st.title("Modelos de Regressão")

    if selected_topic == "Resultados dos Modelos":
        # Dados dos modelos
        resultados = [
                        {"Modelo": "XGBRegressor", "R²": 0.497353, "MAE": 0.491978, "RMSE": 0.646522, "MedAE": 0.394464},
                        {"Modelo": "SVR", "R²": 0.454993, "MAE": 0.512859, "RMSE": 0.673214, "MedAE": 0.408498},
                        {"Modelo": "RandomForestRegressor", "R²": 0.450294, "MAE": 0.512854, "RMSE": 0.676110, "MedAE": 0.403060}
                    ]
        resultados2 = [
                        {"Modelo": "XGBRegressor", "R²": 0.385363, "MAE": 0.549527, "RMSE": 0.714926, "MedAE": 0.444526},
                        {"Modelo": "SVR", "R²": 0.347775, "MAE": 0.567254, "RMSE": 0.736463, "MedAE": 0.464352},
                        {"Modelo": "RandomForestRegressor", "R²": 0.351069, "MAE": 0.564488, "RMSE": 0.734601, "MedAE": 0.455613}
                    ]
        df_resultados1 = pd.DataFrame(resultados)
        df_resultados2 = pd.DataFrame(resultados2)
        styled1 = df_resultados1.style\
            .background_gradient(subset=['R²'], cmap='RdYlGn')\
            .background_gradient(subset=['MAE', 'RMSE', 'MedAE'], cmap='RdYlGn_r')\
            .format({
                'R²': '{:.6f}',
                'MAE': '{:.6f}',
                'RMSE': '{:.6f}',
                'MedAE': '{:.6f}'
            })
        styled2 = df_resultados2.style\
            .background_gradient(subset=['R²'], cmap='RdYlGn')\
            .background_gradient(subset=['MAE', 'RMSE', 'MedAE'], cmap='RdYlGn_r')\
            .format({
                'R²': '{:.6f}',
                'MAE': '{:.6f}',
                'RMSE': '{:.6f}',
                'MedAE': '{:.6f}'
            })
        st.write("### Tabela de Resultados dos Modelos COM pupularidade")
        st.dataframe(styled1, use_container_width=True)
        st.write("### Tabela de Resultados dos Modelos SEM pupularidade")
        st.dataframe(styled2, use_container_width=True)

    elif selected_topic == "Predição":
        st.header("Faça uma predição")

        # Carrega os objetos pickled para pegar as opções válidas
        with open('Streamlit/models/mlb_genres.pkl', 'rb') as f:
            mlb_genres = pickle.load(f)
        with open('Streamlit/models/mlb_production.pkl', 'rb') as f:
            mlb_prod = pickle.load(f)
        with open('Streamlit/models/xgb_pipeline.pkl', 'rb') as f:
            modelo = pickle.load(f)

        # Sugestões e opções
        genre_options = ["Drama", "Comedy", "Thriller", "Action", "Romance", "Adventure", "Crime", "Horror", "Science Fiction", "Fantasy", "Family", "Mystery", "History", "Animation", "War", "Music", "Western", "Documentary", "TV Movie"]
        prod_options = ['Universal Pictures', 'Warner Bros. Pictures', '20th Century Fox', 'Columbia Pictures', 'Paramount', 'Goldwyn', 'Metro', 'Mayer', 'New Line Cinema', 'Canal+', 'Walt Disney Pictures', 'United Artists', 'Lionsgate', 'Miramax', 'Touchstone Pictures', 'StudioCanal', 'Relativity Media', 'TriStar Pictures', 'DreamWorks Pictures', 'Summit Entertainment', 'Outras']
        lang_options = ['en', 'fr', 'hi', 'es', 'it', 'ru', 'ja', 'ko', 'zh', 'de', 'cn', 'pt', 'ta', 'da', 'fi', 'sv', 'no', 'pl', 'nl', 'tr', 'ml', 'th', 'ro', 'te', 'hu', 'he', 'sr', 'uk', 'ar', 'et', 'cs', 'id', 'fa', 'ca', 'is', 'el', 'kn', 'af', 'bo', 'ms', 'bs', 'la', 'kk', 'lv', 'ku', 'sh', 'tl', 'ps', 'lo', 'iu', 'bn', 'mr', 'ur']

        # Entradas do usuário com sugestões
        popularity = st.number_input(
            "Popularidade", min_value=0.0, value=None, placeholder="Ex: 120.3"
        )
        budget = st.number_input(
            "Orçamento", min_value=0.0, value=None, placeholder="Ex: 100000000"
        )
        runtime = st.number_input(
            "Tempo de execução (minutos)", min_value=70, value=None, placeholder="Deve ser maior que 70"
        )
        original_language = st.selectbox(
            "Idioma Original", options=lang_options, index=0, placeholder="Selecione o idioma"
        )
        release_date = st.text_input(
            "Data de lançamento", value="", placeholder="YYYY-MM-DD"
        )
        genres = st.multiselect(
            "Gêneros", options=genre_options, placeholder="Selecione um ou mais gêneros"
        )
        production_companies = st.multiselect(
            "Produtoras", options=prod_options, placeholder="Selecione uma ou mais produtoras"
        )

        # Só permite prever se todos os campos estiverem preenchidos
        campos_preenchidos = (
            popularity is not None and
            budget is not None and
            runtime is not None and
            original_language and
            release_date and
            genres and
            production_companies
        )

        if st.button("Prever", disabled=not campos_preenchidos):
            instancia = {
                'popularity': popularity,
                'budget': budget,
                'runtime': runtime,
                'original_language': original_language,
                'release_date': release_date,
                'genres': genres,
                'production_companies': production_companies
            }
            df = pd.DataFrame([instancia])

            # Mapeia gêneros desconhecidos para 'Outros'
            generos_validos = set(mlb_genres.classes_)
            df['genres'] = df['genres'].apply(
                lambda gs: [g if g in generos_validos else 'Outros' for g in gs]
            )

            # Mapeia produtoras desconhecidas para 'Outras'
            produtoras_validas = set(mlb_prod.classes_)
            df['production_companies'] = df['production_companies'].apply(
                lambda ps: [p if p in produtoras_validas else 'Outras' for p in ps]
            )


            # Aplica MultiLabelBinarizer
            df_gen = pd.DataFrame(
                mlb_genres.transform(df['genres']),
                columns=[f'genre_{cls}' for cls in mlb_genres.classes_],
                index=df.index
            )
            df_prod = pd.DataFrame(
                mlb_prod.transform(df['production_companies']),
                columns=[f'production_{cls}' for cls in mlb_prod.classes_],
                index=df.index
            )
            

            # Concatena e remove colunas originais
            df_final = pd.concat([
                df.drop(columns=['genres', 'production_companies']),
                df_gen, df_prod
            ], axis=1)
            
            

            # Reconstrua as colunas esperadas manualmente:
            colunas_esperadas = list(df.drop(columns=['genres', 'production_companies']).columns)
            colunas_esperadas += [f'genre_{cls}' for cls in mlb_genres.classes_]
            colunas_esperadas += [f'production_{cls}' for cls in mlb_prod.classes_]

            df_final = df_final.reindex(columns=colunas_esperadas, fill_value=0)

            

            # Faz a predição
            y_pred = modelo.predict(df_final)
            st.success(f"Predição do modelo: {y_pred[0]:.2f}")
