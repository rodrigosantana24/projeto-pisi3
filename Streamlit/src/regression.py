import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
from custom_transformers import (
    DateFeatureExtractor,
    CapTransformer,
    RareCategoryGrouper,
    TopNMultiLabelTransformer,
    BudgetRuntimeRatioTransformer
)

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

        with open('Streamlit/models/best_model.pkl', 'rb') as f:
            modelo = pickle.load(f)
            