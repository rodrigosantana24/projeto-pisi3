import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

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
                        {"Modelo": "XGBRegressor", "R²": 0.509629, "MAE": 0.484806, "RMSE": 0.639296, "MedAE": 0.386420},
                        {"Modelo": "SVR", "R²": 0.450195, "MAE": 0.508508, "RMSE": 0.676930, "MedAE": 0.393708},
                        {"Modelo": "RandomForestRegressor", "R²": 0.476488, "MAE": 0.502198, "RMSE": 0.660546, "MedAE": 0.415460}
                    ]
        df_resultados = pd.DataFrame(resultados)
        styled = df_resultados.style\
            .background_gradient(subset=['R²'], cmap='RdYlGn')\
            .background_gradient(subset=['MAE', 'RMSE', 'MedAE'], cmap='RdYlGn_r')\
            .format({
                'R²': '{:.6f}',
                'MAE': '{:.6f}',
                'RMSE': '{:.6f}',
                'MedAE': '{:.6f}'
            })
        st.write("### Tabela de Resultados dos Modelos")
        st.dataframe(styled, use_container_width=True)

    elif selected_topic == "Teste dos modelos":
        st.write("Teste dos modelos - conteúdo a ser implementado.")
