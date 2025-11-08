# Projeto PISI - Sistema de previsão da nota média dos filmes do TMDB e análise de clusters com K-Means para recomendação

Este projeto foi desenvolvido como parte da disciplina de Projeto Interdisciplinar de Sistemas de Informação III (PISI III), com o objetivo de aplicar conceitos de **Machine Learning (ML)** em uma base de dados real.

---

## 🎬 Descrição

Este projeto tem como objetivo analisar e prever a **nota média de filmes** com base em variáveis disponíveis no dataset TMDB, além de **agrupar filmes similares** utilizando **K-Means**, com o intuito de gerar insights e recomendações.  
O dataset utilizado foi obtido da plataforma Kaggle, contendo milhões de registros de filmes com informações como gênero, orçamento, receita, empresas produtoras, idioma, entre outros.

---

## 🧠 Objetivos

- Realizar **análise exploratória** para entender padrões nos dados de filmes.
- Construir **modelos de regressão** para prever a nota média (`vote_average`) dos filmes.
- Utilizar **K-Means** para **clusterização de filmes** com características semelhantes.
- Gerar **recomendações simples** com base na similaridade entre os clusters.
- Avaliar a performance dos modelos com métricas apropriadas.

---

## 📁 Dataset

- **Fonte**: [Millions of Movies - Kaggle](https://www.kaggle.com/datasets/akshaypawar7/millions-of-movies)
- **Formato**: CSV
- **Tamanho**: ~350MB

---

## 💻 Tecnologias Utilizadas

- 🧰 **Ferramentas de Desenvolvimento**: `Python`, `Jupyter`, `Colab`
- 📊 **Análise de Dados**: `pandas`, `numpy`, `scikit-learn`
- 📈 **Visualização de Dados**: `streamlit`, `matplotlib`, `seaborn`, `plotly`
- 🧠 **Modelos de Machine Learning**: `XGBRegressor`, `SVR`, `RandomForestRegressor`, `KMeans`

---

## 📂 Mapeamento de Etapas e Arquivos

| Etapa                                               | Arquivo / Caminho                                                                                          |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **Tratamento de dados para regressão**              | `Regressao/Tratamento_Regressao.ipynb`                                                                      |
| **Treinamento modelos de regressão**                | `Regressao/Main_Regressao.ipynb`  
| **Treinamento sem variável popularidade**           | `Regressao/Main_Regressao_sem_popularidade.ipynb`                                                           |
| **Pré‑processamento de clusterização**              | `Clusterizacao/tratamento_clusterizacao_victor.ipynb`  
| **Pré‑processamento (alternativo)**                 | `Clusterizacao/tratamento_thalyson.ipynb`                                                                  |
| **Análise exploratória de clusterização**           | `Clusterizacao/Analise_DataSetMovies_victor.ipynb`                                                         |
| **Treinamento de K‑Means (clusterização principal)**| `Clusterizacao/clusterizacao_main.ipynb`                                                                   |
| **Interpretação de clusters**                       | `Clusterizacao/InterpretacaoModelos_victor.ipynb`                                                          |
| **Transformers e pipelines (código)**               | `Streamlit/src/custom_transformers.py`                                                                     |
| **Análise exploratória (código)**                   | `Streamlit/src/exploratory_analysis.py`                                                                     |
| **Regras de regressão (código)**                    | `Streamlit/src/regression.py`                                                                              |
| **Rotina de clusterização (código)**                | `Streamlit/src/clustering.py`                                                                              |
| **Aplicação web Streamlit**                         | `Streamlit/src/main_streamlit.py`                                                                           |


## 📓 Notebooks Google Colab adicionais

Abaixo estão os notebooks responsáveis por diferentes etapas do projeto:

| Colabs                                    |
|-------------------------------------------|
| [🔧 Analise_DataSetMovies](https://colab.research.google.com/drive/109EHdbLthtJOpstcyVTnPFyM0ErTXTU3?usp=sharing) |
| [📊 InterpretacaoModelos](https://colab.research.google.com/drive/1WBTnzIWG8app8htbJPi5V6T1hP9zmYEb?usp=sharing) |
| [🧠 Cluesterização](https://colab.research.google.com/drive/1T3NQ3zO1Bm8L8MfojxCSfo89Tz-BRzn1?usp=sharing) |


## 👨‍💻 Desenvolvedores

| Nome                | E-mail                     |
|---------------------|-----------------------------|
| Gabriel Vinícius     | gabrielvto18@gmail.com   |
| Luan Vinícius     | limaluan32104@gmail.com   |
| Rodrigo Santana     | rodrigosantana.dev@gmail.com   |
| Thalyson Kauan    | thalyson.kauan7@gmail.com |
| Victor de Souza     | victorsouza183@gmail.com   |

## 🚀 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/rodrigosantana24/projeto-pisi3
   cd PISI3
   ```

2. Crie e ative o ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate.bat  # Windows
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute a aplicação:
   ```bash
   streamlit run Streamlit/main_streamlit.py
   ```

