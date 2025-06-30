def load_data(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    # Implement any necessary preprocessing steps here
    return df

def plot_histogram(data, column, bins=30):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=bins, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_boxplot(data, x, y):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=data, x=x, y=y)
    plt.title(f'Boxplot of {y} by {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()