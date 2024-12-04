import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_summary_statistics(df):
    """Task 1: Generate summary statistics"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary_stats = df[numeric_cols].describe()
    return summary_stats

def plot_distributions(df):
    """Task 2: Distribution Analysis"""
    continuous_vars = ['Age', 'Weight', 'Height']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, var in enumerate(continuous_vars):
        sns.histplot(data=df, x=var, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {var}')
    plt.tight_layout()
    plt.show()

def explore_relationships(df):
    """Task 3: Relationship Exploration"""
    features = ['Weight', 'FCVC', 'NCP']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, feature in enumerate(features):
        sns.boxplot(data=df, x='NObeyesdad', y=feature, ax=axes[i])
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].set_title(f'{feature} by Obesity Level')
    plt.tight_layout()
    plt.show()

def create_correlation_heatmap(df):
    """Task 4: Correlation Analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')
    
    # Execute EDA tasks
    print("Summary Statistics:")
    print(generate_summary_statistics(df))
    
    plot_distributions(df)
    explore_relationships(df)
    create_correlation_heatmap(df)
