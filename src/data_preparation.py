import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Data Import
filepath = 'data/ObesityDataSet_raw_and_data_sinthetic.csv'

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Task 2: Encoding
def encode_binary_variables(df):
    le = LabelEncoder()
    binary_cols = ['Gender', 'SMOKE', 'family_history_with_overweight', 'FAVC']
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    return df

def encode_categorical_variables(df):
    categorical_cols = ['MTRANS', 'NObeyesdad']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    return df_encoded

# Task 3: Outlier Detection
def detect_outliers(df, columns):
    plt.figure(figsize=(12, 6))
    df[columns].boxplot()
    plt.xticks(rotation=45)
    plt.title('Outlier Detection using Boxplots')
    plt.show()

# Task 4: Feature Scaling
def scale_features(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

if __name__ == "__main__":
    
    df = load_data('data/ObesityDataSet_raw_and_data_sinthetic.csv')
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData Info:")
    print(df.info())