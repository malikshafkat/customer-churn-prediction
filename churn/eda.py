import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_summary(df):
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nClass Distribution:\n", df["Churn"].value_counts())

def plot_distributions(df, num_cols):
    for col in num_cols:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col], kde=True)
        plt.axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
        plt.axvline(df[col].median(), color='green', linestyle='-', label='Median')
        plt.title(f"Distribution of {col}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_boxplots(df, num_cols):
    for col in num_cols:
        plt.figure(figsize=(4, 2.5))
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

def correlation_heatmap(df, cols):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_categorical_counts(df, cat_cols):
    for col in cat_cols:
        plt.figure(figsize=(5, 3))
        sns.countplot(x=col, data=df)
        plt.title(f"Count Plot of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
