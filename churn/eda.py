import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Histograms for numerical features ----------
def plot_histogram(df, column_name):
    """
    Plots histogram with KDE and marks mean & median.
    """
    plt.figure(figsize=(5, 3))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")

    col_mean = df[column_name].mean()
    col_median = df[column_name].median()

    plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
    plt.axvline(col_median, color="green", linestyle="-", label="Median")

    plt.legend()
    plt.show()

# ---------- Boxplot for numerical features ----------
def plot_boxplot(df, column_name):
    """
    Plots a simple boxplot for a numeric column.
    """
    plt.figure(figsize=(5, 3))
    sns.boxplot(y=df[column_name])
    plt.title(f"Box Plot of {column_name}")
    plt.ylabel(column_name)
    plt.show()

# ---------- Correlation heatmap ----------
def plot_correlation_heatmap(df):
    """
    Plots a heatmap for the correlation of numerical columns.
    """
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    plt.show()

# ---------- Countplots for categorical columns ----------
def plot_categorical_counts(df):
    """
    Plots countplots for all categorical columns in the dataframe.
    """
    object_cols = df.select_dtypes(include="object").columns.to_list()

    # SeniorCitizen is numeric, but in notebook it was treated as categorical
    if "SeniorCitizen" not in object_cols and "SeniorCitizen" in df.columns:
        object_cols = ["SeniorCitizen"] + object_cols

    for col in object_cols:
        plt.figure(figsize=(5, 3))
        sns.countplot(x=df[col])
        plt.title(f"Count Plot of {col}")
        plt.show()
