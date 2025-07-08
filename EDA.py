import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# load data
df = pd.read_csv("cleaned_large.csv")

# 2. Sayısal kolonları belirle
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Çıktılar klasörü oluştur
os.makedirs("eda_outputs", exist_ok=True)

# === 1. VARYANS ANALİZİ ===
def plot_variance(df, numeric_cols):
    variances = df[numeric_cols].var().sort_values(ascending=False)
    
    print("\nColumns with the highest variance:")
    print(variances.head(10))
    
    # Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x=variances.head(10).index, y=variances.head(10).values)
    plt.title("En Yüksek Varyansa Sahip Özellikler")
    plt.xticks(rotation=45)
    plt.ylabel("Varyans")
    plt.tight_layout()
    plt.savefig("eda_outputs/raw_feature_variance.png", dpi=300)
    plt.close()

# === 2. KORELASYON ISI HARİTASI ===
def plot_correlation_heatmap(df, numeric_cols):
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_outputs/feature_correlation_heatmap.png", dpi=300)
    plt.close()

# === 3. PCA ANALİZİ ===
def pca_analysis(df, numeric_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    
    pca = PCA()
    pca.fit(scaled)
    
    loadings = pd.DataFrame(pca.components_.T, index=numeric_cols, columns=[f"PC{i+1}" for i in range(len(numeric_cols))])
    explained_variance = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(len(numeric_cols))])
    
    print("\nThe columns that contribute the most to the first Principal Component are:")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
    
    # PCA açıklanan varyans grafiği
    plt.figure(figsize=(8, 4))
    explained_variance.cumsum().plot(marker='o')
    plt.title("PCA Cumulative Explained Variance")
    plt.ylabel("Cumulative Variance Ratio")
    plt.xlabel("Principal Components")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eda_outputs/pca_cumulative_variance.png", dpi=300)
    plt.close()
    
    return loadings

plot_variance(df, numeric_cols)

print("\nCorrelation heatmap created")
plot_correlation_heatmap(df, numeric_cols)


pca_loadings = pca_analysis(df, numeric_cols)

pca_loadings.to_csv("eda_outputs/pca_feature_loadings.csv")
print("\nEDA completed")
