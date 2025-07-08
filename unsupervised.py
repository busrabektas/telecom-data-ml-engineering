import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Read data
df = pd.read_csv("cleaned_large.csv")

# Feature Engineering
df['CallsPerMonth'] = df['VoiceCallMinutes'] / (df['Tenure'] + 1)
df['SMSPerMonth'] = df['SMSsent'] / (df['Tenure'] + 1)
df['SupportCallsPerMonth'] = df['SupportCalls'] / (df['Tenure'] + 1)
df['IssueRate'] = (df['NetworkIssues'] + df['DroppedCalls'] + df['SupportCalls']) / (df['Tenure'] + 1)
df['LatencyRatio'] = df['Latency'] / (df['DataSpeed'] + 1)
df['TechPerformance'] = df['DataSpeed'] / (df['Latency'] + 1)

df['SupportToIssueRatio'] = df['SupportCalls'] / (1 + df['NetworkIssues'] + df['DroppedCalls'])
df['AvgCallDuration'] = df['VoiceCallMinutes'] / (df['SupportCalls'] + 1)


features = [
    'IssueRate', 'CallsPerMonth',
    'SMSPerMonth', 'SupportCallsPerMonth',
    'LatencyRatio', 'TechPerformance',
    'SupportToIssueRatio','AvgCallDuration',
    'Tenure', 'Age' 
]


df[features] = df[features].fillna(df[features].median())

scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[features])
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)


sample_idx = np.random.choice(len(df_scaled), size=10000, replace=False)
sil_score = silhouette_score(df_scaled[sample_idx], df['Cluster'].iloc[sample_idx])
calinski = calinski_harabasz_score(df_scaled[sample_idx], df['Cluster'].iloc[sample_idx])
davies = davies_bouldin_score(df_scaled[sample_idx], df['Cluster'].iloc[sample_idx])

print(f"[INFO] Clustering metrics:\n- Silhouette Score: {sil_score:.4f}\n- Calinski-Harabasz Score: {calinski:.2f}\n- Davies-Bouldin Score: {davies:.4f}")

# PCA plot
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

plt.figure(figsize=(10, 6))
for cluster in sorted(df['Cluster'].unique()):
    plt.scatter(df[df['Cluster'] == cluster]['PCA1'], df[df['Cluster'] == cluster]['PCA2'], label=f"Cluster {cluster}", alpha=0.6)
plt.title("Customer Segments (PCA)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("clustering/pca_clusters_enhanced.png")
plt.show()

# anomaly detection
k_distances = kmeans.transform(df_scaled).min(axis=1)
thresh = np.quantile(k_distances, 0.99)
df['DistanceFromCenter'] = k_distances
anomalies = df[df['DistanceFromCenter'] > thresh]
print(f"Anomalies detected: {len(anomalies)}")

# filter anomaly
df_clean = df[df['DistanceFromCenter'] <= thresh].copy()
df_clean_scaled = scaler.fit_transform(df_clean[features])
kmeans_clean = KMeans(n_clusters=5, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans_clean.fit_predict(df_clean_scaled)

# new scores
i_clean = np.random.choice(len(df_clean_scaled), size=10000, replace=False)
sil_clean = silhouette_score(df_clean_scaled[i_clean], df_clean['Cluster'].iloc[i_clean])
print(f"New Silhouette Score (clean): {sil_clean:.4f}")

grouped = df_clean.groupby('Cluster')[features].mean()
grouped.to_csv("clustering/cluster_summary_enhanced.csv")

#Cluster profiles
def describe_profiles(df, cluster_col='Cluster'):
    profiles = {}
    for cluster in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster]
        profiles[f"Cluster_{cluster}"] = {
            'size': len(cluster_data),
            'percentage': 100 * len(cluster_data) / len(df),
            'avg_tenure': cluster_data['Tenure'].mean(),
            'avg_satisfaction': cluster_data['SatisfactionScore'].mean(),
            'avg_data_usage': cluster_data['DataUsageGB'].mean()        }
    return profiles

profiles = describe_profiles(df_clean)
for name, prof in profiles.items():
    print(f"\n{name}:")
    for key, val in prof.items():
        print(f"  - {key}: {val:.2f}" if isinstance(val, float) else f"  - {key}: {val}")


anomalies.to_csv("clustering/detected_anomalies_enhanced.csv", index=False)
df_clean.to_csv("clustering/segmented_customers_enhanced.csv", index=False)
print("[segmentation pipleine completed")

# cleaned data pca 
pca_clean = PCA(n_components=2)
pca_result_clean = pca_clean.fit_transform(df_clean_scaled)
df_clean['PCA1'] = pca_result_clean[:, 0]
df_clean['PCA2'] = pca_result_clean[:, 1]

plt.figure(figsize=(10, 6))
for cluster in sorted(df_clean['Cluster'].unique()):
    plt.scatter(
        df_clean[df_clean['Cluster'] == cluster]['PCA1'],
        df_clean[df_clean['Cluster'] == cluster]['PCA2'],
        label=f"Cluster {cluster}", alpha=0.6
    )
plt.title("Customer Segments after Removing Anomalies (PCA)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("clustering/pca_clusters_cleaned.png")
plt.show()



joblib.dump(scaler, "clustering/scaler.pkl")
joblib.dump(kmeans_clean, "clustering/kmeans_model.pkl")
joblib.dump(pca_clean, "clustering/pca_model.pkl")
print(" scaler.pkl, kmeans_model.pkl, pca_model.pkl saved.")

# save cluster features
with open("clustering/cluster_features.txt", "w") as f:
    for feat in features:
        f.write(f"{feat}\n")
print("cluster_features.txt saved.")
