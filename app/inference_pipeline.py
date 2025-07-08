import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler

# === Load Models ===
scaler = joblib.load("clustering/scaler.pkl")
kmeans_model = joblib.load("clustering/kmeans_model.pkl")
pca_model = joblib.load("clustering/pca_model.pkl")
feature_scaler = joblib.load("feature_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
clf_model = joblib.load("best_classification_model.pkl")
reg_model = joblib.load("best_regression_model.pkl")

# === Load Feature Names ===
with open("feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

with open("clustering/cluster_features.txt", "r") as f:
    cluster_features = [line.strip() for line in f.readlines()]

# === Load Small Dataset ===
df = pd.read_csv("../data/cleaned_small.csv")

# === Numeric preprocessing ===
numeric_cols = [
    "Age", "Tenure", "VoiceCallMinutes", "SMSsent", "DataUsageGB", "IntlCallMinutes",
    "RoamingCharges", "NetworkIssues", "DroppedCalls", "SupportCalls", "DataSpeed",
    "Latency", "SatisfactionScore", "PaymentHistory", "ServiceDowntime", "ContractRenewal", "Churn"
]

for col in numeric_cols:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# === Feature Engineering (Unsupervised) ===
df['CallsPerMonth'] = df['VoiceCallMinutes'] / (df['Tenure'] + 1)
df['SMSPerMonth'] = df['SMSsent'] / (df['Tenure'] + 1)
df['SupportCallsPerMonth'] = df['SupportCalls'] / (df['Tenure'] + 1)
df['IssueRate'] = (df['NetworkIssues'] + df['DroppedCalls'] + df['SupportCalls']) / (df['Tenure'] + 1)
df['LatencyRatio'] = df['Latency'] / (df['DataSpeed'] + 1)
df['TechPerformance'] = df['DataSpeed'] / (df['Latency'] + 1)
df['SupportToIssueRatio'] = df['SupportCalls'] / (1 + df['NetworkIssues'] + df['DroppedCalls'])
df['AvgCallDuration'] = df['VoiceCallMinutes'] / (df['SupportCalls'] + 1)

# === Cluster Assignment ===
df[cluster_features] = df[cluster_features].fillna(df[cluster_features].median())
cluster_scaled = scaler.transform(df[cluster_features])
df['Cluster'] = kmeans_model.predict(cluster_scaled)
pca_result = pca_model.transform(cluster_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# === Categorical One-Hot Encoding (Match training!) ===
one_hot_cols = ['Gender', 'ServiceType', 'PaymentMethod']
for col in one_hot_cols:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)

# Label encode 'Location' like in training
if 'Location' in df.columns:
    df['Location_encoded'] = df['Location'].astype('category').cat.codes

# === Feature Alignment ===
for col in feature_names:
    if col not in df.columns:
        df[col] = 0  # Eksik feature varsa sıfırla doldur

X = df[feature_names].copy()
X = X.fillna(X.median())
X_scaled = feature_scaler.transform(X)

# === Classification Prediction ===
y_pred_class = clf_model.predict(X_scaled)
df['PredictedPlan'] = label_encoder.inverse_transform(y_pred_class)

# === Regression Prediction ===
df['PredictedDataUsageGB'] = reg_model.predict(X_scaled)

# === Save Output ===
output_path = "deployment/predicted_small_dataset.csv"
os.makedirs("deployment", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"[✅] Tahmin sonuçları kaydedildi: {output_path}")
