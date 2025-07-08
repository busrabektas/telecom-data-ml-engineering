from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained models and scalers 
try:
    clustering_scaler = joblib.load("clustering/scaler.pkl")
    kmeans = joblib.load("clustering/kmeans_model.pkl")
    pca = joblib.load("clustering/pca_model.pkl")
    
    feature_scaler = joblib.load("models/feature_scaler.pkl")
    classifier = joblib.load("models/best_classification_model.pkl")
    regressor = joblib.load("models/best_regression_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    
    with open("feature_names.txt", "r") as f:
        REQUIRED_FEATURES = [line.strip() for line in f.readlines()]
    
    with open("clustering/cluster_features.txt", "r") as f:
        CLUSTERING_FEATURES = [line.strip() for line in f.readlines()]
    
    logger.info("All models loaded successfully")

except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# FastAPI app
app = FastAPI(title="Telecom ML Inference API", version="1.0.0")

# Pydantic input schema
class CustomerData(BaseModel):
    Age: float
    Gender: str
    Location: str
    Tenure: float
    ServiceType: str
    VoiceCallMinutes: float
    SMSsent: float
    IntlCallMinutes: float
    RoamingCharges: float
    NetworkIssues: float
    DroppedCalls: float
    SupportCalls: float
    DataSpeed: float
    Latency: float
    SatisfactionScore: float
    PaymentMethod: str
    PaymentHistory: float
    ServiceDowntime: float
    ContractRenewal: float
    Churn: float

def engineer_features(df):
    df = df.copy()
    df['CallsPerMonth'] = df['VoiceCallMinutes'] / (df['Tenure'] + 1)
    df['SMSPerMonth'] = df['SMSsent'] / (df['Tenure'] + 1)
    df['SupportCallsPerMonth'] = df['SupportCalls'] / (df['Tenure'] + 1)
    df['IssueRate'] = (df['NetworkIssues'] + df['DroppedCalls'] + df['SupportCalls']) / (df['Tenure'] + 1)
    df['LatencyRatio'] = df['Latency'] / (df['DataSpeed'] + 1)
    df['TechPerformance'] = df['DataSpeed'] / (df['Latency'] + 1)
    df['SupportToIssueRatio'] = df['SupportCalls'] / (1 + df['NetworkIssues'] + df['DroppedCalls'])
    df['AvgCallDuration'] = df['VoiceCallMinutes'] / (df['SupportCalls'] + 1)
    return df

def apply_one_hot_encoding(df):
    df = df.copy()
    
    # One-hot encoding
    one_hot_cols = ['Gender', 'ServiceType', 'PaymentMethod']
    for col in one_hot_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
    
    # Location: Label encoding
    if 'Location' in df.columns:
        df['Location_encoded'] = df['Location'].astype('category').cat.codes

    return df

def prepare_for_clustering(df):
    df_featured = engineer_features(df)
    clustering_data = df_featured[CLUSTERING_FEATURES].copy()
    clustering_data = clustering_data.fillna(clustering_data.median())
    return clustering_data

def prepare_for_supervised(df, pca1, pca2, cluster):
    df_featured = engineer_features(df)
    df_encoded = apply_one_hot_encoding(df_featured)
    
    df_encoded['PCA1'] = pca1
    df_encoded['PCA2'] = pca2
    df_encoded['Cluster'] = cluster

    # Fill in missing features with zero
    for feature in REQUIRED_FEATURES:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0.0

    X = df_encoded[REQUIRED_FEATURES].copy()
    X = X.fillna(X.median())
    return X

@app.post("/predict")
async def predict(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])
        logger.info("Processing prediction request...")

        # Clustering
        clustering_data = prepare_for_clustering(df)
        cluster_scaled = clustering_scaler.transform(clustering_data)
        cluster = int(kmeans.predict(cluster_scaled)[0])
        pca_result = pca.transform(cluster_scaled)
        pca1, pca2 = float(pca_result[0, 0]), float(pca_result[0, 1])

        # supervised
        supervised_data = prepare_for_supervised(df, pca1, pca2, cluster)
        supervised_scaled = feature_scaler.transform(supervised_data)

        # predictions
        plan_pred_encoded = classifier.predict(supervised_scaled)[0]
        plan_pred = label_encoder.inverse_transform([plan_pred_encoded])[0]

        data_usage_pred = float(regressor.predict(supervised_scaled)[0])
        classification_proba = classifier.predict_proba(supervised_scaled)[0].tolist()

        #response
        return {
            "PredictedPlanType": plan_pred,
            "PredictedDataUsageGB": round(max(0, data_usage_pred), 2),
            "CustomerSegment": cluster,
            "PCA_Components": {
                "PCA1": round(pca1, 4),
                "PCA2": round(pca2, 4)
            },
            "confidence_scores": {
                "classification_proba": classification_proba
            }
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Telecom ML Inference API is running!",
        "version": "1.0.0",
        "available_endpoints": ["/predict", "/health", "/model-info"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

@app.get("/model-info")
async def model_info():
    return {
        "clustering_features": CLUSTERING_FEATURES,
        "supervised_features": REQUIRED_FEATURES,
        "available_plan_types": label_encoder.classes_.tolist(),
        "feature_count": {
            "clustering": len(CLUSTERING_FEATURES),
            "supervised": len(REQUIRED_FEATURES)
        }
    }
