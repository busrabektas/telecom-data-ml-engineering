# Telecom Customer Segmentation and Prediction

This project demonstrates a complete **ML engineering pipeline** for a telecom company using real-world-like customer data. It covers:

- Data preprocessing  
- Unsupervised segmentation  
- Anomaly detection  
- Supervised modeling (classification + regression)  
- Model deployment with FastAPI and Docker  



---

## Project Structure

## Objectives

- Preprocess telecom customer data
- Create customer segments using K-Means clustering
- Detect anomalies in customer behavior
- Build models to predict:
  - `DataUsageGB` (regression)
  - `PlanType` (classification)
- Serve predictions via a FastAPI endpoint
- Visualize clustering and model performance

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
````
### 2. Preprocess Data

```bash
python preprocess_main.py

````

### 3. Train models:

```bash
python app/models/supervised.py
````
```bash
python app/models/unsupervised.py
````
### 4 Batch Inference:

```bash
python app/models/inference_pipeline.py
````
```bash
python app/models/inference_analyis_pipeline.py
````

### 5 Serve API with Docker

```bash
#build the docker image 
docker build -t telecom-inference .
````
```bash
#run the container
docker run -p 8000:8000 telecom-inference
````


#### Once the container is running, open your browser and go to: http://localhost:8000/docs

This interactive interface allows you to:

Send new customer data via /predict endpoint

Receive predictions for:

- DataUsageGB (regression output)

- PlanType (classification output)

- Customer segment (cluster ID)

### Example POST Request

Wrong PlanType prediction (Original PlanType: Premium DataUsage:34)
![alt text](<images/Ekran Resmi 2025-07-08 12.29.04.png>)

Correct PlanType Prediction (Original PlanType:Basic, DataUsage:34)
![ ](<images/Ekran Resmi 2025-07-08 12.26.45.png>)

### Evaluation Metrics
Clustering
- Silhouette Score

- Calinski-Harabasz Index

- Davies-Bouldin Score

Classification
- Accuracy
- Confusion Matrix
- ROC-AUC

Regression
- Mean Squared Error (MSE)

- R² Score

- Actual vs Predicted Scatter

## Visualizations
Visual outputs for analysis are saved in:

- clustering/: PCA plots, cluster summaries, anomalies

- models/: Confusion matrix, ROC curves, feature importance

- deployment/: Final prediction scatterplots and performance

- eda_outputs/: Correlation heatmaps, PCA loadings, variances