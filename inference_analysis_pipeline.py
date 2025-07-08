import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# load data
df = pd.read_csv("deployment/predicted_small_dataset.csv")

# classification analysis 
if "PlanType" in df.columns and "PredictedPlan" in df.columns:
    print("\n[CLASSIFICATION REPORT]")

    #cleaning
    df = df.dropna(subset=["PlanType", "PredictedPlan"])  
    df["PlanType"] = df["PlanType"].astype(str)
    df["PredictedPlan"] = df["PredictedPlan"].astype(str)

    print(classification_report(df["PlanType"], df["PredictedPlan"]))

    unique_labels = sorted(df["PlanType"].unique())
    cm = confusion_matrix(df["PlanType"], df["PredictedPlan"], labels=unique_labels)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=unique_labels, yticklabels=unique_labels, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("deployment/confusion_matrix.png", dpi=300)
    plt.show()

# regression analysis
if "PredictedDataUsageGB" in df.columns and "DataUsageGB" in df.columns:
    print("\nREGRESSION PERFORMANCE")
    mse = mean_squared_error(df["DataUsageGB"], df["PredictedDataUsageGB"])
    rmse = mse ** 0.5
    r2 = r2_score(df["DataUsageGB"], df["PredictedDataUsageGB"])
    print(f"MSE: {mse:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

    # Gerçek vs Tahmin Scatter
    plt.figure(figsize=(6,6))
    sns.scatterplot(x="DataUsageGB", y="PredictedDataUsageGB", data=df, alpha=0.3)
    plt.plot([df["DataUsageGB"].min(), df["DataUsageGB"].max()], [df["DataUsageGB"].min(), df["DataUsageGB"].max()], '--', color='red')
    plt.xlabel("Actual DataUsageGB")
    plt.ylabel("Predicted DataUsageGB")
    plt.title("Actual vs Predicted Data Usage")
    plt.tight_layout()
    plt.savefig("deployment/datausage_scatter.png", dpi=300)
    plt.show()

# 4. Cluster bazlı doğruluk
if "Cluster" in df.columns:
    print("\nCLUSTER-BASED ANALYSIS")
    cluster_accuracy = df.groupby("Cluster").apply(
        lambda g: (g["PlanType"] == g["PredictedPlan"]).mean()
    )
    cluster_accuracy.plot(kind="bar", title="Accuracy by Cluster", ylabel="Accuracy", xlabel="Cluster ID", color='skyblue')
    plt.tight_layout()
    plt.savefig("deployment/cluster_accuracy.png", dpi=300)
    plt.show()
