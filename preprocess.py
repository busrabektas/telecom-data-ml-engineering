import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow") 
    return df

def basic_eda(df: pd.DataFrame):

    print("\n[INFO] Shape:", df.shape)
    print("\n[INFO] Veri tipleri:\n", df.dtypes)
    print("\n[INFO] Missing values:\n", df.isnull().sum())
    print("\n[INFO] First 5 rows:\n", df.head())

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    missing_vals = ["nan", "NaN", "None", "???", "---", "NULL", "MISSING", "99999", "Not specified", "not specified", " "]
    df.replace(missing_vals, np.nan, inplace=True)

    numeric_cols = [
        "Age", "Tenure","VoiceCallMinutes",
        "SMSsent", "DataUsageGB", "IntlCallMinutes", "RoamingCharges", 
        "NetworkIssues", "DroppedCalls", "SupportCalls", "DataSpeed", 
        "Latency", "SatisfactionScore",  "PaymentHistory", 
        "ServiceDowntime", "ContractRenewal"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  

    return df

def show_missing_ratio(df: pd.DataFrame):

    missing_ratio = (df.isnull().sum() / len(df)) * 100
    print("\nMissing value ratio (%):\n", missing_ratio.sort_values(ascending=False))

def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    def fill_gender(group):
        mode = group["Gender"].mode()
        if not mode.empty:
            return group["Gender"].fillna(mode.iloc[0])
        else:
            return group["Gender"]

    if "Gender" in df.columns and "PlanType" in df.columns:
        df["Gender"] = df.groupby("PlanType", group_keys=False).apply(fill_gender)
        df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

    for col in ["PlanType", "PaymentMethod", "Location", "ServiceType"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    num_cols = [
        "Age", "Tenure", "VoiceCallMinutes",
        "SMSsent", "DataUsageGB", "IntlCallMinutes", "RoamingCharges", 
        "NetworkIssues", "DroppedCalls", "SupportCalls", "DataSpeed", 
        "Latency", "SatisfactionScore",  "PaymentHistory", 
        "ServiceDowntime", "ContractRenewal", "Churn"
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)


    if "CustomerID" in df.columns:
        df = df.dropna(subset=["CustomerID"])

    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        print(f"NaN count: {total_missing}")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("all values filled.")

    return df

def keep_common_columns(df: pd.DataFrame, path: str = "common_features.txt") -> pd.DataFrame:

    with open(path) as f:
        common_cols = f.read().splitlines()
    df_filtered = df[[col for col in common_cols if col in df.columns]].copy()
    print(f"[INFO] Ortak {len(df_filtered.columns)} sütun retained.")
    return df_filtered
