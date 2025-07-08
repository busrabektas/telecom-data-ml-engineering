from preprocess import (
    load_data, basic_eda, clean_data,
    show_missing_ratio, fill_missing_data,
    keep_common_columns
)

def main():

    data_path_large = "telecom_data_large.parquet"
    data_path_small = "telecom_data_small.parquet"

    # Data loading
    print("Data Loaded")
    df_large = load_data(data_path_large)
    df_small = load_data(data_path_small)

    #eda
    print("\n[Large data basic EDA...")
    basic_eda(df_large)
    # print("\nSmall veri EDA...")
    # basic_eda(df_small)

    # common cols
    common_columns = [col for col in df_small.columns if col in df_large.columns]

    with open("common_features.txt", "w") as f:
        for col in common_columns:
            f.write(f"{col}\n")
    print(f"[INFO]  {len(common_columns)} common columns foud saved as 'common_features.txt' ")

    # print("\nSmall dataset missing value ratio:")
    # show_missing_ratio(df_small)

    # Cleaning and Filling ===
    print("\nLarge dataset cleaning")
    df_large = clean_data(df_large)
    show_missing_ratio(df_large)
    df_large = fill_missing_data(df_large)


    # # # small dataset cleaning and filling
    # print("\nSmall dataset temizleniyor...")
    # df_small = clean_data(df_small)
    # show_missing_ratio(df_small)
    # df_small = fill_missing_data(df_small)

    # === Filter with common columns
    df_large = keep_common_columns(df_large)


    df_large.to_csv("cleaned_large.csv", index=False)  # 
    print("cleaned_large.csv saved.")


if __name__ == "__main__":
    main()
