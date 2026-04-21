#PART 1: PREPROCESSING
#________________________
import pandas as pd # For data manipulation and analysis
import numpy as np # For numerical operations and handling missing values
import re # For regular expressions to clean text data
from sklearn.compose import ColumnTransformer # For applying different transformations to different columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler # For encoding categorical variables and scaling numerical features


# Define file the Qualfications and Work 2018-19 and 2022-23 data tables
PATH_2018= r"C:\Users\ASUS\Downloads\Qualifications and work 2018-19 Data Tables (1).xlsx"
PATH_2022= r"C:\Users\ASUS\Downloads\QAW2223DC (2).xlsx"

# Define a function to clean text and variables

#Function to clean text values from the cells 
def clean_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    #To fix double spaces and other whitespace
    s = re.sub(r"\s+", " ", s)
    return s if s else None

#Function to convert values to numeric, handling various formats and missing value indicators
def to_numerical(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in {"", ".", "..", "np", "nan", "-", "--", "—"}:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

#Function to check if row contains case insensitive text
def row_contains_text(row, text):
    vals = [clean_text(v) for v in row.tolist()]
    return text in vals

#Functions to normalise categorical labels
def normalise_sex(label):
    mapping = {"Males": "Male", "Females": "Female", "Persons": "Persons"}
    return mapping.get(label, label)    

def normalise_jobs_status(label):
    mapping = {
        "Employed full-time": "Full-time",
        "Employed part-time": "Part-time",
        "Total employed": "Total employed"
    }
    return mapping.get(label, label)

#Function to normalise qualification group labels into consistent categories for analysis
def normalise_qualification_group(label):
    if label is None:
        return None
    label = clean_text(label)
    mapping = {
        "Has one or more non-school qualifications": "Any_non_school_qualification",
        "One": "One",
        "One non-school qualification": "One",
        "Two": "Two",
        "Two non-school qualifications": "Two",
        "Three or more": "Three_or_more",
        "Three or more non-school qualifications": "Three_or_more",
        "No non-school qualification": "No_qualification",
    }
    return mapping.get(label, label)


#Function to parse the income data from the specified workbook and sheet, applying filters and transformations to extract relevant records
def parse_income_data(workbook_path, sheet_name, year_label):
    df = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)

    keep_groups = {"One", "Two", "Three_or_more", "No_qualification"}
    jobs_status = {"Employed full-time", "Employed part-time", "Total employed"}
    sex_headers = {"Males", "Females", "Persons"}

    records = []
    current_employment_status = None 
    current_sex = None
# Iterate through rows until we find the header row containing "Proportion (%)", then parse subsequent rows for relevant data based on employment status
    for _, row in df.iterrows():
        if row_contains_text(row, "Proportion (%)"):
            break

        label = clean_text(row[0])

        if label is None: 
            continue 

        if label in jobs_status:
            current_employment_status = normalise_jobs_status(label)
            current_sex = None
            continue

        if label in sex_headers:
            current_sex = normalise_sex(label)
            continue

        qualification_group = normalise_qualification_group(label)

        if qualification_group not in keep_groups:
            continue

        if current_employment_status not in {"Full-time", "Part-time"}:
            continue

        if current_sex not in {"Male", "Female"}:
            continue

        quintiles_total = [to_numerical(v) for v in row[1:7].tolist()]
        avg_income = to_numerical(row[8] if len(row) > 8 else np.nan)
        median_income = to_numerical(row[9] if len(row) > 9 else np.nan)
# Skip rows where average income is missing, as this 
# is our target variable for regression
        if pd.isna(avg_income):
            continue
# If we have valid data, append a record with all 
# relevant variables and the target income values
        records.append({
            "year": year_label,
            "sex": current_sex,
            "employment_status": current_employment_status,
            "qualification_group": qualification_group,
            "qualification_count_code": {
                "No_qualification": 0,
                "One": 1,
                "Two": 2,
                "Three_or_more": 3
            }[qualification_group],
            "has_non_school_qualification": int(qualification_group != "No_qualification"),
            "lowest_quintile_000": quintiles_total[0],
            "second_quintile_000": quintiles_total[1],
            "third_quintile_000": quintiles_total[2],
            "fourth_quintile_000": quintiles_total[3],
            "highest_quintile_000": quintiles_total[4],
            "group_size_000": quintiles_total[5],
            "avg_weekly_income": avg_income,
            "median_weekly_income": median_income
        })

    out = pd.DataFrame(records)
# If no valid rows were parsed, raise an error to alert 
# the user to check the filters and sheet contents 
    if  out.empty:
       raise ValueError("No rows were parsed. Check filters and sheet contents.")
    
    out["log_avg_weekly_income"] = np.log(out["avg_weekly_income"])

# Reorder columns for consistency and readability
    out = out[
        [
            "year",
            "sex",
            "employment_status",
            "qualification_group",
            "qualification_count_code",
            "has_non_school_qualification",
            "group_size_000",
            "lowest_quintile_000",
            "second_quintile_000",
            "third_quintile_000",
            "fourth_quintile_000",
            "highest_quintile_000",
            "avg_weekly_income",
            "median_weekly_income",
            "log_avg_weekly_income"
        ]
    ].copy()

    out = out.sort_values(
        by=["employment_status", "sex", "qualification_count_code"]
    ).reset_index(drop=True)

    return out

# Function to build separate datasets for each year by parsing the respective 
# sheets and applying the defined filters and transformations
def build_separate_year_datasets():
    df_2018 = parse_income_data(PATH_2018, "Table 2", "2018")
    df_2022 = parse_income_data(PATH_2022, "Table 3", "2022")
    return df_2018, df_2022

def buildtrain_testbyyear(train_df, test_df):
    feature_columns = [
        "sex",
        "employment_status",
        "qualification_group",
        "qualification_count_code",
        "group_size_000"
    ]

    X_train_raw = train_df[feature_columns].copy()
    y_train = train_df["log_avg_weekly_income"].copy()

    X_test_raw = test_df[feature_columns].copy()
    y_test = test_df["log_avg_weekly_income"].copy()

    categorical_cols = ["sex", "employment_status", "qualification_group"]
    numerical_cols = ["qualification_count_code", "group_size_000"]

    try: 
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )

# Fit the preprocessor on the training data and transform both train and test sets
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

    return (
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        X_train_processed_df,
        X_test_processed_df,
        preprocessor
    )
# Main function to execute the preprocessing steps, build datasets, and 
# save the results to CSV files for both raw and processed data
def main():
    # Keep the years separate
    df_2018, df_2022 = build_separate_year_datasets()

    # Train on 2022, test on 2018
    (
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        X_train_processed_df,
        X_test_processed_df,
        preprocessor
    ) = buildtrain_testbyyear(train_df=df_2022, test_df=df_2018)

    # Save separate yearly datasets
    df_2018.to_csv(
        r"C:\Users\ASUS\Downloads\regression_dataset_2018.csv",
        index=False
    )
    df_2022.to_csv(
        r"C:\Users\ASUS\Downloads\regression_dataset_2022.csv",
        index=False
    )

# Save raw train/test datasets before processing, to allow for future flexibility 
    X_train_raw.to_csv(
        r"C:\Users\ASUS\Downloads\X_train_raw_2022.csv",
        index=False
    )
    y_train.to_csv(
        r"C:\Users\ASUS\Downloads\y_train_2022.csv",
        index=False
    )

    X_test_raw.to_csv(
        r"C:\Users\ASUS\Downloads\X_test_raw_2018.csv",
        index=False
    )
    y_test.to_csv(
        r"C:\Users\ASUS\Downloads\y_test_2018.csv",
        index=False
    )

    # Save processed train/test datasets after encoding and scaling, ready for modeling
    X_train_processed_df.to_csv(
        r"C:\Users\ASUS\Downloads\X_train_processed_2022.csv",
        index=False
    )
    X_test_processed_df.to_csv(
        r"C:\Users\ASUS\Downloads\X_test_processed_2018.csv",
        index=False
    )
# Print summary of the preprocessing steps, dataset shapes, and saved files for verification
    print("Preprocessing complete.")
    print()

    print("2018 dataset shape:", df_2018.shape)
    print("2022 dataset shape:", df_2022.shape)
    print("Expected rows per year: 16")
    print()

    print("2018 preview:")
    print(df_2018.head(8))
    print()

    print("2022 preview:")
    print(df_2022.head(8))
    print()

    print("Training year: 2022")
    print("Testing year: 2018")
    print()

    print("X_train_raw shape:", X_train_raw.shape)
    print("X_test_raw shape:", X_test_raw.shape)
    print("X_train_processed shape:", X_train_processed_df.shape)
    print("X_test_processed shape:", X_test_processed_df.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print()
# List the saved files for reference and verification
    print("Saved files:")
    print(r"C:\Users\ASUS\Downloads\regression_dataset_2018.csv")
    print(r"C:\Users\ASUS\Downloads\regression_dataset_2022.csv")
    print(r"C:\Users\ASUS\Downloads\X_train_raw_2022.csv")
    print(r"C:\Users\ASUS\Downloads\y_train_2022.csv")
    print(r"C:\Users\ASUS\Downloads\X_test_raw_2018.csv")
    print(r"C:\Users\ASUS\Downloads\y_test_2018.csv")
    print(r"C:\Users\ASUS\Downloads\X_train_processed_2022.csv")
    print(r"C:\Users\ASUS\Downloads\X_test_processed_2018.csv")


if __name__ == "__main__":
    main()

#PART 2: VISUALISATION
#________________________

#Imprt library 
import matplotlib.pyplot as plt

#Read the preprocessed datasets for 2018 and 2022
df_2018 = pd.read_csv(r"C:\Users\ASUS\Downloads\regression_dataset_2018.csv")
df_2022 = pd.read_csv(r"C:\Users\ASUS\Downloads\regression_dataset_2022.csv")
#Function to keep a combined EDA
eda_data = pd.concat([df_2018, df_2022], ignore_index=True)

eda_data["year"] = eda_data["year"].astype(str)

#Fix the order of the qualification groups for better visualisation
qualification_order = ["No_qualification", "One", "Two", "Three_or_more"]
eda_data["qualification_group"] = pd.Categorical(
    eda_data["qualification_group"],
    categories=qualification_order,
    ordered=True
)
#Label mapping for clearer visualisation
qualification_labelmap = {
    "No_qualification": "No qualification",
    "One": "One qualification",
    "Two": "Two qualifications",
    "Three_or_more": "Three or more qualifications"
}
#First visualisation: Income by qualification group and year 
income_qualyear = eda_data.groupby(["year", "qualification_group"])["avg_weekly_income"].mean().reset_index()

pivotqualyear = income_qualyear.pivot(index="qualification_group", columns="year", values="avg_weekly_income")

pivotqualyear.index = [qualification_labelmap[idx] for idx in pivotqualyear.index]

plt.figure(figsize=(10, 6))
for col in pivotqualyear.columns:
    plt.plot(pivotqualyear.index, pivotqualyear[col], marker="o", label=col)

plt.xlabel("Qualification Group")
plt.ylabel("Average Weekly Income")
plt.title("Income by Qualification Group and Year")
plt.legend(title = "Year")
plt.savefig(r"C:\Users\ASUS\Downloads\income_by_qualification_year.png")
plt.show()

#Second visualisation: Income by employment status and sex
income_empsex = eda_data.groupby(["employment_status", "sex"])["avg_weekly_income"].mean().reset_index()

pivot_empsex = income_empsex.pivot(index="employment_status", columns="sex", values="avg_weekly_income")

plt.figure(figsize=(10, 6))
pivot_empsex.plot(kind="bar", ax=plt.gca())
plt.xlabel("Employment Status") 
plt.ylabel("Average Weekly Income")
plt.title("Income by Employment Status and Sex")
plt.xticks(rotation=0)                      
plt.legend(title = "Sex")
plt.savefig(r"C:\Users\ASUS\Downloads\income_by_employment_sex.png")
plt.show()

#Third visualisation: Qualification count vs log of average weekly income
plt.figure(figsize=(10, 6))

for year_value, marker_style in zip(["2018", "2022"], ["o", "s"]):
    subset = eda_data[eda_data["year"] == year_value]
    plt.scatter(subset["qualification_count_code"], subset["log_avg_weekly_income"], marker=marker_style, label=year_value, s = 100)

plt.xlabel("Qualification Count Code")
plt.ylabel("Log of Average Weekly Income")      
plt.title("Qualification Count vs Log Income by Year")
plt.legend(title = "Year")
plt.savefig(r"C:\Users\ASUS\Downloads\qualification_count_vs_log_income.png")
plt.show()

#Fourth visualisation: Mean - median gap by year 
mean_median_gap = eda_data.assign(mean_gap = eda_data["avg_weekly_income"] - eda_data["median_weekly_income"]).groupby("year")["mean_gap"].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(mean_median_gap["year"], mean_median_gap["mean_gap"], color=["skyblue", "salmon"])
plt.xlabel("Year")      
plt.ylabel("Mean - Median Income Gap ($)")
plt.title("Mean - Median Weekly Income Gap by Year")
plt.savefig(r"C:\Users\ASUS\Downloads\mean_median_gap_by_year.png")
plt.show()  
