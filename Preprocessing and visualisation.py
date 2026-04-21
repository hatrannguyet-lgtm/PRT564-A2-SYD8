#import libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# files path
PATH_2018 = r"C:\Users\ASUS\Downloads\Qualifications and work 2018-19 Data Tables (1).xlsx"
PATH_2022 = r"C:\Users\ASUS\Downloads\QAW2223DC (2).xlsx"


#part 1: preprocessing
#function to clean extra spaces or blank cells 
def clean_cell(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if text == "":
        return None
    return text

#function to turn symbols like ".." (missing values) into Na
def to_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    missing_markers = {"", ".", "..", "np", "nan", "-", "--", "—"}
    if text in missing_markers:
        return np.nan
    return pd.to_numeric(text, errors="coerce")

#function to check the rows instead of just first cell 
def row_has_text(row, target_text):
    cleaned_values = [clean_cell(item) for item in row.tolist()]
    return target_text in cleaned_values

#function to uniform sex 
def tidy_sex(label):
    label_map = {
        "Males": "Male",
        "Females": "Female",
        "Persons": "Persons"
    }
    return label_map.get(label, label)
#function to uniform job stats 
def tidy_jobstat(label):
    label_map = {
        "Employed full-time": "Full-time",
        "Employed part-time": "Part-time",
        "Total employed": "Total employed"
    }
    return label_map.get(label, label)
#group and uniform the qualifications 
def tidy_qualification_group(label):
    if label is None:
        return None
    label = clean_cell(label)
    aliases = {
        "Has one or more non-school qualifications": "Any_non_school_qualification",
        "One": "One",
        "One non-school qualification": "One",
        "Two": "Two",
        "Two non-school qualifications": "Two",
        "Three or more": "Three_or_more",
        "Three or more non-school qualifications": "Three_or_more",
        "No non-school qualification": "No_qualification",
    }
    return aliases.get(label, label)

#function to parse the income table 
def parse_incometab(book_path, sheet_name, year_tag):
    raw_df = pd.read_excel(book_path, sheet_name=sheet_name, header=None)
    allowed_groups = {"One", "Two", "Three_or_more", "No_qualification"}
    job_labels = {"Employed full-time", "Employed part-time", "Total employed"}
    sex_labels = {"Males", "Females", "Persons"}
    parsed_rows = []
    current_job = None
    current_sex = None

    for _, row in raw_df.iterrows():
        if row_has_text(row, "Proportion (%)"):
            break

        first_cell = clean_cell(row.iloc[0])
        if first_cell is None: 
            continue
        if first_cell in job_labels: #to keep track of the current job status as I move down the rows 
            current_job = tidy_jobstat(first_cell)
            current_sex = None
            continue
        if first_cell in sex_labels:
            current_sex = tidy_sex(first_cell)
            continue
        qual_group = tidy_qualification_group(first_cell)
        if qual_group not in allowed_groups:
            continue
        if current_job not in {"Full-time", "Part-time"}:
            continue
        if current_sex not in {"Male", "Female"}:
            continue
        quintile_values = [to_number(v) for v in row[1:7].tolist()]
        avg_income = to_number(row[8] if len(row) > 8 else np.nan)
        median_income = to_number(row[9] if len(row) > 9 else np.nan)
        if pd.isna(avg_income):
            continue
        qual_lookup = {
            "No_qualification": 0,
            "One": 1,
            "Two": 2,
            "Three_or_more": 3
        }
        row_data = {
            "year": year_tag,
            "sex": current_sex,
            "employment_status": current_job,
            "qualification_group": qual_group,
            "qualification_count_code": qual_lookup[qual_group],
            "has_non_school_qualification": int(qual_group != "No_qualification"),
            "lowest_quintile_000": quintile_values[0],
            "second_quintile_000": quintile_values[1],
            "third_quintile_000": quintile_values[2],
            "fourth_quintile_000": quintile_values[3],
            "highest_quintile_000": quintile_values[4],
            "group_size_000": quintile_values[5],
            "avg_weekly_income": avg_income,
            "median_weekly_income": median_income
        }
        parsed_rows.append(row_data)
    cleaned_df = pd.DataFrame(parsed_rows)
    #test
    if cleaned_df.empty:
        raise ValueError("No rows were parsed. Have another look at the filters or the sheet structure.")
    cleaned_df["log_avg_weekly_income"] = np.log(cleaned_df["avg_weekly_income"])
    #putting fields in order
    ordered_cols = [
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
    cleaned_df = cleaned_df[ordered_cols].copy()
    cleaned_df = cleaned_df.sort_values(
        by=["employment_status", "sex", "qualification_count_code"]
    ).reset_index(drop=True)
    return cleaned_df
#function to build yearly data set seperately one for training and one for testing
def build_yearly_sets():
    data_2018 = parse_incometab(PATH_2018, "Table 2", "2018")
    data_2022 = parse_incometab(PATH_2022, "Table 3", "2022")
    return data_2018, data_2022



def build_train_test_by_year(train_df, test_df):
    feature_cols = [
        "sex",
        "employment_status",
        "qualification_group",
        "qualification_count_code",
        "group_size_000"
    ]
    X_train_raw = train_df[feature_cols].copy()
    y_train = train_df["log_avg_weekly_income"].copy()
    X_test_raw = test_df[feature_cols].copy()
    y_test = test_df["log_avg_weekly_income"].copy()
    cat_cols = ["sex", "employment_status", "qualification_group"]
    num_cols = ["qualification_count_code", "group_size_000"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_cols),
            ("num", StandardScaler(), num_cols)
        ]
    )

    X_train_ready = preprocessor.fit_transform(X_train_raw)
    X_test_ready = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.get_feature_names_out()


    X_train_ready_df = pd.DataFrame(X_train_ready, columns=feature_names)
    X_test_ready_df = pd.DataFrame(X_test_ready, columns=feature_names)

    return (
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        X_train_ready_df,
        X_test_ready_df,
        preprocessor
    )


def run_preprocessing():
    data_2018, data_2022 = build_yearly_sets()
    (
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        X_train_ready_df,
        X_test_ready_df,
        preprocessor
    ) = build_train_test_by_year(train_df=data_2022, test_df=data_2018)

    out_2018 = r"C:\Users\ASUS\Downloads\regression_dataset_2018.csv"
    out_2022 = r"C:\Users\ASUS\Downloads\regression_dataset_2022.csv"
    x_train_out = r"C:\Users\ASUS\Downloads\X_train_raw_2022.csv"
    y_train_out = r"C:\Users\ASUS\Downloads\y_train_2022.csv"
    x_test_out = r"C:\Users\ASUS\Downloads\X_test_raw_2018.csv"
    y_test_out = r"C:\Users\ASUS\Downloads\y_test_2018.csv"
    x_train_ready_out = r"C:\Users\ASUS\Downloads\X_train_processed_2022.csv"
    x_test_ready_out = r"C:\Users\ASUS\Downloads\X_test_processed_2018.csv"

    data_2018.to_csv(out_2018, index=False)
    data_2022.to_csv(out_2022, index=False)

    X_train_raw.to_csv(x_train_out, index=False)
    y_train.to_csv(y_train_out, index=False)

    X_test_raw.to_csv(x_test_out, index=False)
    y_test.to_csv(y_test_out, index=False)

    X_train_ready_df.to_csv(x_train_ready_out, index=False)
    X_test_ready_df.to_csv(x_test_ready_out, index=False)


    
    print("2018 dataset shape:", data_2018.shape)
    print("2022 dataset shape:", data_2022.shape)
    print("2018 preview:")
    print(data_2018.head(8))
    print(data_2022.head(8))
    print("Training year: 2022")
    print("Testing year: 2018")
    print("X_train_raw shape:", X_train_raw.shape)
    print("X_test_raw shape:", X_test_raw.shape)
    print("X_train_processed shape:", X_train_ready_df.shape)
    print("X_test_processed shape:", X_test_ready_df.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    
    return data_2018, data_2022






#part 2: visualisation
#use both year for comparison 
def make_eda_graphs(data_2018, data_2022):
    combined_eda = pd.concat([data_2018, data_2022], ignore_index=True)
    combined_eda["year"] = combined_eda["year"].astype(str)
    qualification_order = ["No_qualification", "One", "Two", "Three_or_more"]
    combined_eda["qualification_group"] = pd.Categorical(
        combined_eda["qualification_group"],
        categories=qualification_order,
        ordered=True
    )
    #chart labelling 
    qualification_labels = {
        "No_qualification": "No qualification",
        "One": "One qualification",
        "Two": "Two qualifications",
        "Three_or_more": "Three or more qualifications"
    }

    # 1 visual: income by qualification and year 
    income_by_qual_year = (
        combined_eda
        .groupby(["year", "qualification_group"], observed=False)["avg_weekly_income"]
        .mean()
        .reset_index()
    )
    qual_year_pivot = income_by_qual_year.pivot(
        index="qualification_group",
        columns="year",
        values="avg_weekly_income"
    )
    qual_year_pivot.index = [qualification_labels[idx] for idx in qual_year_pivot.index]
    chart1 = r"C:\Users\ASUS\Downloads\income_by_qualification_year.png"
    plt.figure(figsize=(10, 6))
    for year_name in qual_year_pivot.columns:
        plt.plot(
            qual_year_pivot.index,
            qual_year_pivot[year_name],
            marker="o",
            label=year_name
        )
    plt.xlabel("Qualification Group")
    plt.ylabel("Average Weekly Income")
    plt.title("Income by Qualification Group and Year")
    plt.legend(title="Year")
    plt.savefig(chart1)
    plt.show()

    # 2 visual: income by employment status and sex 
    income_by_emp_sex = (
        combined_eda
        .groupby(["employment_status", "sex"], observed=False)["avg_weekly_income"]
        .mean()
        .reset_index()
    )
    emp_sex_pivot = income_by_emp_sex.pivot(
        index="employment_status",
        columns="sex",
        values="avg_weekly_income"
    )
    chart2 = r"C:\Users\ASUS\Downloads\income_by_employment_sex.png"
    plt.figure(figsize=(10, 6))
    emp_sex_pivot.plot(kind="bar", ax=plt.gca())
    plt.xlabel("Employment Status")
    plt.ylabel("Average Weekly Income")
    plt.title("Income by Employment Status and Sex")
    plt.xticks(rotation=0)
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.savefig(chart2)
    plt.show()

    # 3 visual: qualification count vs log income 
    chart3 = r"C:\Users\ASUS\Downloads\qualification_count_vs_log_income.png"
    plt.figure(figsize=(10, 6))
    for yr, marker_style in zip(["2018", "2022"], ["o", "s"]):
        year_slice = combined_eda[combined_eda["year"] == yr]
        plt.scatter(
            year_slice["qualification_count_code"],
            year_slice["log_avg_weekly_income"],
            marker=marker_style,
            label=yr,
            s=100
        )
    plt.xlabel("Qualification Count Code")
    plt.ylabel("Log of Average Weekly Income")
    plt.title("Qualification Count vs Log Income by Year")
    plt.legend(title="Year")
    plt.tight_layout()
    plt.savefig(chart3)
    plt.show()

    # 4 visual: mean, median gap by year
    mean_median_gap = (
        combined_eda
        .assign(mean_gap=combined_eda["avg_weekly_income"] - combined_eda["median_weekly_income"])
        .groupby("year")["mean_gap"]
        .mean()
        .reset_index()
    )

    chart4 = r"C:\Users\ASUS\Downloads\mean_median_gap_by_year.png"
    plt.figure(figsize=(8, 5))
    plt.bar(mean_median_gap["year"], mean_median_gap["mean_gap"], color=["skyblue", "salmon"])
    plt.xlabel("Year")
    plt.ylabel("Mean - Median Income Gap ($)")
    plt.title("Mean - Median Weekly Income Gap by Year")
    plt.tight_layout()
    plt.savefig(chart4)
    plt.show()

def main():
    data_2018, data_2022 = run_preprocessing()
    make_eda_graphs(data_2018, data_2022)


if __name__ == "__main__":
    main()