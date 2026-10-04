"""
modules/data_processing.py
Data Processing and Statistical Engine for Hospital Patient Management System.

Explicitly demonstrates:
- NumPy for numerical and statistical computations:
  * np.mean, np.median, np.min, np.max, np.std, np.sum, np.percentile
  * NumPy arrays, boolean masking, vector operations
- Pandas for dataset loading, cleaning, filtering, transformation, grouping, aggregation:
  * Series and DataFrame manipulation
  * Handling missing values and data type conversions
  * Frequency counts (value_counts)
  * GroupBy aggregations (mean, sum, count, median)
  * Binning and categorization (pd.cut)
  * DateTime feature engineering (pd.to_datetime, month extraction)
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def load_clean_patient_dataset(csv_filepath: str) -> pd.DataFrame:
    """
    Load and clean patient dataset using Pandas.
    Demonstrates Pandas I/O, type coercion, and missing value handling.
    """
    if not os.path.exists(csv_filepath):
        return pd.DataFrame()

    df = pd.read_csv(csv_filepath)
    if df.empty:
        return df

    # Data Type Conversions and Cleaning
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    df["gender"] = df["gender"].astype(str).str.strip().str.capitalize()
    df["treatment_cost"] = pd.to_numeric(df["treatment_cost"], errors="coerce").fillna(0.0)
    
    # Date Handling
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
    
    # Feature Engineering: Registration Year-Month
    df["reg_year_month"] = df["registration_date"].dt.strftime("%Y-%m")
    
    # Categorize Age Groups using pd.cut
    age_bins = [0, 18, 35, 50, 65, 125]
    age_labels = ["0-18 (Pediatric)", "19-35 (Young Adult)", "36-50 (Adult)", "51-65 (Middle Aged)", "65+ (Senior)"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)

    return df


def calculate_age_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate patient age statistics explicitly demonstrating NumPy.
    NumPy functions used:
    - np.array: converts pandas series to ndarray
    - np.mean: average age
    - np.median: median age
    - np.min: minimum age
    - np.max: maximum age
    - np.std: standard deviation
    - np.percentile: 25th, 50th, 75th percentiles (quartiles)
    """
    if df.empty or "age" not in df.columns:
        return {
            "mean_age": 0.0,
            "median_age": 0.0,
            "min_age": 0,
            "max_age": 0,
            "std_age": 0.0,
            "q25_age": 0.0,
            "q75_age": 0.0,
            "iqr_age": 0.0,
            "total_count": 0
        }

    # Extract NumPy array of ages
    age_array = np.array(df["age"].dropna().values, dtype=float)

    if len(age_array) == 0:
        return {
            "mean_age": 0.0,
            "median_age": 0.0,
            "min_age": 0,
            "max_age": 0,
            "std_age": 0.0,
            "q25_age": 0.0,
            "q75_age": 0.0,
            "iqr_age": 0.0,
            "total_count": 0
        }

    # Explicit NumPy Computations
    mean_val = float(np.mean(age_array))
    median_val = float(np.median(age_array))
    min_val = int(np.min(age_array))
    max_val = int(np.max(age_array))
    std_val = float(np.std(age_array))
    
    # Quartiles (25th, 50th, 75th percentiles)
    q25, q50, q75 = np.percentile(age_array, [25, 50, 75])
    iqr = float(q75 - q25)

    return {
        "mean_age": round(mean_val, 2),
        "median_age": round(median_val, 2),
        "min_age": min_val,
        "max_age": max_val,
        "std_age": round(std_val, 2),
        "q25_age": round(float(q25), 2),
        "q75_age": round(float(q75), 2),
        "iqr_age": round(iqr, 2),
        "total_count": len(age_array)
    }


def calculate_cost_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate treatment cost statistics explicitly demonstrating NumPy.
    NumPy functions used:
    - np.mean: average treatment cost
    - np.median: median treatment cost
    - np.sum: total revenue / treatment cost
    - np.min, np.max: cost bounds
    - np.std: standard deviation of costs
    """
    if df.empty or "treatment_cost" not in df.columns:
        return {
            "total_cost": 0.0,
            "mean_cost": 0.0,
            "median_cost": 0.0,
            "min_cost": 0.0,
            "max_cost": 0.0,
            "std_cost": 0.0
        }

    cost_array = np.array(df["treatment_cost"].dropna().values, dtype=float)
    if len(cost_array) == 0:
        return {
            "total_cost": 0.0,
            "mean_cost": 0.0,
            "median_cost": 0.0,
            "min_cost": 0.0,
            "max_cost": 0.0,
            "std_cost": 0.0
        }

    total_cost = float(np.sum(cost_array))
    mean_cost = float(np.mean(cost_array))
    median_cost = float(np.median(cost_array))
    min_cost = float(np.min(cost_array))
    max_cost = float(np.max(cost_array))
    std_cost = float(np.std(cost_array))

    return {
        "total_cost": round(total_cost, 2),
        "mean_cost": round(mean_cost, 2),
        "median_cost": round(median_cost, 2),
        "min_cost": round(min_cost, 2),
        "max_cost": round(max_cost, 2),
        "std_cost": round(std_cost, 2)
    }


def get_disease_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute frequency and percentage of each disease using Pandas value_counts.
    """
    if df.empty or "disease" not in df.columns:
        return pd.DataFrame(columns=["disease", "count", "percentage"])

    counts = df["disease"].value_counts()
    percentages = (df["disease"].value_counts(normalize=True) * 100).round(2)
    
    result_df = pd.DataFrame({
        "disease": counts.index,
        "count": counts.values,
        "percentage": percentages.values
    }).reset_index(drop=True)
    
    return result_df


def get_gender_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute patient count and proportion by gender using Pandas.
    """
    if df.empty or "gender" not in df.columns:
        return pd.DataFrame(columns=["gender", "count", "percentage"])

    counts = df["gender"].value_counts()
    percentages = (df["gender"].value_counts(normalize=True) * 100).round(2)
    
    return pd.DataFrame({
        "gender": counts.index,
        "count": counts.values,
        "percentage": percentages.values
    }).reset_index(drop=True)


def get_department_patient_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute department-wise patient count and total cost using Pandas GroupBy.
    """
    if df.empty or "department" not in df.columns:
        return pd.DataFrame(columns=["department", "patient_count", "total_cost", "avg_cost"])

    grouped = df.groupby("department").agg(
        patient_count=("patient_id", "count"),
        total_cost=("treatment_cost", "sum"),
        avg_cost=("treatment_cost", "mean")
    ).reset_index()

    grouped["total_cost"] = grouped["total_cost"].round(2)
    grouped["avg_cost"] = grouped["avg_cost"].round(2)
    return grouped.sort_values(by="patient_count", ascending=False).reset_index(drop=True)


def get_treatment_outcome_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute treatment outcome distribution using Pandas.
    Outcomes: Recovered, Improved, Under Treatment, Unsuccessful.
    """
    if df.empty or "treatment_outcome" not in df.columns:
        return pd.DataFrame(columns=["treatment_outcome", "count", "percentage"])

    counts = df["treatment_outcome"].value_counts()
    percentages = (df["treatment_outcome"].value_counts(normalize=True) * 100).round(2)

    return pd.DataFrame({
        "treatment_outcome": counts.index,
        "count": counts.values,
        "percentage": percentages.values
    }).reset_index(drop=True)


def get_monthly_registration_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly patient registration timeline using Pandas DateTime.
    """
    if df.empty or "registration_date" not in df.columns:
        return pd.DataFrame(columns=["month", "patient_count"])

    temp_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df["registration_date"]):
        temp_df["registration_date"] = pd.to_datetime(temp_df["registration_date"], errors="coerce")

    temp_df = temp_df.dropna(subset=["registration_date"])
    temp_df["month"] = temp_df["registration_date"].dt.strftime("%Y-%m")

    monthly = temp_df.groupby("month").size().reset_index(name="patient_count")
    return monthly.sort_values(by="month").reset_index(drop=True)


def get_cost_and_age_by_disease(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform multi-aggregate grouping by disease for average cost and average age.
    Demonstrates Pandas multi-column aggregation.
    """
    if df.empty or "disease" not in df.columns:
        return pd.DataFrame(columns=["disease", "patient_count", "avg_cost", "avg_age", "median_cost"])

    summary = df.groupby("disease").agg(
        patient_count=("patient_id", "count"),
        avg_cost=("treatment_cost", "mean"),
        median_cost=("treatment_cost", "median"),
        avg_age=("age", "mean")
    ).reset_index()

    summary["avg_cost"] = summary["avg_cost"].round(2)
    summary["median_cost"] = summary["median_cost"].round(2)
    summary["avg_age"] = summary["avg_age"].round(1)

    return summary.sort_values(by="avg_cost", ascending=False).reset_index(drop=True)


def get_age_group_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute age-bracket distribution using Pandas Categorical cut.
    """
    if df.empty or "age" not in df.columns:
        return pd.DataFrame(columns=["age_group", "count", "percentage"])

    if "age_group" not in df.columns:
        age_bins = [0, 18, 35, 50, 65, 125]
        age_labels = ["0-18 (Pediatric)", "19-35 (Young Adult)", "36-50 (Adult)", "51-65 (Middle Aged)", "65+ (Senior)"]
        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)

    counts = df["age_group"].value_counts(sort=False)
    percentages = (df["age_group"].value_counts(normalize=True, sort=False) * 100).round(2)

    return pd.DataFrame({
        "age_group": counts.index.astype(str),
        "count": counts.values,
        "percentage": percentages.values
    }).reset_index(drop=True)


def filter_dataset(
    df: pd.DataFrame,
    genders: Optional[list] = None,
    diseases: Optional[list] = None,
    departments: Optional[list] = None,
    doctors: Optional[list] = None,
    outcomes: Optional[list] = None,
    age_range: Optional[tuple] = None,
    date_range: Optional[tuple] = None
) -> pd.DataFrame:
    """
    General purpose multi-filter function using Pandas boolean masks.
    """
    if df.empty:
        return df

    filtered = df.copy()

    # Filter Gender
    if genders and "All" not in genders:
        filtered = filtered[filtered["gender"].isin(genders)]

    # Filter Disease
    if diseases and "All" not in diseases:
        filtered = filtered[filtered["disease"].isin(diseases)]

    # Filter Department
    if departments and "All" not in departments:
        filtered = filtered[filtered["department"].isin(departments)]

    # Filter Doctor
    if doctors and "All" not in doctors:
        filtered = filtered[filtered["doctor"].isin(doctors)]

    # Filter Outcome
    if outcomes and "All" not in outcomes:
        filtered = filtered[filtered["treatment_outcome"].isin(outcomes)]

    # Filter Age Range
    if age_range and len(age_range) == 2:
        min_age, max_age = age_range
        filtered = filtered[(filtered["age"] >= min_age) & (filtered["age"] <= max_age)]

    # Filter Date Range
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            if not pd.api.types.is_datetime64_any_dtype(filtered["registration_date"]):
                filtered["registration_date"] = pd.to_datetime(filtered["registration_date"], errors="coerce")
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            filtered = filtered[(filtered["registration_date"] >= start_dt) & (filtered["registration_date"] <= end_dt)]

    return filtered
