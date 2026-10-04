"""
modules/analytics.py
Healthcare Analytics and Reporting Module for Hospital Patient Management System.

Aggregates healthcare metrics, statistical summaries, and generates
executive and clinical reports for college presentations and hospital administration.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from modules.data_processing import (
    calculate_age_statistics,
    calculate_cost_statistics,
    get_disease_frequency,
    get_gender_distribution,
    get_department_patient_count,
    get_treatment_outcome_statistics,
    get_cost_and_age_by_disease
)


def compute_dashboard_metrics(patients_df: pd.DataFrame, appointments_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute essential KPI metrics for the executive dashboard:
    - Total patients
    - Total appointments
    - Male, Female, Other counts
    - Total unique diseases
    - Most common disease
    - Completed treatments
    - Treatment success rate (% Recovered + Improved over all resolved treatments)
    - Total revenue / treatment spend
    """
    if patients_df.empty:
        return {
            "total_patients": 0,
            "total_appointments": 0,
            "male_count": 0,
            "female_count": 0,
            "other_count": 0,
            "disease_count": 0,
            "most_common_disease": "N/A",
            "completed_treatments": 0,
            "treatment_success_rate": 0.0,
            "avg_age": 0.0,
            "total_cost": 0.0
        }

    total_patients = len(patients_df)
    total_appointments = len(appointments_df) if not appointments_df.empty else 0

    # Gender breakdown
    gender_counts = patients_df["gender"].value_counts()
    male_count = int(gender_counts.get("Male", 0))
    female_count = int(gender_counts.get("Female", 0))
    other_count = int(gender_counts.get("Other", 0))

    # Disease metrics
    unique_diseases = patients_df["disease"].nunique()
    if not patients_df["disease"].dropna().empty:
        most_common_disease = patients_df["disease"].mode()[0]
    else:
        most_common_disease = "N/A"

    # Treatment outcomes
    outcome_counts = patients_df["treatment_outcome"].value_counts()
    recovered = int(outcome_counts.get("Recovered", 0))
    improved = int(outcome_counts.get("Improved", 0))
    unsuccessful = int(outcome_counts.get("Unsuccessful", 0))
    under_treatment = int(outcome_counts.get("Under Treatment", 0))

    # Completed treatments = Recovered + Improved + Unsuccessful
    resolved_treatments = recovered + improved + unsuccessful
    completed_treatments = resolved_treatments

    # Success rate = (Recovered + Improved) / Resolved treatments * 100
    if resolved_treatments > 0:
        success_rate = round(((recovered + improved) / resolved_treatments) * 100, 1)
    else:
        success_rate = 0.0

    # Age and Cost
    age_stats = calculate_age_statistics(patients_df)
    cost_stats = calculate_cost_statistics(patients_df)

    return {
        "total_patients": total_patients,
        "total_appointments": total_appointments,
        "male_count": male_count,
        "female_count": female_count,
        "other_count": other_count,
        "disease_count": unique_diseases,
        "most_common_disease": most_common_disease,
        "completed_treatments": completed_treatments,
        "under_treatment_count": under_treatment,
        "treatment_success_rate": success_rate,
        "avg_age": age_stats["mean_age"],
        "total_cost": cost_stats["total_cost"],
        "avg_cost": cost_stats["mean_cost"]
    }


def get_recent_registrations(patients_df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """
    Retrieve latest registered patients sorted by registration date.
    """
    if patients_df.empty:
        return pd.DataFrame()

    temp_df = patients_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df["registration_date"]):
        temp_df["registration_date"] = pd.to_datetime(temp_df["registration_date"], errors="coerce")

    sorted_df = temp_df.sort_values(by="registration_date", ascending=False).head(limit)
    
    # Return presentation-friendly subset of columns
    display_cols = [
        "patient_id", "name", "age", "gender", "disease",
        "department", "doctor", "registration_date", "treatment_outcome"
    ]
    available_cols = [c for c in display_cols if c in sorted_df.columns]
    
    clean_display = sorted_df[available_cols].copy()
    clean_display["registration_date"] = clean_display["registration_date"].dt.strftime("%Y-%m-%d")
    return clean_display


def generate_comprehensive_report_text(patients_df: pd.DataFrame, appointments_df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """
    Generates a structured medical and statistical report for capstone presentation,
    meeting all requirements for Patient, Treatment, and Department statistics.
    Returns both formatted Markdown/Text string and raw dictionary.
    """
    if patients_df.empty:
        return "No patient data available to generate report.", {}

    age_stats = calculate_age_statistics(patients_df)
    cost_stats = calculate_cost_statistics(patients_df)
    disease_df = get_disease_frequency(patients_df)
    gender_df = get_gender_distribution(patients_df)
    dept_df = get_department_patient_count(patients_df)
    outcome_df = get_treatment_outcome_statistics(patients_df)

    # 1. Patient Statistics
    total_patients = len(patients_df)
    avg_age = age_stats["mean_age"]
    median_age = age_stats["median_age"]
    min_age = age_stats["min_age"]
    max_age = age_stats["max_age"]

    male_count = int(patients_df[patients_df["gender"] == "Male"].shape[0])
    female_count = int(patients_df[patients_df["gender"] == "Female"].shape[0])
    other_count = int(patients_df[patients_df["gender"] == "Other"].shape[0])

    most_common_disease = disease_df.iloc[0]["disease"] if not disease_df.empty else "N/A"
    most_common_disease_count = int(disease_df.iloc[0]["count"]) if not disease_df.empty else 0

    # 2. Treatment Statistics
    outcome_dict = dict(zip(outcome_df["treatment_outcome"], outcome_df["count"]))
    recovered = outcome_dict.get("Recovered", 0)
    improved = outcome_dict.get("Improved", 0)
    unsuccessful = outcome_dict.get("Unsuccessful", 0)
    under_treatment = outcome_dict.get("Under Treatment", 0)
    
    successful_treatments = recovered + improved
    total_treatments = len(patients_df)
    resolved_treatments = successful_treatments + unsuccessful

    success_pct = round((successful_treatments / resolved_treatments * 100), 2) if resolved_treatments > 0 else 0.0
    avg_cost = cost_stats["mean_cost"]
    total_cost = cost_stats["total_cost"]

    # 3. Department Statistics
    most_visited_dept = dept_df.iloc[0]["department"] if not dept_df.empty else "N/A"
    most_visited_count = int(dept_df.iloc[0]["patient_count"]) if not dept_df.empty else 0

    # Format Markdown Report
    report_md = f"""# Hospital Patient Management and Health Analytics Report
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Project: Python & Data Science Healthcare Capstone*

---

## 1. Executive Summary & Patient Statistics (Demographics)
* **Total Registered Patients:** {total_patients:,}
* **Average Patient Age:** {avg_age} years (Median: {median_age}, Range: {min_age} - {max_age} years)
* **Age Standard Deviation:** {age_stats['std_age']} years
* **Male Patients:** {male_count} ({(male_count/total_patients*100):.1f}%)
* **Female Patients:** {female_count} ({(female_count/total_patients*100):.1f}%)
* **Other / Non-Binary Patients:** {other_count} ({(other_count/total_patients*100):.1f}%)
* **Most Prevalent Disease:** {most_common_disease} ({most_common_disease_count} patients, {(most_common_disease_count/total_patients*100):.1f}%)

---

## 2. Treatment Outcomes & Clinical Performance
* **Total Hospital Cases / Treatments:** {total_treatments:,}
* **Successful Outcomes (Recovered + Improved):** {successful_treatments}
  * Recovered: {recovered} ({(recovered/total_treatments*100):.1f}%)
  * Improved: {improved} ({(improved/total_treatments*100):.1f}%)
* **Unsuccessful Treatments:** {unsuccessful} ({(unsuccessful/total_treatments*100):.1f}%)
* **Active / Under Treatment Cases:** {under_treatment} ({(under_treatment/total_treatments*100):.1f}%)
* **Clinical Treatment Success Percentage:** {success_pct}% *(calculated on resolved cases)*
* **Average Treatment Cost:** ₹{avg_cost:,.2f}
* **Total Hospital Treatment Revenue:** ₹{total_cost:,.2f}

---

## 3. Department Performance & Patient Load
* **Most Visited Department:** **{most_visited_dept}** ({most_visited_count} admissions)

### Department Patient Distribution:
"""

    for _, row in dept_df.iterrows():
        report_md += f"- **{row['department']}**: {int(row['patient_count'])} patients | Total Revenue: ₹{row['total_cost']:,.2f} | Avg Cost: ₹{row['avg_cost']:,.2f}\n"

    report_md += f"""
---

## 4. Appointment Operations
* **Total Appointments Scheduled/Logged:** {len(appointments_df) if not appointments_df.empty else 0}
"""
    if not appointments_df.empty and "status" in appointments_df.columns:
        apt_counts = appointments_df["status"].value_counts()
        for status_name, cnt in apt_counts.items():
            report_md += f"- **{status_name} Appointments:** {cnt}\n"

    report_md += """
---
*Report compiled automatically by Hospital Patient Management & Health Analytics System.*
"""

    summary_dict = {
        "total_patients": total_patients,
        "avg_age": avg_age,
        "male_patients": male_count,
        "female_patients": female_count,
        "most_common_disease": most_common_disease,
        "total_treatments": total_treatments,
        "successful_treatments": successful_treatments,
        "unsuccessful_treatments": unsuccessful,
        "treatment_success_percentage": success_pct,
        "average_treatment_cost": avg_cost,
        "total_cost": total_cost,
        "most_visited_department": most_visited_dept,
        "dept_distribution": dept_df.to_dict(orient="records")
    }

    return report_md, summary_dict
