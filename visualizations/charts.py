"""
visualizations/charts.py
Matplotlib and Seaborn Chart Generation for Hospital Patient Management & Health Analytics.

Generates 7 core clinical and operational visualizations:
1. Disease Distribution – Bar Chart
2. Gender Distribution – Pie / Donut Chart
3. Age Distribution – Histogram with KDE curve
4. Department-Wise Patients – Bar Chart
5. Monthly Patient Registrations – Line Chart
6. Treatment Outcomes – Pie / Donut Chart
7. Average Treatment Cost by Disease – Bar Chart
Plus optional Demographic and Financial deep-dive charts.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Streamlit and server rendering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set global Seaborn & Matplotlib aesthetics
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True
})


def _create_empty_figure(title: str, message: str = "No data available for selected filters.") -> plt.Figure:
    """Helper to return an informative placeholder figure when data is empty."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#7F8C8D", fontsize=12, style="italic")
    ax.set_title(title, pad=12, color="#2C3E50")
    ax.set_axis_off()
    return fig


def plot_disease_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Visualization 1: Disease Distribution – Bar Chart
    Shows frequency of each disease in the patient population.
    """
    if df.empty or "disease" not in df.columns:
        return _create_empty_figure("Disease Distribution (Bar Chart)")

    counts = df["disease"].value_counts().reset_index()
    counts.columns = ["disease", "count"]

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("mako", n_colors=len(counts))

    bars = sns.barplot(
        data=counts,
        x="count",
        y="disease",
        hue="disease",
        legend=False,
        palette=palette,
        ax=ax,
        edgecolor="#2C3E50",
        linewidth=0.8
    )

    # Annotate bar counts
    for bar in bars.patches:
        width = bar.get_width()
        ax.text(
            width + max(counts["count"]) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#2C3E50"
        )

    ax.set_title("Disease Prevalence & Frequency Distribution", pad=14, color="#1A365D")
    ax.set_xlabel("Number of Diagnosed Patients", labelpad=8)
    ax.set_ylabel("Disease / Diagnosis", labelpad=8)
    ax.set_xlim(0, max(counts["count"]) * 1.15)
    plt.tight_layout()
    return fig


def plot_gender_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Visualization 2: Gender Distribution – Pie / Donut Chart
    Shows patient proportions by gender.
    """
    if df.empty or "gender" not in df.columns:
        return _create_empty_figure("Gender Distribution (Pie Chart)")

    gender_counts = df["gender"].value_counts()
    labels = gender_counts.index.tolist()
    counts = gender_counts.values.tolist()

    # Medical gender color palette
    color_map = {
        "Male": "#2B6CB0",     # Deep medical blue
        "Female": "#DD6B20",   # Warm coral orange
        "Other": "#805AD5"     # Soft purple
    }
    colors = [color_map.get(lbl, "#718096") for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
        pctdistance=0.75,
        textprops=dict(fontsize=10, fontweight="bold")
    )

    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(10.5)

    ax.legend(
        wedges,
        [f"{lbl} ({cnt})" for lbl, cnt in zip(labels, counts)],
        title="Gender (Count)",
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        frameon=True
    )

    ax.set_title("Patient Demographics: Gender Breakdown", pad=14, color="#1A365D")
    plt.tight_layout()
    return fig


def plot_age_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Visualization 3: Age Distribution – Histogram with KDE curve
    Visualizes patient age distribution with mean and median indicators.
    """
    if df.empty or "age" not in df.columns:
        return _create_empty_figure("Age Distribution (Histogram)")

    ages = df["age"].dropna().astype(float)
    if len(ages) == 0:
        return _create_empty_figure("Age Distribution (Histogram)")

    mean_age = np.mean(ages)
    median_age = np.median(ages)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    
    sns.histplot(
        ages,
        kde=True,
        bins=15,
        color="#3182CE",
        edgecolor="#1A365D",
        alpha=0.65,
        ax=ax,
        line_kws={"linewidth": 2.5, "color": "#1A365D"}
    )

    # Reference indicator lines
    ax.axvline(mean_age, color="#E53E3E", linestyle="--", linewidth=2, label=f"Mean Age: {mean_age:.1f}")
    ax.axvline(median_age, color="#38A169", linestyle="-.", linewidth=2, label=f"Median Age: {median_age:.1f}")

    ax.set_title("Patient Age Distribution & Density Profile", pad=14, color="#1A365D")
    ax.set_xlabel("Patient Age (Years)", labelpad=8)
    ax.set_ylabel("Patient Count (Frequency)", labelpad=8)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    return fig


def plot_department_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Visualization 4: Department-wise Patients – Bar Chart
    Shows total admissions/patients across hospital departments.
    """
    if df.empty or "department" not in df.columns:
        return _create_empty_figure("Department Distribution (Bar Chart)")

    dept_counts = df["department"].value_counts().reset_index()
    dept_counts.columns = ["department", "count"]

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("crest", n_colors=len(dept_counts))

    bars = sns.barplot(
        data=dept_counts,
        x="department",
        y="count",
        hue="department",
        legend=False,
        palette=palette,
        ax=ax,
        edgecolor="#2C3E50",
        linewidth=0.8
    )

    for bar in bars.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(dept_counts["count"]) * 0.015,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#2C3E50"
        )

    ax.set_title("Department-Wise Patient Distribution", pad=14, color="#1A365D")
    ax.set_xlabel("Hospital Medical Department", labelpad=8)
    ax.set_ylabel("Total Admitted Patients", labelpad=8)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, max(dept_counts["count"]) * 1.15)
    plt.tight_layout()
    return fig


def plot_monthly_registrations(df: pd.DataFrame) -> plt.Figure:
    """
    Visualization 5: Monthly Patient Registrations – Line Chart
    Demonstrates time-series trend of hospital registrations.
    """
    if df.empty or "registration_date" not in df.columns:
        return _create_empty_figure("Monthly Registrations (Line Chart)")

    temp_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df["registration_date"]):
        temp_df["registration_date"] = pd.to_datetime(temp_df["registration_date"], errors="coerce")

    temp_df = temp_df.dropna(subset=["registration_date"])
    if temp_df.empty:
        return _create_empty_figure("Monthly Registrations (Line Chart)")

    temp_df["month"] = temp_df["registration_date"].dt.strftime("%Y-%m")
    monthly = temp_df.groupby("month").size().reset_index(name="patient_count").sort_values("month")

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.plot(
        monthly["month"],
        monthly["patient_count"],
        marker="o",
        markersize=7,
        linewidth=2.5,
        color="#2B6CB0",
        label="Monthly Registrations"
    )

    # Fill area under curve for modern look
    ax.fill_between(monthly["month"], monthly["patient_count"], color="#3182CE", alpha=0.15)

    # Annotate points
    for _, row in monthly.iterrows():
        ax.annotate(
            f"{int(row['patient_count'])}",
            (row["month"], row["patient_count"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1A365D"
        )

    ax.set_title("Monthly Patient Registration Trend (Timeline)", pad=14, color="#1A365D")
    ax.set_xlabel("Registration Month (Year-Month)", labelpad=8)
    ax.set_ylabel("New Patients Registered", labelpad=8)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, max(monthly["patient_count"]) * 1.25)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


def plot_treatment_outcomes(df: pd.DataFrame) -> plt.Figure:
    """
    Visualization 6: Treatment Outcomes – Donut / Pie Chart
    Shows clinical resolution: Recovered, Improved, Under Treatment, Unsuccessful.
    """
    if df.empty or "treatment_outcome" not in df.columns:
        return _create_empty_figure("Treatment Outcomes (Pie Chart)")

    outcome_counts = df["treatment_outcome"].value_counts()
    labels = outcome_counts.index.tolist()
    counts = outcome_counts.values.tolist()

    outcome_colors = {
        "Recovered": "#38A169",       # Clinical Green
        "Improved": "#3182CE",        # Calming Blue
        "Under Treatment": "#DD6B20", # Active Amber/Orange
        "Unsuccessful": "#E53E3E"     # Red
    }
    colors = [outcome_colors.get(lbl, "#718096") for lbl in labels]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=120,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
        pctdistance=0.75,
        textprops=dict(fontsize=10, fontweight="bold")
    )

    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(10)

    ax.legend(
        wedges,
        [f"{lbl}: {cnt}" for lbl, cnt in zip(labels, counts)],
        title="Outcome (Count)",
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        frameon=True
    )

    ax.set_title("Clinical Treatment Outcomes & Success Rates", pad=14, color="#1A365D")
    plt.tight_layout()
    return fig


def plot_cost_by_disease(df: pd.DataFrame) -> plt.Figure:
    """
    Visualization 7: Average Treatment Cost by Disease – Bar Chart
    Shows financial analytics with formatted ₹ currency values.
    """
    if df.empty or "disease" not in df.columns or "treatment_cost" not in df.columns:
        return _create_empty_figure("Average Treatment Cost by Disease")

    summary = df.groupby("disease")["treatment_cost"].mean().reset_index()
    summary.columns = ["disease", "avg_cost"]
    summary = summary.sort_values(by="avg_cost", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    palette = sns.color_palette("flare", n_colors=len(summary))

    bars = sns.barplot(
        data=summary,
        x="avg_cost",
        y="disease",
        hue="disease",
        legend=False,
        palette=palette,
        ax=ax,
        edgecolor="#2C3E50",
        linewidth=0.8
    )

    max_c = max(summary["avg_cost"])
    for bar in bars.patches:
        w = bar.get_width()
        ax.text(
            w + max_c * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"₹{w:,.0f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#2C3E50"
        )

    ax.set_title("Average Treatment Cost by Disease (₹ INR)", pad=14, color="#1A365D")
    ax.set_xlabel("Average Cost in Indian Rupees (₹)", labelpad=8)
    ax.set_ylabel("Disease / Diagnosis", labelpad=8)
    ax.set_xlim(0, max_c * 1.22)
    plt.tight_layout()
    return fig


def plot_age_group_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Complementary Visualization: Age Group Distribution
    Categorizes patients into Pediatric, Young Adult, Adult, Middle-Aged, and Senior.
    """
    if df.empty or "age" not in df.columns:
        return _create_empty_figure("Age Groups Distribution")

    temp_df = df.copy()
    if "age_group" not in temp_df.columns:
        age_bins = [0, 18, 35, 50, 65, 125]
        age_labels = ["0-18 (Pediatric)", "19-35 (Young Adult)", "36-50 (Adult)", "51-65 (Middle Aged)", "65+ (Senior)"]
        temp_df["age_group"] = pd.cut(temp_df["age"], bins=age_bins, labels=age_labels, right=True)

    group_counts = temp_df["age_group"].value_counts(sort=False).reset_index()
    group_counts.columns = ["age_group", "count"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    palette = sns.color_palette("Blues_d", n_colors=len(group_counts))

    bars = sns.barplot(
        data=group_counts,
        x="age_group",
        y="count",
        hue="age_group",
        legend=False,
        palette=palette,
        ax=ax,
        edgecolor="#1A365D",
        linewidth=0.8
    )

    for bar in bars.patches:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(group_counts["count"]) * 0.02,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1A365D"
        )

    ax.set_title("Demographics: Patient Age-Bracket Distribution", pad=14, color="#1A365D")
    ax.set_xlabel("Age Classification Bracket", labelpad=8)
    ax.set_ylabel("Number of Patients", labelpad=8)
    ax.set_ylim(0, max(group_counts["count"]) * 1.18)
    plt.tight_layout()
    return fig
