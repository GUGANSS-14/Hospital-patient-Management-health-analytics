"""
app.py
Hospital Patient Management and Health Analytics System
Main Streamlit Application

A complete Python & Data Science capstone project demonstrating:
- Variables, Data types, Conditionals, Loops, Functions, Lists, Dictionaries
- Object-Oriented Programming (Classes)
- File Handling & Persistence (CSV)
- Exception Handling & Validation
- NumPy (Statistical and numerical computations)
- Pandas (Data loading, cleaning, filtering, grouping, aggregation)
- Matplotlib & Seaborn (Interactive clinical & operational charts)
- Streamlit (Modern responsive web UI)
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Hospital Patient Management & Health Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure local imports resolve correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from modules.patient import PatientManager, Patient
from modules.appointment import AppointmentManager, Appointment
from modules.data_processing import (
    load_clean_patient_dataset,
    calculate_age_statistics,
    calculate_cost_statistics,
    get_disease_frequency,
    get_gender_distribution,
    get_department_patient_count,
    get_treatment_outcome_statistics,
    get_monthly_registration_trends,
    get_cost_and_age_by_disease,
    get_age_group_distribution,
    filter_dataset
)
from modules.analytics import (
    compute_dashboard_metrics,
    get_recent_registrations,
    generate_comprehensive_report_text
)
from visualizations.charts import (
    plot_disease_distribution,
    plot_gender_distribution,
    plot_age_distribution,
    plot_department_distribution,
    plot_monthly_registrations,
    plot_treatment_outcomes,
    plot_cost_by_disease,
    plot_age_group_distribution
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F7FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #2B6CB0;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #A0AEC0;
        margin-top: 0.3rem;
    }
    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2D3748;
        border-left: 4px solid #3182CE;
        padding-left: 0.6rem;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    /* Badges */
    .badge-recovered { background-color: #C6F6D5; color: #22543D; padding: 2px 8px; border-radius: 6px; font-weight: bold; }
    .badge-improved { background-color: #BEE3F8; color: #2A4365; padding: 2px 8px; border-radius: 6px; font-weight: bold; }
    .badge-treatment { background-color: #FEEBC8; color: #7B341E; padding: 2px 8px; border-radius: 6px; font-weight: bold; }
    .badge-unsuccessful { background-color: #FED7D7; color: #742A2A; padding: 2px 8px; border-radius: 6px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# Initialize Data Managers in Session State
@st.cache_resource
def get_managers():
    """Singleton pattern to instantiate managers and load CSV records."""
    pm = PatientManager()
    am = AppointmentManager()
    return pm, am

patient_manager, appointment_manager = get_managers()

# Department, Doctor, and Disease Knowledge Base for forms
DOCTOR_CATALOG = {
    "Cardiology": {
        "doctors": ["Dr. Rajesh Sharma", "Dr. Vikram Malhotra"],
        "diseases": ["Hypertension", "Coronary Artery Disease"]
    },
    "Endocrinology": {
        "doctors": ["Dr. Priya Patel", "Dr. Suresh Menon"],
        "diseases": ["Type 2 Diabetes", "Hypothyroidism"]
    },
    "Pulmonology": {
        "doctors": ["Dr. Arvind Kumar", "Dr. Deepa Krishnan"],
        "diseases": ["Bronchial Asthma", "Acute Pneumonia"]
    },
    "Orthopedics": {
        "doctors": ["Dr. Sneha Reddy", "Dr. Vivek Deshmukh"],
        "diseases": ["Osteoarthritis", "Lumbar Spondylosis"]
    },
    "Neurology": {
        "doctors": ["Dr. Ananya Sen", "Dr. Siddharth Kapoor"],
        "diseases": ["Migraine Headache", "Epilepsy"]
    },
    "Nephrology": {
        "doctors": ["Dr. Madhav Joshi", "Dr. Kavita Gupta"],
        "diseases": ["Chronic Kidney Disease", "Kidney Stones"]
    },
    "General Medicine": {
        "doctors": ["Dr. Alok Verma", "Dr. Sunita Nair"],
        "diseases": ["Dengue Fever", "Gastroenteritis"]
    }
}

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/hospital-3.png", width=72)
st.sidebar.title("Hospital System")
st.sidebar.markdown("**Healthcare Analytics & Management**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Dashboard",
        "📝 Patient Registration",
        "📋 Patient Records",
        "📅 Appointments",
        "📊 Healthcare Analytics",
        "📑 Reports & Insights"
    ]
)

st.sidebar.markdown("---")
# Quick System Status in Sidebar
live_patients_df = patient_manager.get_all_patients_df()
live_apts_df = appointment_manager.get_appointments_df()

st.sidebar.caption("🏥 **System Database Status**")
st.sidebar.caption(f"• Total Patients: **{len(live_patients_df)}**")
st.sidebar.caption(f"• Total Appointments: **{len(live_apts_df)}**")
st.sidebar.caption(f"• Data Source: `data/patients.csv`")
st.sidebar.caption("🎓 *Python & Data Science Capstone Project*")


# ==============================================================================
# 1. DASHBOARD MODULE
# ==============================================================================
if menu == "🏠 Dashboard":
    st.markdown('<div class="main-title">🏥 Hospital Patient Management & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Live Clinical Operations Dashboard, Patient Inflow & Healthcare KPIs</div>', unsafe_allow_html=True)

    # Compute KPI Metrics
    kpis = compute_dashboard_metrics(live_patients_df, live_apts_df)

    # Top Row Metrics (4 Cards)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Patients</div>
            <div class="metric-value">{kpis['total_patients']}</div>
            <div class="metric-sub">Registered in Hospital</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Appointments</div>
            <div class="metric-value" style="color:#319795;">{kpis['total_appointments']}</div>
            <div class="metric-sub">Consultations Scheduled</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gender Ratio</div>
            <div class="metric-value" style="color:#D69E2E;">{kpis['male_count']}M : {kpis['female_count']}F</div>
            <div class="metric-sub">Other: {kpis['other_count']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Clinical Diagnoses</div>
            <div class="metric-value" style="color:#805AD5;">{kpis['disease_count']}</div>
            <div class="metric-sub">Unique Disease Categories</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    # Second Row Metrics (4 Cards)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Most Common Disease</div>
            <div class="metric-value" style="font-size:1.3rem; color:#DD6B20; padding-top:0.4rem;">{kpis['most_common_disease']}</div>
            <div class="metric-sub">Highest Patient Inflow</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Completed Treatments</div>
            <div class="metric-value" style="color:#38A169;">{kpis['completed_treatments']}</div>
            <div class="metric-sub">Active: {kpis['under_treatment_count']} under care</div>
        </div>
        """, unsafe_allow_html=True)
    with c7:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Treatment Success Rate</div>
            <div class="metric-value" style="color:#38A169;">{kpis['treatment_success_rate']}%</div>
            <div class="metric-sub">Recovered + Improved Cases</div>
        </div>
        """, unsafe_allow_html=True)
    with c8:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Patient Age</div>
            <div class="metric-value" style="color:#4A5568;">{kpis['avg_age']} <span style="font-size:1rem;">yrs</span></div>
            <div class="metric-sub">Demographic Average</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Operational Highlights & Clinical Distribution</div>', unsafe_allow_html=True)

    # Dashboard Charts Preview
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        st.pyplot(plot_disease_distribution(live_patients_df), clear_figure=True)
    with ch_col2:
        st.pyplot(plot_department_distribution(live_patients_df), clear_figure=True)

    # Recent Registrations Table
    st.markdown('<div class="section-header">Recent Patient Registrations</div>', unsafe_allow_html=True)
    recent_df = get_recent_registrations(live_patients_df, limit=6)
    if not recent_df.empty:
        st.dataframe(
            recent_df,
            use_container_width=True,
            column_config={
                "patient_id": "Patient ID",
                "name": "Patient Name",
                "age": "Age",
                "gender": "Gender",
                "disease": "Diagnosis",
                "department": "Department",
                "doctor": "Attending Doctor",
                "registration_date": "Reg Date",
                "treatment_outcome": "Outcome"
            },
            hide_index=True
        )
    else:
        st.info("No patient records found.")


# ==============================================================================
# 2. PATIENT REGISTRATION MODULE
# ==============================================================================
elif menu == "📝 Patient Registration":
    st.markdown('<div class="main-title">📝 Patient Registration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Register new patients with automated ID generation, validation, and persistent storage.</div>', unsafe_allow_html=True)

    # Auto-generate next Patient ID
    suggested_id = patient_manager.generate_patient_id()

    with st.form("patient_registration_form", clear_on_submit=False):
        st.subheader("1. General & Demographics")
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            patient_id_input = st.text_input("Patient ID (Auto-Generated)", value=suggested_id, help="Unique identifier for patient")
        with r1_c2:
            name_input = st.text_input("Patient Full Name *", placeholder="e.g. Ramesh Patel")
        with r1_c3:
            age_input = st.number_input("Age *", min_value=1, max_value=120, value=35, step=1)

        r2_c1, r2_c2, r2_c3 = st.columns(3)
        with r2_c1:
            gender_input = st.selectbox("Gender *", ["Male", "Female", "Other"])
        with r2_c2:
            phone_input = st.text_input("Phone Number (10 digits) *", placeholder="e.g. 9876543210")
        with r2_c3:
            blood_group_input = st.selectbox("Blood Group *", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

        address_input = st.text_input("Residential Address *", placeholder="e.g. #42, Bandra West, Mumbai")

        st.subheader("2. Clinical & Departmental Assignment")
        r3_c1, r3_c2, r3_c3 = st.columns(3)
        with r3_c1:
            dept_choice = st.selectbox("Department *", list(DOCTOR_CATALOG.keys()))
        with r3_c2:
            avail_doctors = DOCTOR_CATALOG[dept_choice]["doctors"]
            doctor_input = st.selectbox("Doctor Assigned *", avail_doctors)
        with r3_c3:
            avail_diseases = DOCTOR_CATALOG[dept_choice]["diseases"]
            disease_input = st.selectbox("Diagnosis / Disease *", avail_diseases)

        r4_c1, r4_c2, r4_c3 = st.columns(3)
        with r4_c1:
            treatment_input = st.text_input("Treatment Plan", value="Standard Clinical Protocol")
        with r4_c2:
            cost_input = st.number_input("Treatment Cost (₹ INR)", min_value=0.0, value=15000.0, step=500.0)
        with r4_c3:
            outcome_input = st.selectbox("Initial Outcome Status", ["Under Treatment", "Improved", "Recovered", "Unsuccessful"])

        r5_c1, r5_c2 = st.columns(2)
        with r5_c1:
            reg_date_input = st.date_input("Registration Date", value=datetime.date.today())
        with r5_c2:
            adm_date_input = st.date_input("Admission Date", value=datetime.date.today())

        submitted = st.form_submit_button("Register Patient", use_container_width=True, type="primary")

        if submitted:
            # Build patient record dictionary
            new_record = {
                "patient_id": patient_id_input,
                "name": name_input,
                "age": age_input,
                "gender": gender_input,
                "phone": phone_input,
                "address": address_input,
                "blood_group": blood_group_input,
                "disease": disease_input,
                "doctor": doctor_input,
                "department": dept_choice,
                "treatment": treatment_input,
                "treatment_cost": cost_input,
                "treatment_outcome": outcome_input,
                "admission_date": adm_date_input.strftime("%Y-%m-%d"),
                "discharge_date": "Ongoing" if outcome_input == "Under Treatment" else datetime.date.today().strftime("%Y-%m-%d"),
                "registration_date": reg_date_input.strftime("%Y-%m-%d")
            }

            # Register through patient manager
            success, msg = patient_manager.register_patient(new_record)
            if success:
                st.success(f"✅ {msg}")
                st.balloons()
            else:
                st.error(f"❌ Registration Failed: {msg}")


# ==============================================================================
# 3. PATIENT RECORDS MODULE
# ==============================================================================
elif menu == "📋 Patient Records":
    st.markdown('<div class="main-title">📋 Patient Records Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Search, filter, view, update, and manage patient medical records.</div>', unsafe_allow_html=True)

    # Search & Filter Controls
    with st.expander("🔍 Search and Filter Options", expanded=True):
        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            search_query = st.text_input("Search Patient", placeholder="Enter ID or Name...")
        with f_c2:
            gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female", "Other"])
        with f_c3:
            all_diseases = ["All"] + sorted(list(live_patients_df["disease"].dropna().unique()))
            disease_filter = st.selectbox("Filter by Disease", all_diseases)

        f_c4, f_c5, f_c6 = st.columns(3)
        with f_c4:
            all_depts = ["All"] + sorted(list(live_patients_df["department"].dropna().unique()))
            dept_filter = st.selectbox("Filter by Department", all_depts)
        with f_c5:
            all_outcomes = ["All"] + sorted(list(live_patients_df["treatment_outcome"].dropna().unique()))
            outcome_filter = st.selectbox("Filter by Outcome", all_outcomes)
        with f_c6:
            age_range = st.slider("Filter by Age Range", 0, 100, (0, 100))

    # Apply Filters using PatientManager
    filtered_df = live_patients_df.copy()

    # Search
    if search_query.strip():
        q = search_query.strip()
        filtered_df = filtered_df[
            filtered_df["patient_id"].astype(str).str.contains(q, case=False, na=False) |
            filtered_df["name"].astype(str).str.contains(q, case=False, na=False)
        ]

    # Filters
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df["gender"] == gender_filter]
    if disease_filter != "All":
        filtered_df = filtered_df[filtered_df["disease"] == disease_filter]
    if dept_filter != "All":
        filtered_df = filtered_df[filtered_df["department"] == dept_filter]
    if outcome_filter != "All":
        filtered_df = filtered_df[filtered_df["treatment_outcome"] == outcome_filter]
    filtered_df = filtered_df[(filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])]

    st.write(f"Showing **{len(filtered_df)}** of **{len(live_patients_df)}** patient records.")

    # Data Table View
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "patient_id": "Patient ID",
            "name": "Name",
            "age": "Age",
            "gender": "Gender",
            "phone": "Phone",
            "address": "Address",
            "blood_group": "Blood",
            "disease": "Diagnosis",
            "doctor": "Doctor",
            "department": "Department",
            "treatment": "Treatment",
            "treatment_cost": st.column_config.NumberColumn("Cost (₹)", format="₹%d"),
            "treatment_outcome": "Outcome",
            "admission_date": "Admission",
            "discharge_date": "Discharge",
            "registration_date": "Reg Date"
        }
    )

    # Download Button
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered Records as CSV",
        data=csv_data,
        file_name=f"patient_records_{datetime.date.today()}.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Record Actions (View Details, Update, Delete)
    st.markdown('<div class="section-header">Record Actions: View Details, Update, or Delete</div>', unsafe_allow_html=True)
    all_pids = live_patients_df["patient_id"].tolist() if not live_patients_df.empty else []

    if all_pids:
        action_col1, action_col2 = st.columns([1, 2])
        with action_col1:
            selected_pid = st.selectbox("Select Patient by ID", all_pids)
            action_type = st.radio("Select Action", ["View Individual Details", "Update Patient Info", "Delete Patient Record"])

        patient_record = patient_manager.get_patient_by_id(selected_pid)

        with action_col2:
            if action_type == "View Individual Details" and patient_record:
                st.markdown(f"### Patient File: **{patient_record['name']}** (`{patient_record['patient_id']}`)")
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.write(f"• **Age / Gender:** {patient_record['age']} yrs / {patient_record['gender']}")
                    st.write(f"• **Phone:** {patient_record['phone']}")
                    st.write(f"• **Blood Group:** {patient_record['blood_group']}")
                    st.write(f"• **Address:** {patient_record['address']}")
                    st.write(f"• **Registration Date:** {patient_record['registration_date']}")
                with vc2:
                    st.write(f"• **Department:** {patient_record['department']}")
                    st.write(f"• **Attending Doctor:** {patient_record['doctor']}")
                    st.write(f"• **Diagnosis:** {patient_record['disease']}")
                    st.write(f"• **Treatment:** {patient_record['treatment']}")
                    st.write(f"• **Cost:** ₹{float(patient_record['treatment_cost']):,.2f}")
                    st.write(f"• **Status / Outcome:** **{patient_record['treatment_outcome']}**")

            elif action_type == "Update Patient Info" and patient_record:
                st.markdown(f"### Update Record: **{patient_record['name']}** (`{patient_record['patient_id']}`)")
                with st.form("update_patient_form"):
                    u_c1, u_c2 = st.columns(2)
                    with u_c1:
                        u_name = st.text_input("Name", value=patient_record["name"])
                        u_age = st.number_input("Age", min_value=1, max_value=120, value=int(patient_record["age"]))
                        u_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(patient_record["gender"]) if patient_record["gender"] in ["Male", "Female", "Other"] else 0)
                        u_phone = st.text_input("Phone", value=patient_record["phone"])
                        u_addr = st.text_input("Address", value=patient_record["address"])
                    with u_c2:
                        u_blood = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], index=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].index(patient_record["blood_group"]) if patient_record["blood_group"] in ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"] else 0)
                        u_dept = st.selectbox("Department", list(DOCTOR_CATALOG.keys()), index=list(DOCTOR_CATALOG.keys()).index(patient_record["department"]) if patient_record["department"] in DOCTOR_CATALOG else 0)
                        u_doc = st.text_input("Doctor", value=patient_record["doctor"])
                        u_disease = st.text_input("Disease", value=patient_record["disease"])
                        u_outcome = st.selectbox("Outcome", ["Under Treatment", "Recovered", "Improved", "Unsuccessful"], index=["Under Treatment", "Recovered", "Improved", "Unsuccessful"].index(patient_record["treatment_outcome"]) if patient_record["treatment_outcome"] in ["Under Treatment", "Recovered", "Improved", "Unsuccessful"] else 0)
                        u_cost = st.number_input("Treatment Cost (₹)", min_value=0.0, value=float(patient_record["treatment_cost"]))

                    update_btn = st.form_submit_button("Save Changes", type="primary")
                    if update_btn:
                        updated_dict = {
                            "patient_id": patient_record["patient_id"],
                            "name": u_name,
                            "age": u_age,
                            "gender": u_gender,
                            "phone": u_phone,
                            "address": u_addr,
                            "blood_group": u_blood,
                            "disease": u_disease,
                            "doctor": u_doc,
                            "department": u_dept,
                            "treatment": patient_record.get("treatment", "Standard Care"),
                            "treatment_cost": u_cost,
                            "treatment_outcome": u_outcome,
                            "admission_date": patient_record.get("admission_date", ""),
                            "discharge_date": datetime.date.today().strftime("%Y-%m-%d") if u_outcome != "Under Treatment" else "Ongoing",
                            "registration_date": patient_record.get("registration_date", "")
                        }
                        ok, msg = patient_manager.update_patient(selected_pid, updated_dict)
                        if ok:
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

            elif action_type == "Delete Patient Record" and patient_record:
                st.warning(f"⚠️ Are you sure you want to permanently delete patient **{patient_record['name']}** (ID: `{selected_pid}`)?")
                if st.button("Confirm Delete Patient", type="primary"):
                    ok, msg = patient_manager.delete_patient(selected_pid)
                    if ok:
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
    else:
        st.info("No patients available in database.")


# ==============================================================================
# 4. APPOINTMENT MANAGEMENT MODULE
# ==============================================================================
elif menu == "📅 Appointments":
    st.markdown('<div class="main-title">📅 Appointment Management Module</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Schedule consultations, view appointment queues, update statuses, and manage cancellations.</div>', unsafe_allow_html=True)

    apt_tab1, apt_tab2 = st.tabs(["📅 View & Manage Appointments", "➕ Book New Appointment"])

    with apt_tab1:
        # Metrics Row
        total_apts = len(live_apts_df)
        scheduled_cnt = len(live_apts_df[live_apts_df["status"] == "Scheduled"]) if not live_apts_df.empty else 0
        completed_cnt = len(live_apts_df[live_apts_df["status"] == "Completed"]) if not live_apts_df.empty else 0
        cancelled_cnt = len(live_apts_df[live_apts_df["status"] == "Cancelled"]) if not live_apts_df.empty else 0

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Total Appointments", total_apts)
        m_c2.metric("Scheduled", scheduled_cnt)
        m_c3.metric("Completed", completed_cnt)
        m_c4.metric("Cancelled", cancelled_cnt)

        st.markdown("---")
        # Search & Filter
        s_c1, s_c2 = st.columns([2, 1])
        with s_c1:
            apt_search = st.text_input("Search Appointments", placeholder="Search by Patient Name, ID, or Doctor...")
        with s_c2:
            status_filter = st.selectbox("Filter Status", ["All", "Scheduled", "Completed", "Cancelled"])

        display_apts = appointment_manager.search_appointments(apt_search, status_filter)

        st.dataframe(
            display_apts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "appointment_id": "Appointment ID",
                "patient_id": "Patient ID",
                "patient_name": "Patient Name",
                "doctor_name": "Doctor",
                "department": "Department",
                "appointment_date": "Date",
                "appointment_time": "Time Slot",
                "status": "Status"
            }
        )

        st.markdown('<div class="section-header">Update Appointment Status or Cancel</div>', unsafe_allow_html=True)
        if not live_apts_df.empty:
            u_col1, u_col2, u_col3 = st.columns(3)
            with u_col1:
                sel_apt_id = st.selectbox("Select Appointment ID", live_apts_df["appointment_id"].tolist())
            with u_col2:
                new_st = st.selectbox("New Status", ["Scheduled", "Completed", "Cancelled"])
            with u_col3:
                st.write("")
                st.write("")
                if st.button("Update Status", type="primary"):
                    ok, msg = appointment_manager.update_appointment_status(sel_apt_id, new_st)
                    if ok:
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    with apt_tab2:
        st.markdown("### ➕ Schedule a Consultation Appointment")
        suggested_apt_id = appointment_manager.generate_appointment_id()

        with st.form("book_appointment_form"):
            b_c1, b_c2 = st.columns(2)
            with b_c1:
                st.text_input("Appointment ID (Auto)", value=suggested_apt_id, disabled=True)
                
                # Link with registered patients
                patient_options = {}
                for p in patient_manager.get_all_patients():
                    patient_options[f"{p['patient_id']} - {p['name']}"] = p

                if patient_options:
                    selected_p_str = st.selectbox("Select Patient *", list(patient_options.keys()))
                    selected_p_data = patient_options[selected_p_str]
                else:
                    st.warning("No registered patients available.")
                    selected_p_data = None

                dept_apt = st.selectbox("Department *", list(DOCTOR_CATALOG.keys()))
                doc_apt = st.selectbox("Doctor Assigned *", DOCTOR_CATALOG[dept_apt]["doctors"])

            with b_c2:
                apt_date = st.date_input("Appointment Date *", value=datetime.date.today() + datetime.timedelta(days=1))
                apt_time = st.selectbox("Time Slot *", [
                    "09:00 AM", "09:45 AM", "10:30 AM", "11:15 AM", "12:00 PM",
                    "02:00 PM", "02:45 PM", "03:30 PM", "04:15 PM", "05:00 PM"
                ])
                apt_status = st.selectbox("Initial Status", ["Scheduled", "Completed"])

            book_btn = st.form_submit_button("Confirm Appointment Booking", type="primary")

            if book_btn:
                if not selected_p_data:
                    st.error("Please select a valid registered patient.")
                else:
                    new_apt = {
                        "appointment_id": suggested_apt_id,
                        "patient_id": selected_p_data["patient_id"],
                        "patient_name": selected_p_data["name"],
                        "doctor_name": doc_apt,
                        "department": dept_apt,
                        "appointment_date": apt_date.strftime("%Y-%m-%d"),
                        "appointment_time": apt_time,
                        "status": apt_status
                    }
                    ok, msg = appointment_manager.add_appointment(new_apt)
                    if ok:
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")


# ==============================================================================
# 5. HEALTHCARE ANALYTICS MODULE
# ==============================================================================
elif menu == "📊 Healthcare Analytics":
    st.markdown('<div class="main-title">📊 Healthcare Analytics & Data Science Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Demonstrating NumPy Statistical Computations, Pandas Aggregations, and Seaborn Visualizations.</div>', unsafe_allow_html=True)

    # Interactive Global Filter Bar
    with st.expander("⚙️ Interactive Filter Panel (Dynamically Updates All Charts & Statistics)", expanded=False):
        fl_c1, fl_c2, fl_c3, fl_c4 = st.columns(4)
        with fl_c1:
            filter_gender = st.multiselect("Gender", ["Male", "Female", "Other"], default=["Male", "Female", "Other"])
        with fl_c2:
            depts_avail = sorted(list(live_patients_df["department"].dropna().unique()))
            filter_dept = st.multiselect("Department", depts_avail, default=depts_avail)
        with fl_c3:
            outcomes_avail = sorted(list(live_patients_df["treatment_outcome"].dropna().unique()))
            filter_outcome = st.multiselect("Treatment Outcome", outcomes_avail, default=outcomes_avail)
        with fl_c4:
            filter_age = st.slider("Age Range", 0, 100, (0, 100))

    # Apply Filters to dataset for analytics
    analytics_df = filter_dataset(
        live_patients_df,
        genders=filter_gender,
        departments=filter_dept,
        outcomes=filter_outcome,
        age_range=filter_age
    )

    if analytics_df.empty:
        st.warning("⚠️ No records match the selected filters. Please adjust your criteria.")
    else:
        # Section A: Explicit NumPy Statistical Computations
        st.markdown('<div class="section-header">1. Numerical & Statistical Insights (Powered by NumPy)</div>', unsafe_allow_html=True)

        age_stats = calculate_age_statistics(analytics_df)
        cost_stats = calculate_cost_statistics(analytics_df)

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Average Age (NumPy mean)", f"{age_stats['mean_age']} yrs")
        sc2.metric("Median Age (NumPy median)", f"{age_stats['median_age']} yrs")
        sc3.metric("Age Range (min - max)", f"{age_stats['min_age']} - {age_stats['max_age']} yrs")
        sc4.metric("Avg Treatment Cost", f"₹{cost_stats['mean_cost']:,.0f}")
        sc5.metric("Total Treatment Spend", f"₹{cost_stats['total_cost']:,.0f}")

        st.caption(f"📐 *NumPy Statistical Verification: Age Standard Deviation = **{age_stats['std_age']}** yrs | IQR = **{age_stats['iqr_age']}** yrs | Cost Standard Deviation = **₹{cost_stats['std_cost']:,.2f}***")

        st.markdown("---")

        # Section B: 7 Core Visualizations
        st.markdown('<div class="section-header">2. Core Clinical & Demographic Visualizations (Matplotlib & Seaborn)</div>', unsafe_allow_html=True)

        # Row 1: Disease Distribution & Gender Breakdown
        chart_r1_1, chart_r1_2 = st.columns([1.2, 0.8])
        with chart_r1_1:
            st.pyplot(plot_disease_distribution(analytics_df), clear_figure=True)
        with chart_r1_2:
            st.pyplot(plot_gender_distribution(analytics_df), clear_figure=True)

        # Row 2: Age Distribution Histogram & Age-Group Bar Chart
        chart_r2_1, chart_r2_2 = st.columns(2)
        with chart_r2_1:
            st.pyplot(plot_age_distribution(analytics_df), clear_figure=True)
        with chart_r2_2:
            st.pyplot(plot_age_group_distribution(analytics_df), clear_figure=True)

        # Row 3: Department-wise Patients & Monthly Trends
        chart_r3_1, chart_r3_2 = st.columns(2)
        with chart_r3_1:
            st.pyplot(plot_department_distribution(analytics_df), clear_figure=True)
        with chart_r3_2:
            st.pyplot(plot_monthly_registrations(analytics_df), clear_figure=True)

        # Row 4: Treatment Outcomes & Cost by Disease
        chart_r4_1, chart_r4_2 = st.columns([0.8, 1.2])
        with chart_r4_1:
            st.pyplot(plot_treatment_outcomes(analytics_df), clear_figure=True)
        with chart_r4_2:
            st.pyplot(plot_cost_by_disease(analytics_df), clear_figure=True)

        st.markdown("---")

        # Section C: GroupBy Multi-Aggregation Analysis (Pandas)
        st.markdown('<div class="section-header">3. Disease Financial & Demographic Summary (Pandas GroupBy)</div>', unsafe_allow_html=True)
        disease_summary_df = get_cost_and_age_by_disease(analytics_df)
        st.dataframe(
            disease_summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "disease": "Disease / Diagnosis",
                "patient_count": "Patient Count",
                "avg_cost": st.column_config.NumberColumn("Average Cost", format="₹%d"),
                "median_cost": st.column_config.NumberColumn("Median Cost", format="₹%d"),
                "avg_age": "Average Age (Yrs)"
            }
        )


# ==============================================================================
# 6. REPORTS & INSIGHTS MODULE
# ==============================================================================
elif menu == "📑 Reports & Insights":
    st.markdown('<div class="main-title">📑 Clinical & Statistical Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Generate comprehensive statistical reports for hospital management and capstone viva presentation.</div>', unsafe_allow_html=True)

    report_md, report_dict = generate_comprehensive_report_text(live_patients_df, live_apts_df)

    # 3 Summary Cards as required by the problem statement:
    # 1. Patient Statistics, 2. Treatment Statistics, 3. Department Statistics
    rep_c1, rep_c2, rep_c3 = st.columns(3)

    with rep_c1:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <h4 style="color:#2B6CB0; margin-top:0;">👥 Patient Statistics</h4>
            <p>• <b>Total Patients:</b> {report_dict.get('total_patients', 0)}</p>
            <p>• <b>Average Age:</b> {report_dict.get('avg_age', 0)} years</p>
            <p>• <b>Male Patients:</b> {report_dict.get('male_patients', 0)}</p>
            <p>• <b>Female Patients:</b> {report_dict.get('female_patients', 0)}</p>
            <p>• <b>Top Disease:</b> {report_dict.get('most_common_disease', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

    with rep_c2:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <h4 style="color:#38A169; margin-top:0;">🩺 Treatment Statistics</h4>
            <p>• <b>Total Treatments:</b> {report_dict.get('total_treatments', 0)}</p>
            <p>• <b>Successful:</b> {report_dict.get('successful_treatments', 0)}</p>
            <p>• <b>Unsuccessful:</b> {report_dict.get('unsuccessful_treatments', 0)}</p>
            <p>• <b>Success Rate:</b> {report_dict.get('treatment_success_percentage', 0)}%</p>
            <p>• <b>Average Cost:</b> ₹{report_dict.get('average_treatment_cost', 0):,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with rep_c3:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <h4 style="color:#D69E2E; margin-top:0;">🏥 Department Statistics</h4>
            <p>• <b>Most Visited:</b> {report_dict.get('most_visited_department', 'N/A')}</p>
            <p>• <b>Total Departments:</b> {len(report_dict.get('dept_distribution', []))}</p>
            <p>• <b>Total Revenue:</b> ₹{report_dict.get('total_cost', 0):,.2f}</p>
            <p>• <b>Appointments:</b> {len(live_apts_df)}</p>
            <p>• <b>Operational Status:</b> Normal</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Full Formatted Medical Report Display
    st.markdown('<div class="section-header">Full Automated Healthcare Report</div>', unsafe_allow_html=True)
    st.markdown(report_md)

    # Download Report Button
    st.download_button(
        label="📥 Download Full Statistical Report (.MD)",
        data=report_md,
        file_name=f"Hospital_Health_Analytics_Report_{datetime.date.today()}.md",
        mime="text/markdown",
        type="primary"
    )

    st.markdown("---")

    # Capstone Viva & Concepts Review
    with st.expander("🎓 College Capstone Viva & Python Concepts Demonstrated", expanded=False):
        st.markdown("""
        ### Integrated Python & Data Science Concept Mapping:
        1. **Variables & Data Types**: Strings, Integers (Age), Floats (Treatment Cost), Booleans, Datetime objects.
        2. **Conditional Statements (`if / elif / else`)**: Input validation, search queries, multi-filtering, clinical outcomes.
        3. **Loops (`for / while`)**: Patient ID auto-generation, list filtering, dictionary iteration, statistics calculation.
        4. **Functions (`def`)**: Modular functions with type hinting, return values, docstrings, and parameter validation.
        5. **Python Lists (`[]`)**: Dynamic collections of patient and appointment records, filtering, and comprehension.
        6. **Python Dictionaries (`{}`)**: Key-value mapping representing structured medical records and patient entities.
        7. **Object-Oriented Programming (Classes)**: `Patient`, `PatientManager`, `Appointment`, `AppointmentManager`.
        8. **File Handling (`open`, CSV)**: Safe persistence to `data/patients.csv` and `data/appointments.csv`.
        9. **Exception Handling (`try-except`)**: Robust parsing of user input, file errors, and numerical conversions.
        10. **NumPy**: `np.mean()`, `np.median()`, `np.min()`, `np.max()`, `np.std()`, `np.percentile()`, ndarrays.
        11. **Pandas**: `DataFrame`, `Series`, `read_csv()`, `to_csv()`, `value_counts()`, `groupby()`, `agg()`, `pd.cut()`, `pd.to_datetime()`.
        12. **Matplotlib & Seaborn**: 7 required clinical visualizations (bar, pie, histogram, line, donut) with styling.
        13. **Streamlit UI**: Reactive web framework with metric cards, tables, forms, filters, and chart rendering.
        """)

st.markdown("---")
st.caption("Hospital Patient Management and Health Analytics System • Python & Data Science Capstone Project")
