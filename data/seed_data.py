"""
seed_data.py
Generates realistic sample healthcare datasets (patients and appointments)
with at least 100+ patient records for college capstone demonstration.
"""

import os
import random
import datetime
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PATIENTS_CSV_PATH = os.path.join(DATA_DIR, "patients.csv")
APPOINTMENTS_CSV_PATH = os.path.join(DATA_DIR, "appointments.csv")


def generate_sample_data(num_patients=125, num_appointments=45, force=False):
    """
    Generate realistic healthcare sample data if files do not exist or force=True.
    Demonstrates: Lists, Dictionaries, Loops, File Handling, Pandas, NumPy.
    """
    if os.path.exists(PATIENTS_CSV_PATH) and os.path.exists(APPOINTMENTS_CSV_PATH) and not force:
        return

    random.seed(42)
    np.random.seed(42)

    male_first_names = [
        "Aarav", "Amit", "Arjun", "Deepak", "Gaurav", "Ishaan", "Kiran", "Madhav",
        "Manish", "Nikhil", "Pranav", "Rahul", "Rajesh", "Rakesh", "Ramesh",
        "Rohan", "Sanjay", "Siddharth", "Suresh", "Tarun", "Varun", "Vikram",
        "Vivek", "Kunal", "Alok"
    ]
    
    female_first_names = [
        "Aditi", "Ananya", "Divya", "Kavita", "Meera", "Neha", "Pooja", "Priya",
        "Ritu", "Sneha", "Sunita", "Tanvi", "Zoya", "Shruti", "Swati", "Deepa",
        "Anjali", "Pallavi", "Shreya", "Nandini"
    ]

    last_names = [
        "Sharma", "Verma", "Patel", "Reddy", "Iyer", "Nair", "Kumar", "Singh", 
        "Gupta", "Malhotra", "Joshi", "Sen", "Mehta", "Bhat", "Chopra", "Das", 
        "Kapoor", "Nambiar", "Mukherjee", "Deshmukh", "Menon", "Krishnan", "Saxena"
    ]

    cities = [
        ("Indira Nagar", "Bengaluru"), ("Bandra West", "Mumbai"), ("Connaught Place", "New Delhi"),
        ("Anna Nagar", "Chennai"), ("Banjara Hills", "Hyderabad"), ("Salt Lake", "Kolkata"),
        ("Kothrud", "Pune"), ("Navrangpura", "Ahmedabad"), ("Aliganj", "Lucknow"),
        ("Mansarovar", "Jaipur"), ("Kakkanad", "Kochi"), ("Boring Road", "Patna")
    ]

    blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    blood_group_weights = [0.26, 0.05, 0.31, 0.05, 0.08, 0.03, 0.20, 0.02]

    # Departments, doctors, associated diseases and realistic treatment costs
    dept_info = {
        "Cardiology": {
            "doctors": ["Dr. Rajesh Sharma", "Dr. Vikram Malhotra"],
            "diseases": [
                ("Hypertension", 8500, 16000, "ACE Inhibitors & Lifestyle Plan"),
                ("Coronary Artery Disease", 95000, 165000, "Coronary Angioplasty & Stent")
            ]
        },
        "Endocrinology": {
            "doctors": ["Dr. Priya Patel", "Dr. Suresh Menon"],
            "diseases": [
                ("Type 2 Diabetes", 11000, 24000, "Metformin & Insulin Therapy"),
                ("Hypothyroidism", 6500, 12000, "Levothyroxine Hormone Therapy")
            ]
        },
        "Pulmonology": {
            "doctors": ["Dr. Arvind Kumar", "Dr. Deepa Krishnan"],
            "diseases": [
                ("Bronchial Asthma", 14000, 28000, "Inhaled Corticosteroids & Inhaler"),
                ("Acute Pneumonia", 26000, 52000, "IV Antibiotics & Oxygen Therapy")
            ]
        },
        "Orthopedics": {
            "doctors": ["Dr. Sneha Reddy", "Dr. Vivek Deshmukh"],
            "diseases": [
                ("Osteoarthritis", 35000, 68000, "Joint Injections & Physiotherapy"),
                ("Lumbar Spondylosis", 18000, 34000, "Spinal Decompression & Rehab")
            ]
        },
        "Neurology": {
            "doctors": ["Dr. Ananya Sen", "Dr. Siddharth Kapoor"],
            "diseases": [
                ("Migraine Headache", 7500, 16500, "Triptans & Preventive Regimen"),
                ("Epilepsy", 22000, 44000, "Anticonvulsant Drug Therapy")
            ]
        },
        "Nephrology": {
            "doctors": ["Dr. Madhav Joshi", "Dr. Kavita Gupta"],
            "diseases": [
                ("Chronic Kidney Disease", 45000, 95000, "Hemodialysis & Renal Protocol"),
                ("Kidney Stones", 28000, 54000, "Lithotripsy & Medical Expulsive Therapy")
            ]
        },
        "General Medicine": {
            "doctors": ["Dr. Alok Verma", "Dr. Sunita Nair"],
            "diseases": [
                ("Dengue Fever", 15000, 32000, "IV Fluid Resuscitation & Platelet Monitoring"),
                ("Gastroenteritis", 8000, 17000, "Oral Rehydration & Antibiotic Regimen")
            ]
        }
    }

    outcomes = ["Recovered", "Improved", "Under Treatment", "Unsuccessful"]
    outcome_weights = [0.46, 0.34, 0.14, 0.06]

    start_date = datetime.date(2025, 1, 15)
    end_date = datetime.date(2026, 9, 25)
    days_span = (end_date - start_date).days

    patients_list = []
    departments = list(dept_info.keys())

    for i in range(1, num_patients + 1):
        patient_id = f"PAT-{1000 + i}"
        
        # Gender and realistic First Name
        gender_seed = random.random()
        if gender_seed < 0.52:
            gender = "Male"
            fn = random.choice(male_first_names)
        elif gender_seed < 0.98:
            gender = "Female"
            fn = random.choice(female_first_names)
        else:
            gender = "Other"
            fn = random.choice(["Kiran", "Tanvi", "Deepak", "Aditi"])
            
        ln = random.choice(last_names)
        name = f"{fn} {ln}"
        
        # Age: Realistic distribution centered around 48 with range 8 to 85
        age = int(np.clip(int(np.random.normal(48, 17)), 8, 85))
        
        # Phone: 10-digit valid Indian format
        phone = f"{random.randint(6, 9)}{random.randint(100000000, 999999999)}"
        
        # Address
        locality, city = random.choice(cities)
        address = f"#{random.randint(12, 199)}, {locality}, {city}"
        
        # Blood Group
        blood_group = random.choices(blood_groups, weights=blood_group_weights)[0]
        
        # Department & Doctor & Disease
        dept = random.choice(departments)
        doc = random.choice(dept_info[dept]["doctors"])
        disease_choice = random.choice(dept_info[dept]["diseases"])
        disease_name = disease_choice[0]
        min_c, max_c, default_treatment = disease_choice[1], disease_choice[2], disease_choice[3]
        
        treatment_cost = float(round(random.uniform(min_c, max_c), 2))
        outcome = random.choices(outcomes, weights=outcome_weights)[0]
        
        # Registration & Admission Dates
        rand_days = random.randint(0, days_span)
        reg_date = start_date + datetime.timedelta(days=rand_days)
        adm_date = reg_date + datetime.timedelta(days=random.randint(0, 2))
        
        if outcome == "Under Treatment":
            disch_date = "Ongoing"
        else:
            stay_days = random.randint(2, 12)
            disch_date = (adm_date + datetime.timedelta(days=stay_days)).strftime("%Y-%m-%d")

        patient_record = {
            "patient_id": patient_id,
            "name": name,
            "age": age,
            "gender": gender,
            "phone": phone,
            "address": address,
            "blood_group": blood_group,
            "disease": disease_name,
            "doctor": doc,
            "department": dept,
            "treatment": default_treatment,
            "treatment_cost": treatment_cost,
            "treatment_outcome": outcome,
            "admission_date": adm_date.strftime("%Y-%m-%d"),
            "discharge_date": disch_date,
            "registration_date": reg_date.strftime("%Y-%m-%d")
        }
        patients_list.append(patient_record)

    # Save Patients DataFrame
    patients_df = pd.DataFrame(patients_list)
    patients_df.to_csv(PATIENTS_CSV_PATH, index=False)

    # Generate Appointments linked to these patients
    time_slots = ["09:00 AM", "09:45 AM", "10:30 AM", "11:15 AM", "12:00 PM", 
                  "02:00 PM", "02:45 PM", "03:30 PM", "04:15 PM", "05:00 PM"]
    statuses = ["Completed", "Scheduled", "Cancelled"]
    status_weights = [0.55, 0.35, 0.10]

    appointments_list = []
    sampled_patients = random.sample(patients_list, min(num_appointments, len(patients_list)))

    for idx, p in enumerate(sampled_patients, start=1):
        apt_id = f"APT-{2000 + idx}"
        apt_status = random.choices(statuses, weights=status_weights)[0]
        
        reg_dt = datetime.datetime.strptime(p["registration_date"], "%Y-%m-%d").date()
        apt_date = reg_dt + datetime.timedelta(days=random.randint(5, 45))
        
        apt_record = {
            "appointment_id": apt_id,
            "patient_id": p["patient_id"],
            "patient_name": p["name"],
            "doctor_name": p["doctor"],
            "department": p["department"],
            "appointment_date": apt_date.strftime("%Y-%m-%d"),
            "appointment_time": random.choice(time_slots),
            "status": apt_status
        }
        appointments_list.append(apt_record)

    appointments_df = pd.DataFrame(appointments_list)
    appointments_df.to_csv(APPOINTMENTS_CSV_PATH, index=False)
    print(f"Sample data successfully regenerated: {len(patients_df)} patients, {len(appointments_df)} appointments.")


if __name__ == "__main__":
    generate_sample_data(force=True)
