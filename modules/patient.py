"""
modules/patient.py
Patient Management Module for Hospital Patient Management and Health Analytics System.

Demonstrates:
- Object-Oriented Programming (Patient, PatientManager classes)
- Python Lists and Dictionaries for data structure manipulation
- Control flow: Conditional statements (if-elif-else) and Loops (for, while)
- Functions with type annotations and validation logic
- File Handling: Reading and writing CSV datasets
- Exception Handling: try-except blocks for data integrity and safe I/O
"""

import os
import re
import datetime
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd

# Path configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients.csv")


class Patient:
    """
    Represents an individual hospital patient.
    Demonstrates Python Class structure, constructor (__init__), and encapsulation.
    """
    def __init__(
        self,
        patient_id: str,
        name: str,
        age: int,
        gender: str,
        phone: str,
        address: str,
        blood_group: str,
        disease: str,
        doctor: str,
        department: str,
        treatment: str = "Standard Medical Care",
        treatment_cost: float = 0.0,
        treatment_outcome: str = "Under Treatment",
        admission_date: Optional[str] = None,
        discharge_date: str = "Ongoing",
        registration_date: Optional[str] = None
    ):
        # Instance Variables and Types
        self.patient_id: str = str(patient_id).strip()
        self.name: str = str(name).strip().title()
        self.age: int = int(age)
        self.gender: str = str(gender).strip()
        self.phone: str = str(phone).strip()
        self.address: str = str(address).strip()
        self.blood_group: str = str(blood_group).strip()
        self.disease: str = str(disease).strip()
        self.doctor: str = str(doctor).strip()
        self.department: str = str(department).strip()
        self.treatment: str = str(treatment).strip()
        self.treatment_cost: float = float(treatment_cost)
        self.treatment_outcome: str = str(treatment_outcome).strip()
        
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        self.registration_date: str = registration_date or today_str
        self.admission_date: str = admission_date or self.registration_date
        self.discharge_date: str = discharge_date

    def to_dict(self) -> Dict[str, Any]:
        """Convert Patient object attributes to a standard Python Dictionary."""
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "phone": self.phone,
            "address": self.address,
            "blood_group": self.blood_group,
            "disease": self.disease,
            "doctor": self.doctor,
            "department": self.department,
            "treatment": self.treatment,
            "treatment_cost": self.treatment_cost,
            "treatment_outcome": self.treatment_outcome,
            "admission_date": self.admission_date,
            "discharge_date": self.discharge_date,
            "registration_date": self.registration_date
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Patient":
        """Factory method to instantiate a Patient from a Python Dictionary."""
        return cls(
            patient_id=data.get("patient_id", ""),
            name=data.get("name", ""),
            age=int(data.get("age", 0)),
            gender=data.get("gender", "Other"),
            phone=str(data.get("phone", "")),
            address=data.get("address", ""),
            blood_group=data.get("blood_group", ""),
            disease=data.get("disease", ""),
            doctor=data.get("doctor", ""),
            department=data.get("department", "General Medicine"),
            treatment=data.get("treatment", "Standard Care"),
            treatment_cost=float(data.get("treatment_cost", 0.0)),
            treatment_outcome=data.get("treatment_outcome", "Under Treatment"),
            admission_date=data.get("admission_date"),
            discharge_date=data.get("discharge_date", "Ongoing"),
            registration_date=data.get("registration_date")
        )


class PatientManager:
    """
    Manages collection of patient records using Python Lists and Dictionaries,
    with persistent CSV storage, CRUD operations, searching, and validation.
    """
    def __init__(self, csv_filepath: str = PATIENTS_CSV):
        self.csv_filepath = csv_filepath
        # In-memory storage using Python List of Dictionaries
        self.patients_list: List[Dict[str, Any]] = []
        self._ensure_storage_ready()
        self.load_patients()

    def _ensure_storage_ready(self) -> None:
        """Ensure data directory and sample dataset exist."""
        os.makedirs(os.path.dirname(self.csv_filepath), exist_ok=True)
        if not os.path.exists(self.csv_filepath):
            try:
                from data.seed_data import generate_sample_data
                generate_sample_data()
            except Exception as e:
                # If seed_data cannot run, create empty CSV with required headers
                empty_df = pd.DataFrame(columns=[
                    "patient_id", "name", "age", "gender", "phone", "address",
                    "blood_group", "disease", "doctor", "department", "treatment",
                    "treatment_cost", "treatment_outcome", "admission_date",
                    "discharge_date", "registration_date"
                ])
                empty_df.to_csv(self.csv_filepath, index=False)

    def load_patients(self) -> List[Dict[str, Any]]:
        """
        Load patient records from CSV file into Python List of Dictionaries.
        Demonstrates Exception Handling and File Reading.
        """
        try:
            if os.path.exists(self.csv_filepath):
                df = pd.read_csv(self.csv_filepath)
                # Fill missing values cleanly
                df["phone"] = df["phone"].astype(str).str.replace(".0", "", regex=False)
                df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
                df["treatment_cost"] = pd.to_numeric(df["treatment_cost"], errors="coerce").fillna(0.0)
                # Convert DataFrame to Python List of Dictionaries
                self.patients_list = df.to_dict(orient="records")
            else:
                self.patients_list = []
        except Exception as e:
            print(f"[Error loading patients]: {e}")
            self.patients_list = []
        return self.patients_list

    def save_patients(self) -> bool:
        """
        Save the in-memory Python List of Dictionaries to CSV.
        Demonstrates Exception Handling and File Writing.
        """
        try:
            df = pd.DataFrame(self.patients_list)
            df.to_csv(self.csv_filepath, index=False)
            return True
        except Exception as e:
            print(f"[Error saving patients]: {e}")
            return False

    def generate_patient_id(self) -> str:
        """
        Automatically generate a unique Patient ID in the format 'PAT-XXXX'.
        Demonstrates Loops, Conditionals, and String Parsing.
        """
        max_num = 1000
        for patient in self.patients_list:
            pid = str(patient.get("patient_id", ""))
            match = re.search(r"PAT-(\d+)", pid)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        return f"PAT-{max_num + 1}"

    def validate_patient_data(self, data: Dict[str, Any], is_update: bool = False) -> Tuple[bool, str]:
        """
        Validates patient input fields.
        Demonstrates Conditional Statements, Type Checking, and Regular Expressions.
        """
        # Validate Required Fields
        required_keys = ["name", "age", "gender", "phone", "address", "blood_group", "disease", "doctor", "department"]
        for key in required_keys:
            val = data.get(key)
            if val is None or str(val).strip() == "":
                return False, f"Field '{key.replace('_', ' ').title()}' is required and cannot be empty."

        # Validate Name (at least 2 letters, alphabetic + spaces)
        name = str(data.get("name", "")).strip()
        if len(name) < 2 or not re.match(r"^[A-Za-z\s\.\'-]+$", name):
            return False, "Patient Name must be at least 2 characters and contain only letters and spaces."

        # Validate Age (integer between 0 and 125)
        try:
            age = int(data.get("age", 0))
            if age <= 0 or age > 120:
                return False, "Age must be a valid number between 1 and 120."
        except (ValueError, TypeError):
            return False, "Age must be an integer."

        # Validate Phone (must be a valid 10-digit number)
        phone = str(data.get("phone", "")).strip()
        if not re.match(r"^\d{10}$", phone):
            return False, "Phone Number must be a valid 10-digit numeric string."

        # Validate Patient ID uniqueness for new registrations
        pid = str(data.get("patient_id", "")).strip()
        if not pid:
            return False, "Patient ID cannot be empty."

        if not is_update:
            for patient in self.patients_list:
                if str(patient.get("patient_id", "")).strip().upper() == pid.upper():
                    return False, f"Duplicate Patient ID: '{pid}' already exists in the system."

        # Validate Cost
        try:
            cost = float(data.get("treatment_cost", 0.0))
            if cost < 0:
                return False, "Treatment cost cannot be negative."
        except (ValueError, TypeError):
            return False, "Treatment cost must be a numeric value."

        return True, "Validation successful."

    def register_patient(self, patient_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Registers a new patient and stores the record in memory and CSV file.
        Demonstrates List Appending, Validation, and Exception Handling.
        """
        is_valid, message = self.validate_patient_data(patient_dict, is_update=False)
        if not is_valid:
            return False, message

        try:
            # Instantiate Patient object to clean and normalize values
            patient_obj = Patient.from_dict(patient_dict)
            clean_dict = patient_obj.to_dict()

            # Append to Python List of Dictionaries
            self.patients_list.append(clean_dict)

            # Persist to CSV file
            saved = self.save_patients()
            if saved:
                return True, f"Patient {clean_dict['name']} (ID: {clean_dict['patient_id']}) successfully registered."
            else:
                return False, "Failed to write patient record to storage file."
        except Exception as e:
            return False, f"Unexpected error during registration: {str(e)}"

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Return the current list of all patient dictionaries."""
        return self.patients_list

    def get_all_patients_df(self) -> pd.DataFrame:
        """
        Return all patient records as a Pandas DataFrame.
        Demonstrates integration between Python Lists and Pandas.
        """
        if not self.patients_list:
            return pd.DataFrame()
        return pd.DataFrame(self.patients_list)

    def get_patient_by_id(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Search for a patient by exact Patient ID.
        Demonstrates Python Loops and Conditionals.
        """
        target = str(patient_id).strip().upper()
        for patient in self.patients_list:
            if str(patient.get("patient_id", "")).strip().upper() == target:
                return patient
        return None

    def search_patients(self, query: str, search_by: str = "All") -> pd.DataFrame:
        """
        Search patient records by ID, Name, or All fields.
        Demonstrates Pandas String Operations and Filtering.
        """
        df = self.get_all_patients_df()
        if df.empty or not query.strip():
            return df

        q = query.strip()
        if search_by == "Patient ID":
            filtered = df[df["patient_id"].astype(str).str.contains(q, case=False, na=False)]
        elif search_by == "Patient Name":
            filtered = df[df["name"].astype(str).str.contains(q, case=False, na=False)]
        else:
            # Search both ID and Name
            filtered = df[
                df["patient_id"].astype(str).str.contains(q, case=False, na=False) |
                df["name"].astype(str).str.contains(q, case=False, na=False)
            ]
        return filtered

    def filter_patients(
        self,
        gender: Optional[str] = None,
        disease: Optional[str] = None,
        min_age: int = 0,
        max_age: int = 120,
        department: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter patient records based on multiple healthcare criteria.
        Demonstrates Pandas Boolean Masking and Multi-Condition Filtering.
        """
        df = self.get_all_patients_df()
        if df.empty:
            return df

        # Age filtering
        df = df[(df["age"] >= min_age) & (df["age"] <= max_age)]

        # Gender filtering
        if gender and gender != "All":
            df = df[df["gender"] == gender]

        # Disease filtering
        if disease and disease != "All":
            df = df[df["disease"] == disease]

        # Department filtering
        if department and department != "All":
            df = df[df["department"] == department]

        # Outcome filtering
        if outcome and outcome != "All":
            df = df[df["treatment_outcome"] == outcome]

        return df

    def update_patient(self, patient_id: str, updated_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update an existing patient record.
        Demonstrates List modification, Validation, and Persistence.
        """
        is_valid, message = self.validate_patient_data(updated_data, is_update=True)
        if not is_valid:
            return False, message

        target = str(patient_id).strip().upper()
        found_idx = -1
        for idx, patient in enumerate(self.patients_list):
            if str(patient.get("patient_id", "")).strip().upper() == target:
                found_idx = idx
                break

        if found_idx == -1:
            return False, f"Patient ID '{patient_id}' not found."

        try:
            # Preserve existing ID and update fields
            updated_data["patient_id"] = self.patients_list[found_idx]["patient_id"]
            patient_obj = Patient.from_dict(updated_data)
            self.patients_list[found_idx] = patient_obj.to_dict()

            self.save_patients()
            return True, f"Patient record for '{updated_data.get('name')}' updated successfully."
        except Exception as e:
            return False, f"Failed to update patient record: {str(e)}"

    def delete_patient(self, patient_id: str) -> Tuple[bool, str]:
        """
        Delete a patient record from memory and CSV.
        Demonstrates Python List deletion and Exception Handling.
        """
        target = str(patient_id).strip().upper()
        found_idx = -1
        patient_name = ""
        for idx, patient in enumerate(self.patients_list):
            if str(patient.get("patient_id", "")).strip().upper() == target:
                found_idx = idx
                patient_name = patient.get("name", "Unknown")
                break

        if found_idx == -1:
            return False, f"Patient ID '{patient_id}' not found."

        try:
            # Remove from Python List
            del self.patients_list[found_idx]
            self.save_patients()
            return True, f"Patient '{patient_name}' (ID: {patient_id}) deleted successfully."
        except Exception as e:
            return False, f"Error deleting patient: {str(e)}"
