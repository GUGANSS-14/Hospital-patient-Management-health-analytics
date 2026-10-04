"""
modules/appointment.py
Appointment Management Module for Hospital Patient Management and Health Analytics System.

Demonstrates:
- Object-Oriented Programming (Appointment, AppointmentManager classes)
- Python Lists and Dictionaries for appointment records
- Functions with parameter validation and status transitions
- File Handling: Reading and writing to appointments.csv
- Integration with Patient records
"""

import os
import re
import datetime
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd

# Path configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
APPOINTMENTS_CSV = os.path.join(DATA_DIR, "appointments.csv")


class Appointment:
    """
    Represents an individual hospital consultation appointment.
    Demonstrates Python Object-Oriented Class modeling.
    """
    STATUS_CHOICES = ["Scheduled", "Completed", "Cancelled"]

    def __init__(
        self,
        appointment_id: str,
        patient_id: str,
        patient_name: str,
        doctor_name: str,
        department: str,
        appointment_date: str,
        appointment_time: str,
        status: str = "Scheduled"
    ):
        self.appointment_id: str = str(appointment_id).strip()
        self.patient_id: str = str(patient_id).strip()
        self.patient_name: str = str(patient_name).strip()
        self.doctor_name: str = str(doctor_name).strip()
        self.department: str = str(department).strip()
        self.appointment_date: str = str(appointment_date).strip()
        self.appointment_time: str = str(appointment_time).strip()
        self.status: str = status.strip().capitalize() if status in self.STATUS_CHOICES else "Scheduled"

    def to_dict(self) -> Dict[str, Any]:
        """Convert Appointment object attributes to Python Dictionary."""
        return {
            "appointment_id": self.appointment_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "doctor_name": self.doctor_name,
            "department": self.department,
            "appointment_date": self.appointment_date,
            "appointment_time": self.appointment_time,
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Appointment":
        """Factory method to instantiate Appointment from a Python Dictionary."""
        return cls(
            appointment_id=data.get("appointment_id", ""),
            patient_id=data.get("patient_id", ""),
            patient_name=data.get("patient_name", ""),
            doctor_name=data.get("doctor_name", ""),
            department=data.get("department", "General Medicine"),
            appointment_date=str(data.get("appointment_date", "")),
            appointment_time=str(data.get("appointment_time", "")),
            status=data.get("status", "Scheduled")
        )


class AppointmentManager:
    """
    Manages hospital appointment schedules, booking, status updating,
    searching, and persistence using CSV files.
    """
    def __init__(self, csv_filepath: str = APPOINTMENTS_CSV):
        self.csv_filepath = csv_filepath
        self.appointments_list: List[Dict[str, Any]] = []
        self._ensure_storage_ready()
        self.load_appointments()

    def _ensure_storage_ready(self) -> None:
        """Ensure appointments CSV exists; if missing, trigger generator."""
        os.makedirs(os.path.dirname(self.csv_filepath), exist_ok=True)
        if not os.path.exists(self.csv_filepath):
            try:
                from data.seed_data import generate_sample_data
                generate_sample_data()
            except Exception as e:
                empty_df = pd.DataFrame(columns=[
                    "appointment_id", "patient_id", "patient_name", "doctor_name",
                    "department", "appointment_date", "appointment_time", "status"
                ])
                empty_df.to_csv(self.csv_filepath, index=False)

    def load_appointments(self) -> List[Dict[str, Any]]:
        """Load appointments from CSV into Python list of dictionaries."""
        try:
            if os.path.exists(self.csv_filepath):
                df = pd.read_csv(self.csv_filepath)
                df["appointment_date"] = df["appointment_date"].astype(str)
                self.appointments_list = df.to_dict(orient="records")
            else:
                self.appointments_list = []
        except Exception as e:
            print(f"[Error loading appointments]: {e}")
            self.appointments_list = []
        return self.appointments_list

    def save_appointments(self) -> bool:
        """Persist appointment list to CSV."""
        try:
            df = pd.DataFrame(self.appointments_list)
            df.to_csv(self.csv_filepath, index=False)
            return True
        except Exception as e:
            print(f"[Error saving appointments]: {e}")
            return False

    def generate_appointment_id(self) -> str:
        """
        Generate unique Appointment ID formatted as 'APT-XXXX'.
        Demonstrates loops, regular expressions, and numeric tracking.
        """
        max_num = 2000
        for apt in self.appointments_list:
            aid = str(apt.get("appointment_id", ""))
            match = re.search(r"APT-(\d+)", aid)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        return f"APT-{max_num + 1}"

    def validate_appointment_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate required fields, dates, and status."""
        required = ["patient_id", "patient_name", "doctor_name", "department", "appointment_date", "appointment_time"]
        for field in required:
            val = data.get(field)
            if val is None or str(val).strip() == "":
                return False, f"Field '{field.replace('_', ' ').title()}' is required."

        status = data.get("status", "Scheduled")
        if status not in Appointment.STATUS_CHOICES:
            return False, f"Invalid status '{status}'. Must be one of {Appointment.STATUS_CHOICES}."

        # Validate date format (YYYY-MM-DD)
        date_str = str(data.get("appointment_date", ""))
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return False, "Appointment Date must be in YYYY-MM-DD format."

        return True, "Validation successful."

    def add_appointment(self, appointment_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Add a new appointment with validation and file persistence.
        Demonstrates Dictionary operations, Class instantiation, and Error handling.
        """
        is_valid, msg = self.validate_appointment_data(appointment_dict)
        if not is_valid:
            return False, msg

        try:
            # Check for duplicate appointment ID
            apt_id = appointment_dict.get("appointment_id", "")
            if not apt_id:
                apt_id = self.generate_appointment_id()
                appointment_dict["appointment_id"] = apt_id
            else:
                for a in self.appointments_list:
                    if str(a.get("appointment_id", "")).strip().upper() == apt_id.strip().upper():
                        return False, f"Appointment ID '{apt_id}' already exists."

            appointment_obj = Appointment.from_dict(appointment_dict)
            self.appointments_list.append(appointment_obj.to_dict())
            self.save_appointments()
            return True, f"Appointment {apt_id} successfully booked for {appointment_obj.patient_name}."
        except Exception as e:
            return False, f"Failed to book appointment: {str(e)}"

    def get_all_appointments(self) -> List[Dict[str, Any]]:
        """Return the current list of appointment dictionaries."""
        return self.appointments_list

    def get_appointments_df(self) -> pd.DataFrame:
        """Return appointment records as a Pandas DataFrame."""
        if not self.appointments_list:
            return pd.DataFrame(columns=[
                "appointment_id", "patient_id", "patient_name", "doctor_name",
                "department", "appointment_date", "appointment_time", "status"
            ])
        return pd.DataFrame(self.appointments_list)

    def search_appointments(self, query: str, status_filter: str = "All") -> pd.DataFrame:
        """
        Search appointments by Patient Name, ID, or Doctor Name, and optional Status filter.
        Demonstrates Pandas filtering and string manipulation.
        """
        df = self.get_appointments_df()
        if df.empty:
            return df

        if status_filter and status_filter != "All":
            df = df[df["status"] == status_filter]

        if query.strip():
            q = query.strip()
            df = df[
                df["patient_name"].astype(str).str.contains(q, case=False, na=False) |
                df["patient_id"].astype(str).str.contains(q, case=False, na=False) |
                df["doctor_name"].astype(str).str.contains(q, case=False, na=False) |
                df["appointment_id"].astype(str).str.contains(q, case=False, na=False)
            ]
        return df

    def update_appointment_status(self, appointment_id: str, new_status: str) -> Tuple[bool, str]:
        """
        Update the status of an appointment (Scheduled, Completed, Cancelled).
        Demonstrates List traversal, item modification, and state persistence.
        """
        if new_status not in Appointment.STATUS_CHOICES:
            return False, f"Invalid status: '{new_status}'. Allowed: {Appointment.STATUS_CHOICES}"

        target = str(appointment_id).strip().upper()
        found = False
        for apt in self.appointments_list:
            if str(apt.get("appointment_id", "")).strip().upper() == target:
                apt["status"] = new_status
                found = True
                break

        if not found:
            return False, f"Appointment ID '{appointment_id}' not found."

        self.save_appointments()
        return True, f"Appointment {appointment_id} status updated to '{new_status}'."

    def delete_appointment(self, appointment_id: str) -> Tuple[bool, str]:
        """Delete an appointment by ID."""
        target = str(appointment_id).strip().upper()
        found_idx = -1
        for idx, apt in enumerate(self.appointments_list):
            if str(apt.get("appointment_id", "")).strip().upper() == target:
                found_idx = idx
                break

        if found_idx == -1:
            return False, f"Appointment ID '{appointment_id}' not found."

        try:
            del self.appointments_list[found_idx]
            self.save_appointments()
            return True, f"Appointment {appointment_id} deleted successfully."
        except Exception as e:
            return False, f"Error deleting appointment: {str(e)}"
