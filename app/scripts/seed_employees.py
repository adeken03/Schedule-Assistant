from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database import Employee, EmployeeUnavailability, EmployeeSessionLocal, init_database
from roles import defined_roles


DAY_INDEX = {
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
    "Sat": 5,
    "Sun": 6,
}

VALID_ROLES = set(defined_roles())
DEFAULT_START_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def parse_time(label: str) -> datetime.time:
    hour, minute = [int(part) for part in label.split(":", 1)]
    hour = min(max(hour, 0), 23)
    minute = min(max(minute, 0), 59)
    return datetime.time(hour, minute)


def add_unavailability_rows(session, employee_id: int, rows: Dict[str, List[Tuple[str, str]]]) -> None:
    for day_name, windows in rows.items():
        day_idx = DAY_INDEX.get(day_name, 0)
        for start_label, end_label in windows:
            session.add(
                EmployeeUnavailability(
                    employee_id=employee_id,
                    day_of_week=day_idx,
                    start_time=parse_time(start_label),
                    end_time=parse_time(end_label),
                )
            )


def normalize_roles(roles: List[str], employee_name: str) -> List[str]:
    cleaned: List[str] = []
    missing: List[str] = []
    for role in roles:
        if role in VALID_ROLES:
            cleaned.append(role)
        else:
            missing.append(role)
    if missing:
        print(f"[seed] Skipping undefined roles for {employee_name}: {', '.join(missing)}")
    return cleaned


def resolve_start_fields(entry: Dict, index: int) -> Tuple[int, int]:
    month = entry.get("start_month") or ((index % 12) + 1)
    year = entry.get("start_year") or DEFAULT_START_YEARS[index % len(DEFAULT_START_YEARS)]
    return month, year


def build_notes(entry: Dict, roles: List[str]) -> str:
    if entry.get("notes"):
        return entry["notes"]
    if not roles:
        return "Flexible coverage; set roles before scheduling."
    focus = roles[0]
    desired = entry.get("desired_hours", 25)
    return f"Prefers {focus.lower()} shifts (~{desired} hrs/week target)."


SAMPLE_EMPLOYEES: List[Dict] = [
    # Servers - Dining
    {
        "name": "Alex Nguyen",
        "roles": ["Server - Dining", "Server - Dining Closer"],
        "desired_hours": 28,
        "unavailability": {"Mon": [("08:00", "11:00")], "Wed": [("09:00", "12:00")]},
    },
    {
        "name": "Maya Thompson",
        "roles": ["Server - Dining", "Server - Dining Opener"],
        "desired_hours": 30,
        "unavailability": {"Tue": [("08:00", "11:00")], "Thu": [("08:30", "11:30")]},
    },
    {
        "name": "Jordan Ellis",
        "roles": ["Server - Dining", "Server - Dining Preclose"],
        "desired_hours": 26,
        "unavailability": {"Fri": [("09:00", "12:00")]},
    },
    {
        "name": "Sofia Ramirez",
        "roles": ["Server - Dining", "Server - Patio"],
        "desired_hours": 24,
        "unavailability": {"Tue": [("10:00", "13:00")], "Thu": [("10:00", "13:00")]},
    },
    {
        "name": "Logan Patel",
        "roles": ["Server - Dining", "Server - Dining Closer"],
        "desired_hours": 32,
        "unavailability": {"Wed": [("08:00", "11:00")]},
    },
    {
        "name": "Harper Reed",
        "roles": ["Server - Dining", "Server - Dining Opener"],
        "desired_hours": 27,
        "unavailability": {"Mon": [("07:30", "10:30")], "Fri": [("08:30", "11:30")]},
    },
    # Servers - Cocktail
    {
        "name": "Noah Price",
        "roles": ["Server - Cocktail", "Server - Cocktail Closer"],
        "desired_hours": 33,
        "unavailability": {"Tue": [("11:00", "14:00")]},
    },
    {
        "name": "Avery Brooks",
        "roles": ["Server - Cocktail", "Server - Cocktail Opener"],
        "desired_hours": 25,
        "unavailability": {"Thu": [("09:00", "12:00")], "Fri": [("09:00", "11:30")]},
    },
    {
        "name": "Caleb Foster",
        "roles": ["Server - Cocktail", "Server - Patio"],
        "desired_hours": 20,
        "unavailability": {"Mon": [("08:00", "12:00")], "Wed": [("14:00", "16:00")]},
    },
    {
        "name": "Lena Brooks",
        "roles": ["Server - Cocktail", "Server - Cocktail Preclose"],
        "desired_hours": 22,
        "unavailability": {"Tue": [("09:30", "12:30")], "Thu": [("09:30", "12:30")]},
    },
    {
        "name": "Elias Carter",
        "roles": ["Server - Dining", "Server - Cocktail"],
        "desired_hours": 34,
        "unavailability": {"Sun": [("10:00", "13:00")]},
    },
    {
        "name": "Zoe Armstrong",
        "roles": ["Server - Dining", "Server - Dining Closer"],
        "desired_hours": 29,
        "unavailability": {"Wed": [("08:00", "11:00")], "Fri": [("08:00", "11:00")]},
    },
    # Bartenders
    {
        "name": "Ethan Cole",
        "roles": ["Bartender", "Bartender - Closer"],
        "desired_hours": 32,
        "unavailability": {"Mon": [("09:00", "12:00")]},
    },
    {
        "name": "Gianna Lopez",
        "roles": ["Bartender", "Bartender - Opener"],
        "desired_hours": 30,
        "unavailability": {"Wed": [("08:00", "11:00")], "Fri": [("08:00", "11:00")]},
    },
    {
        "name": "Marcus Bennett",
        "roles": ["Bartender", "Bartender - Closer"],
        "desired_hours": 35,
        "unavailability": {"Thu": [("10:00", "13:00")]},
    },
    {
        "name": "Devon Brooks",
        "roles": ["Bartender", "Bartender - Opener"],
        "desired_hours": 28,
        "unavailability": {"Tue": [("09:00", "12:00")], "Thu": [("09:00", "12:00")]},
    },
    # Cashier & Guest Services
    {
        "name": "Riley Patel",
        "roles": ["Cashier", "Cashier - To-Go Specialist"],
        "desired_hours": 24,
        "unavailability": {"Mon": [("08:30", "11:30")], "Wed": [("08:30", "11:30")]},
    },
    {
        "name": "Sadie Hill",
        "roles": ["Cashier", "Host"],
        "desired_hours": 22,
        "unavailability": {"Fri": [("09:00", "12:00")], "Sun": [("09:00", "12:00")]},
    },
    {
        "name": "Owen Rivera",
        "roles": ["Cashier", "Cashier - To-Go Specialist"],
        "desired_hours": 26,
        "unavailability": {"Tue": [("10:00", "13:00")]},
    },
    {
        "name": "Lily Summers",
        "roles": ["Cashier", "Host"],
        "desired_hours": 18,
        "unavailability": {"Thu": [("09:00", "12:00")]},
    },
    {
        "name": "Tyler James",
        "roles": ["Cashier", "Cashier - To-Go Specialist"],
        "desired_hours": 20,
        "unavailability": {"Sun": [("08:00", "11:00")]},
    },
    # Heart of House
    {
        "name": "Maria Lopez",
        "roles": ["Grill", "Kitchen Closer"],
        "desired_hours": 38,
        "unavailability": {"Tue": [("06:00", "09:00")]},
    },
    {
        "name": "Anthony Rhodes",
        "roles": ["Expo", "Shake"],
        "desired_hours": 34,
        "unavailability": {"Mon": [("07:00", "09:00")], "Wed": [("07:00", "09:00")]},
    },
    {
        "name": "Bianca Taylor",
        "roles": ["Prep", "Grill"],
        "desired_hours": 36,
        "unavailability": {"Fri": [("06:00", "09:00")]},
    },
    {
        "name": "Darius Mitchell",
        "roles": ["Chip", "Expo"],
        "desired_hours": 32,
        "unavailability": {"Thu": [("07:30", "10:30")]},
    },
    {
        "name": "Nina Alvarez",
        "roles": ["Shake", "Prep"],
        "desired_hours": 30,
        "unavailability": {"Tue": [("06:30", "09:30")], "Thu": [("06:30", "09:30")]},
    },
    {
        "name": "Peter Zhang",
        "roles": ["Grill", "Expo"],
        "desired_hours": 37,
        "unavailability": {"Sun": [("08:00", "11:00")]},
    },
    {
        "name": "Quinn Foster",
        "roles": ["Chip", "Prep"],
        "desired_hours": 28,
        "unavailability": {"Mon": [("06:00", "09:00")]},
    },
    {
        "name": "Samantha Green",
        "roles": ["Expo", "Kitchen Opener"],
        "desired_hours": 33,
        "unavailability": {"Wed": [("06:30", "09:30")], "Fri": [("06:30", "09:30")]},
    },
    # Additional coverage to fill every role
    {
        "name": "Chloe Martin",
        "roles": ["Cashier - To-Go Specialist", "Cashier"],
        "desired_hours": 23,
        "unavailability": {"Mon": [("11:00", "14:00")], "Thu": [("15:00", "17:00")]},
    },
    {
        "name": "Brandon Yates",
        "roles": ["Cook", "Kitchen Closer"],
        "desired_hours": 38,
        "unavailability": {"Wed": [("05:00", "08:00")]},
    },
    {
        "name": "Emily Carter",
        "roles": ["Cook", "Prep"],
        "desired_hours": 34,
        "unavailability": {"Tue": [("06:00", "09:00")], "Thu": [("06:00", "09:00")]},
    },
    {
        "name": "Felix Moore",
        "roles": ["Bartender", "Server - Cocktail Preclose"],
        "desired_hours": 30,
        "unavailability": {"Mon": [("10:00", "12:00")], "Fri": [("10:00", "12:00")]},
    },
    {
        "name": "Gwen Hollis",
        "roles": ["Server - Dining", "Expo"],
        "desired_hours": 28,
        "unavailability": {"Wed": [("12:00", "15:00")], "Sat": [("08:00", "10:00")]},
    },
    {
        "name": "Hector Silva",
        "roles": ["Cook", "Kitchen Closer"],
        "desired_hours": 32,
        "unavailability": {"Sun": [("07:00", "10:00")]},
    },
    {
        "name": "Ivy Tran",
        "roles": ["Cashier - To-Go Specialist", "Server - Dining Opener"],
        "desired_hours": 26,
        "unavailability": {"Tue": [("08:00", "11:00")], "Sat": [("09:00", "12:00")]},
    },
    {
        "name": "Jesse Park",
        "roles": ["Host", "Prep"],
        "desired_hours": 22,
        "unavailability": {"Thu": [("13:00", "16:00")]},
    },
    {
        "name": "Luca Fernandez",
        "roles": ["Cook", "Grill"],
        "desired_hours": 37,
        "unavailability": {"Mon": [("05:00", "08:00")], "Fri": [("05:00", "08:00")]},
    },
    {
        "name": "Mara Singh",
        "roles": ["Cashier - To-Go Specialist", "Server - Patio"],
        "desired_hours": 24,
        "unavailability": {"Sun": [("09:00", "12:00")], "Tue": [("12:00", "14:00")]},
    },
    {
        "name": "Nadia Lopez",
        "roles": ["Server - Cocktail", "Server - Patio"],
        "desired_hours": 34,
        "unavailability": {"Wed": [("15:00", "18:00")]},
    },
    {
        "name": "Owen Clark",
        "roles": ["Cashier - To-Go Specialist", "Host"],
        "desired_hours": 29,
        "unavailability": {"Thu": [("09:00", "11:00")]},
    },
    {
        "name": "Priya Desai",
        "roles": ["Kitchen Opener", "Cook"],
        "desired_hours": 36,
        "unavailability": {"Sun": [("05:00", "08:00")]},
    },
    {
        "name": "Ramon Ortiz",
        "roles": ["Kitchen Closer", "Grill"],
        "desired_hours": 37,
        "unavailability": {"Tue": [("08:00", "10:00")]},
    },
    {
        "name": "Tessa Wright",
        "roles": ["Server - Patio", "Server - Dining"],
        "desired_hours": 30,
        "unavailability": {"Mon": [("08:00", "10:00")]},
    },
    {
        "name": "Ulises Vega",
        "roles": ["Chip", "Kitchen Opener"],
        "desired_hours": 33,
        "unavailability": {"Fri": [("05:00", "07:00")]},
    },
    {
        "name": "Viola Chen",
        "roles": ["Shake", "Prep"],
        "desired_hours": 25,
        "unavailability": {"Tue": [("16:00", "18:00")]},
    },
    {
        "name": "Wesley Adams",
        "roles": ["Cashier", "Server - Patio"],
        "desired_hours": 28,
        "unavailability": {"Sat": [("10:00", "12:00")]},
    },
    {
        "name": "Ximena Garza",
        "roles": ["Server - Cocktail", "Bartender"],
        "desired_hours": 32,
        "unavailability": {"Thu": [("10:00", "12:00")]},
    },
    {
        "name": "Yael Monroe",
        "roles": ["Kitchen Opener", "Expo"],
        "desired_hours": 31,
        "unavailability": {"Wed": [("05:30", "07:30")]},
    },
    {
        "name": "Zara Allen",
        "roles": ["Host", "Cashier", "Cashier - To-Go Specialist"],
        "desired_hours": 30,
        "unavailability": {"Tue": [("14:00", "16:00")]},
    },
    {
        "name": "Amir Bennett",
        "roles": ["Host", "Server - Patio"],
        "desired_hours": 28,
        "unavailability": {"Fri": [("13:00", "15:00")]},
    },
    {
        "name": "Bailey Ortiz",
        "roles": ["Chip", "Shake"],
        "desired_hours": 32,
        "unavailability": {"Wed": [("09:00", "11:00")]},
    },
    {
        "name": "Callie Rivers",
        "roles": ["Chip", "Prep"],
        "desired_hours": 30,
        "unavailability": {"Mon": [("06:00", "08:00")]},
    },
    {
        "name": "Damon Ellis",
        "roles": ["Shake", "Expo"],
        "desired_hours": 31,
        "unavailability": {"Thu": [("07:00", "09:00")]},
    },
    {
        "name": "Elise Moran",
        "roles": ["Shake", "Cashier - To-Go Specialist"],
        "desired_hours": 27,
        "unavailability": {"Tue": [("16:00", "18:00")]},
    },
    {
        "name": "Gio Alvarez",
        "roles": ["Server - Dining", "Server - Cocktail"],
        "desired_hours": 33,
        "unavailability": {"Sun": [("08:00", "10:00")]},
    },
    {
        "name": "Hana Kim",
        "roles": ["Server - Patio", "Server - Cocktail"],
        "desired_hours": 29,
        "unavailability": {"Wed": [("12:00", "14:00")]},
    },
    {
        "name": "Isaiah Ford",
        "roles": ["Cashier", "Cashier - To-Go Specialist", "Host"],
        "desired_hours": 31,
        "unavailability": {"Mon": [("11:00", "13:00")]},
    },
    {
        "name": "Jada Pierce",
        "roles": ["Kitchen Closer", "Chip"],
        "desired_hours": 35,
        "unavailability": {"Thu": [("04:00", "06:00")]},
    },
    {
        "name": "Kara Neal",
        "roles": ["Server - Dining Closer", "Server - Cocktail"],
        "desired_hours": 34,
        "unavailability": {"Mon": [("09:00", "11:00")]},
    },
    {
        "name": "Liam Porter",
        "roles": ["Server - Patio", "Server - Dining"],
        "desired_hours": 31,
        "unavailability": {"Wed": [("11:00", "13:00")]},
    },
    {
        "name": "Molly Garrison",
        "roles": ["Server - Cocktail Closer", "Bartender"],
        "desired_hours": 36,
        "unavailability": {"Tue": [("10:00", "12:00")]},
    },
    {
        "name": "Noel Rasmussen",
        "roles": ["Server - Dining Opener", "Server - Patio"],
        "desired_hours": 28,
        "unavailability": {"Sat": [("07:00", "09:00")]},
    },
    {
        "name": "Nora Bell",
        "roles": ["Prep", "Chip"],
        "desired_hours": 30,
        "unavailability": {"Mon": [("06:00", "08:00")]},
    },
    {
        "name": "Oscar Lane",
        "roles": ["Chip", "Shake"],
        "desired_hours": 32,
        "unavailability": {"Wed": [("05:00", "07:00")]},
    },
    {
        "name": "Piper Hart",
        "roles": ["Shake", "Prep"],
        "desired_hours": 29,
        "unavailability": {"Thu": [("06:00", "08:00")]},
    },
    {
        "name": "Riley Shaw",
        "roles": ["Cashier - To-Go Specialist", "Prep"],
        "desired_hours": 27,
        "unavailability": {"Tue": [("14:00", "16:00")]},
    },
    {
        "name": "Sage Donovan",
        "roles": ["Prep", "Kitchen Opener"],
        "desired_hours": 34,
        "unavailability": {"Fri": [("05:00", "07:00")]},
    },
]


def seed_employees() -> None:
    init_database()
    created = 0
    refreshed = 0
    with EmployeeSessionLocal() as session:
        for index, entry in enumerate(SAMPLE_EMPLOYEES):
            roles = normalize_roles(entry.get("roles", []), entry["name"])
            if not roles:
                print(f"[seed] Skipping {entry['name']} because no valid roles remain.")
                continue

            desired_hours = entry.get("desired_hours", 25)
            status = entry.get("status", "active")
            notes = build_notes(entry, roles)
            start_month, start_year = resolve_start_fields(entry, index)

            stmt = select(Employee).where(Employee.full_name == entry["name"])
            employee = session.scalars(stmt).first()
            if not employee:
                employee = Employee(
                    full_name=entry["name"],
                    desired_hours=desired_hours,
                    status=status,
                    notes=notes,
                    start_month=start_month,
                    start_year=start_year,
                )
                employee.role_list = roles
                session.add(employee)
                session.flush()
                add_unavailability_rows(session, employee.id, entry.get("unavailability", {}))
                created += 1
            else:
                employee.desired_hours = desired_hours
                employee.status = status
                employee.notes = notes
                employee.start_month = start_month
                employee.start_year = start_year
                employee.role_list = roles
                refreshed += 1
        session.commit()
    print(f"Seed complete. Created {created} employees, refreshed {refreshed} profiles.")


if __name__ == "__main__":
    seed_employees()
