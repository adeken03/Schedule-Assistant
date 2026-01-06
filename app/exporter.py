from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import QMarginsF, QRectF, Qt
from PySide6.QtGui import QFont, QPainter, QPageLayout, QPageSize, QPdfWriter

from database import Employee, EmployeeSessionLocal, SessionLocal, Shift, WeekSchedule


DATA_DIR = Path(__file__).resolve().parent / "data" / "exports"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def _format_shift_time(value: datetime.datetime | None) -> str:
    if not isinstance(value, datetime.datetime):
        return ""
    if value.tzinfo is not None:
        value = value.astimezone()
    return value.strftime("%H:%M")


def _format_shift_date(value: datetime.datetime | None) -> str:
    if not isinstance(value, datetime.datetime):
        return ""
    if value.tzinfo is not None:
        value = value.astimezone()
    return value.strftime("%Y-%m-%d")


def _load_week_export_rows(week_id: int) -> Dict[str, List[Dict[str, str]]]:
    with SessionLocal() as schedule_session, EmployeeSessionLocal() as employee_session:
        week = schedule_session.get(WeekSchedule, week_id)
        if not week:
            raise ValueError(f"Week with id {week_id} not found.")
        shifts = list(
            schedule_session.query(Shift)
            .filter(Shift.week_id == week_id)
            .order_by(Shift.start, Shift.end)
        )
        employees = {emp.id: emp for emp in employee_session.query(Employee).all()}
    rows: List[Dict[str, str]] = []
    for shift in shifts:
        employee = employees.get(shift.employee_id) if shift.employee_id else None
        start = shift.start
        day_label = WEEKDAY_LABELS[start.weekday()] if isinstance(start, datetime.datetime) else ""
        rows.append(
            {
                "day": day_label,
                "date": _format_shift_date(start),
                "start": _format_shift_time(start),
                "end": _format_shift_time(shift.end),
                "role": shift.role,
                "employee": employee.full_name if employee else "Unassigned",
                "location": shift.location,
                "notes": shift.notes,
                "labor_rate": f"{shift.labor_rate:.2f}" if shift.labor_rate else "",
                "labor_cost": f"{shift.labor_cost:.2f}" if shift.labor_cost else "",
            }
        )
    return {"week_label": week.label, "rows": rows}


def _export_week_csv(filename: Path, rows: List[Dict[str, str]]) -> None:
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Day",
                "Date",
                "Start",
                "End",
                "Role",
                "Employee",
                "Location",
                "Notes",
                "Labor Rate",
                "Labor Cost",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["day"],
                    row["date"],
                    row["start"],
                    row["end"],
                    row["role"],
                    row["employee"],
                    row["location"],
                    row["notes"],
                    row["labor_rate"],
                    row["labor_cost"],
                ]
            )


def _export_week_pdf(filename: Path, week_label: str, rows: List[Dict[str, str]]) -> None:
    writer = QPdfWriter(str(filename))
    layout = QPageLayout(
        QPageSize(QPageSize.Letter),
        QPageLayout.Landscape,
        QMarginsF(36, 36, 36, 36),
    )
    writer.setPageLayout(layout)
    painter = QPainter(writer)
    try:
        page_rect = writer.pageLayout().paintRectPixels(writer.resolution())
        margin_left = page_rect.left()
        margin_top = page_rect.top()
        content_width = page_rect.width()
        x = margin_left
        y = margin_top

        title_font = QFont("Segoe UI", 14, QFont.Bold)
        body_font = QFont("Segoe UI", 9)
        header_font = QFont("Segoe UI", 9, QFont.Bold)

        painter.setFont(title_font)
        painter.drawText(QRectF(x, y, content_width, 24), Qt.AlignLeft, f"Schedule Export — {week_label}")
        y += 32

        columns = [
            ("Day", 50),
            ("Date", 75),
            ("Start", 45),
            ("End", 45),
            ("Role", 140),
            ("Employee", 120),
            ("Location", 90),
            ("Notes", 180),
        ]
        scale = content_width / sum(width for _, width in columns)
        columns = [(label, int(width * scale)) for label, width in columns]

        row_height = 18
        painter.setFont(header_font)
        col_x = x
        for label, width in columns:
            painter.drawText(QRectF(col_x, y, width, row_height), Qt.AlignLeft | Qt.AlignVCenter, label)
            col_x += width
        y += row_height + 4

        painter.setFont(body_font)
        for row in rows:
            if y + row_height > page_rect.bottom() - 12:
                writer.newPage()
                y = margin_top
                painter.setFont(header_font)
                col_x = x
                for label, width in columns:
                    painter.drawText(QRectF(col_x, y, width, row_height), Qt.AlignLeft | Qt.AlignVCenter, label)
                    col_x += width
                y += row_height + 4
                painter.setFont(body_font)
            values = [
                row["day"],
                row["date"],
                row["start"],
                row["end"],
                row["role"],
                row["employee"],
                row["location"],
                row["notes"],
            ]
            col_x = x
            for value, (_label, width) in zip(values, columns):
                painter.drawText(QRectF(col_x, y, width, row_height), Qt.AlignLeft | Qt.AlignVCenter, value)
                col_x += width
            y += row_height
    finally:
        painter.end()


def export_week(week_id: int, format: str = "pdf") -> Path:
    """Export a week to PDF or CSV and return the output path."""
    format = format.lower()
    if format not in {"pdf", "csv"}:
        raise ValueError("format must be 'pdf' or 'csv'")
    filename = DATA_DIR / f"week_{week_id}.{format}"
    payload = _load_week_export_rows(week_id)
    rows = payload["rows"]
    if format == "csv":
        _export_week_csv(filename, rows)
    else:
        _export_week_pdf(filename, payload["week_label"], rows)
    return filename
