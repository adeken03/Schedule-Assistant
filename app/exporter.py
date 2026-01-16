from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Any, Dict, List

from database import EmployeeSessionLocal, SessionLocal, WeekSchedule, get_shifts_for_week


DATA_DIR = Path(__file__).resolve().parent / "data" / "exports"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PDF_PAGE_WIDTH = 612
PDF_PAGE_HEIGHT = 792
PDF_MARGIN_LEFT = 54
PDF_MARGIN_TOP = 72
PDF_MARGIN_BOTTOM = 72
PDF_FONT_SIZE = 10
PDF_LINE_HEIGHT = 12


def _format_datetime(value: Any) -> str:
    if isinstance(value, datetime.datetime):
        return value.isoformat(sep=" ", timespec="minutes")
    return str(value) if value is not None else ""


def _format_time(value: Any) -> str:
    if isinstance(value, datetime.datetime):
        return value.strftime("%H:%M")
    return ""


def _load_week_export(week_id: int) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with SessionLocal() as session:
        week = session.get(WeekSchedule, week_id)
        if not week:
            raise ValueError(f"WeekSchedule with id {week_id} was not found.")
        week_data = {
            "id": week.id,
            "iso_year": week.iso_year,
            "iso_week": week.iso_week,
            "label": week.label,
            "week_start_date": week.week_start_date,
        }
        with EmployeeSessionLocal() as employee_session:
            shifts = get_shifts_for_week(
                session,
                week.week_start_date,
                employee_session=employee_session,
            )
    return week_data, shifts


def _write_csv(path: Path, shifts: List[Dict[str, Any]]) -> None:
    headers = [
        "role",
        "start",
        "end",
        "location",
        "notes",
        "status",
        "labor_rate",
        "labor_cost",
        "employee_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for shift in shifts:
            writer.writerow(
                [
                    shift.get("role") or "",
                    _format_datetime(shift.get("start")),
                    _format_datetime(shift.get("end")),
                    shift.get("location") or "",
                    shift.get("notes") or "",
                    shift.get("status") or "",
                    shift.get("labor_rate", ""),
                    shift.get("labor_cost", ""),
                    shift.get("employee_name") or "",
                ]
            )


def _pdf_escape(value: str) -> str:
    sanitized = value.encode("ascii", "replace").decode("ascii")
    sanitized = sanitized.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return sanitized


def _chunk_lines(lines: List[str], lines_per_page: int) -> List[List[str]]:
    if lines_per_page <= 0:
        return [lines]
    return [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]


def _write_pdf(path: Path, title: str, shifts: List[Dict[str, Any]]) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"Schedule Export - {title}",
        f"Generated: {now}",
        "",
        "Date       Start  End    Role                        Employee            Loc     Cost",
        "-" * 86,
    ]
    for shift in shifts:
        start = shift.get("start")
        end = shift.get("end")
        date_label = start.date().isoformat() if isinstance(start, datetime.datetime) else ""
        start_label = _format_time(start)
        end_label = _format_time(end)
        role = (shift.get("role") or "")[:26]
        employee = (shift.get("employee_name") or "")[:18]
        location = (shift.get("location") or "")[:8]
        cost = shift.get("labor_cost")
        try:
            cost_label = f"{float(cost):.2f}" if cost is not None else ""
        except (TypeError, ValueError):
            cost_label = ""
        lines.append(
            f"{date_label} {start_label:>5} {end_label:>5} {role:<26} {employee:<18} {location:<8} {cost_label:>7}"
        )

    usable_height = PDF_PAGE_HEIGHT - PDF_MARGIN_TOP - PDF_MARGIN_BOTTOM
    lines_per_page = max(1, int(usable_height / PDF_LINE_HEIGHT))
    pages = _chunk_lines(lines, lines_per_page)

    header = b"%PDF-1.4\n"
    parts: List[bytes] = [header]
    offsets: List[int] = []
    cursor = len(header)

    def add_object(obj_id: int, content: str) -> None:
        nonlocal cursor
        obj_bytes = f"{obj_id} 0 obj\n{content}\nendobj\n".encode("latin-1")
        offsets.append(cursor)
        parts.append(obj_bytes)
        cursor += len(obj_bytes)

    page_ids = [4 + idx * 2 for idx in range(len(pages))]
    add_object(1, "<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    add_object(2, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")
    add_object(3, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for idx, page_lines in enumerate(pages):
        page_id = page_ids[idx]
        content_id = page_id + 1
        add_object(
            page_id,
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PDF_PAGE_WIDTH} {PDF_PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>",
        )
        stream_lines: List[str] = [
            "BT",
            f"/F1 {PDF_FONT_SIZE} Tf",
            f"{PDF_MARGIN_LEFT} {PDF_PAGE_HEIGHT - PDF_MARGIN_TOP} Td",
        ]
        for line_index, raw_line in enumerate(page_lines):
            if line_index:
                stream_lines.append(f"0 -{PDF_LINE_HEIGHT} Td")
            stream_lines.append(f"({_pdf_escape(raw_line)}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines)
        stream_bytes = stream.encode("latin-1")
        content = f"<< /Length {len(stream_bytes)} >>\nstream\n{stream}\nendstream"
        add_object(content_id, content)

    xref_offset = cursor
    xref_lines = ["xref", f"0 {len(offsets) + 1}", "0000000000 65535 f "]
    xref_lines.extend(f"{offset:010d} 00000 n " for offset in offsets)
    xref = "\n".join(xref_lines) + "\n"
    trailer = (
        "trailer\n"
        f"<< /Size {len(offsets) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    parts.append(xref.encode("latin-1"))
    parts.append(trailer.encode("latin-1"))
    path.write_bytes(b"".join(parts))


def export_week(week_id: int, format: str = "pdf") -> Path:
    format = (format or "pdf").lower()
    if format not in {"pdf", "csv"}:
        raise ValueError("format must be 'pdf' or 'csv'")
    week, shifts = _load_week_export(int(week_id))
    filename = DATA_DIR / f"week_{week['id']}.{format}"
    if format == "csv":
        _write_csv(filename, shifts)
        return filename
    _write_pdf(filename, week.get("label", f"Week {week['id']}"), shifts)
    return filename
    format = format.lower()
    if format not in {"pdf", "csv"}:
        raise ValueError("format must be 'pdf' or 'csv'")
    filename = DATA_DIR / f"week_{week_id}.{format}"
    filename.write_text(
        f"Placeholder export for week {week_id} ({format.upper()})",
        encoding="utf-8",
    )
    return filename
