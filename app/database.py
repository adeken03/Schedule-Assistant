from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json

from sqlalchemy import (
    Date,
    DateTime,
    delete,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
    text,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.types import Time

from roles import is_manager_role


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EMPLOYEE_DATABASE_URL = f"sqlite:///{(DATA_DIR / 'employees.db').as_posix()}"
SCHEDULE_DATABASE_URL = f"sqlite:///{(DATA_DIR / 'schedule.db').as_posix()}"
POLICY_DATABASE_URL = f"sqlite:///{(DATA_DIR / 'policy.db').as_posix()}"
PROJECTIONS_DATABASE_URL = f"sqlite:///{(DATA_DIR / 'projections.db').as_posix()}"
WEEK_STATUS_CHOICES = {"draft", "validated", "exported"}


def _normalize_week_start(date_value: datetime.date) -> datetime.date:
    """Return the Monday for the provided date."""
    if isinstance(date_value, datetime.datetime):
        date_value = date_value.date()
    weekday = date_value.weekday()
    if weekday == 0:
        return date_value
    return date_value - datetime.timedelta(days=weekday)


def _format_week_label(week_start: datetime.date) -> str:
    iso_year, iso_week, _ = week_start.isocalendar()
    end = week_start + datetime.timedelta(days=6)
    start_str = week_start.strftime("%b %d")
    end_str = end.strftime("%b %d")
    if week_start.year != end.year:
        start_str = week_start.strftime("%b %d %Y")
        end_str = end.strftime("%b %d %Y")
    return f"{iso_year} W{iso_week:02d} ({start_str} - {end_str})"


class EmployeeBase(DeclarativeBase):
    """Standalone metadata for employee tables living in employees.db."""

    pass


class PolicyBase(DeclarativeBase):
    """Standalone metadata for policy tables living in policy.db."""

    pass


class ProjectionsBase(DeclarativeBase):
    """Standalone metadata for projections tables living in projections.db."""

    pass


class Base(DeclarativeBase):
    """Metadata for schedule/policy tables living in schedule.db."""

    pass


class Employee(EmployeeBase):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    roles: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    desired_hours: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(12), default="active", nullable=False)
    notes: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    start_month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
    )

    unavailability: Mapped[List["EmployeeUnavailability"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )
    role_wages: Mapped[List["EmployeeRoleWage"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )

    @property
    def role_list(self) -> List[str]:
        return [role.strip() for role in self.roles.split(",") if role.strip()]

    @role_list.setter
    def role_list(self, roles: Iterable[str]) -> None:
        ordered: List[str] = []
        seen: set[str] = set()
        for role in roles:
            label = role.strip()
            if not label or label in seen:
                continue
            seen.add(label)
            ordered.append(label)
        self.roles = ", ".join(ordered)

    @property
    def start_date_label(self) -> str:
        if self.start_month and self.start_year:
            month_name = datetime.date(1900, self.start_month, 1).strftime("%b")
            return f"{month_name} {self.start_year}"
        if self.start_year:
            return str(self.start_year)
        return "Not set"


class EmployeeUnavailability(EmployeeBase):
    __tablename__ = "employee_unavailability"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"))
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 = Monday
    start_time: Mapped[datetime.time] = mapped_column(Time, nullable=False)
    end_time: Mapped[datetime.time] = mapped_column(Time, nullable=False)

    employee: Mapped[Employee] = relationship(back_populates="unavailability")


class EmployeeRoleWage(EmployeeBase):
    __tablename__ = "employee_role_wages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(80), nullable=False)
    wage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confirmed: Mapped[bool] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
    )

    employee: Mapped[Employee] = relationship(back_populates="role_wages")

    __table_args__ = (
        UniqueConstraint("employee_id", "role", name="uq_employee_role_wage_role"),
    )


class WeekContext(Base):
    __tablename__ = "week_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    iso_year: Mapped[int] = mapped_column(Integer, nullable=False)
    iso_week: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )

    __table_args__ = (UniqueConstraint("iso_year", "iso_week", name="uq_week_context_year_week"),)

    modifiers: Mapped[List["Modifier"]] = relationship(
        back_populates="week", cascade="all, delete-orphan"
    )


class WeekProjectionContext(ProjectionsBase):
    __tablename__ = "week_projection_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schedule_context_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    iso_year: Mapped[int] = mapped_column(Integer, nullable=False)
    iso_week: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint("iso_year", "iso_week", name="uq_projection_context_year_week"),
        UniqueConstraint("schedule_context_id", name="uq_projection_schedule_context"),
    )


class WeekDailyProjection(ProjectionsBase):
    __tablename__ = "week_daily_projections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    projection_context_id: Mapped[int] = mapped_column(Integer, nullable=False)
    schedule_context_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    iso_year: Mapped[int] = mapped_column(Integer, nullable=False)
    iso_week: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(40), nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    projected_sales_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    projected_notes: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
    )

    __table_args__ = (UniqueConstraint("projection_context_id", "day_of_week", name="uq_daily_projection_week_day"),)


class Modifier(Base):
    __tablename__ = "modifiers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    week_id: Mapped[int] = mapped_column(ForeignKey("week_context.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(80), nullable=False)
    modifier_type: Mapped[str] = mapped_column(String(24), nullable=False, default="increase")
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[datetime.time] = mapped_column(Time, nullable=False)
    end_time: Mapped[datetime.time] = mapped_column(Time, nullable=False)
    pct_change: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    notes: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    created_by: Mapped[str] = mapped_column(String(60), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
    )

    week: Mapped[WeekContext] = relationship(back_populates="modifiers")


class SavedModifier(Base):
    __tablename__ = "saved_modifiers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(80), nullable=False)
    modifier_type: Mapped[str] = mapped_column(String(24), nullable=False, default="increase")
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_time: Mapped[datetime.time] = mapped_column(Time, nullable=False)
    end_time: Mapped[datetime.time] = mapped_column(Time, nullable=False)
    pct_change: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    notes: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    created_by: Mapped[str] = mapped_column(String(60), nullable=False, default="system")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
    )


class Policy(PolicyBase):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    paramsJSON: Mapped[str] = mapped_column(String(8000), nullable=False, default="{}")
    lastEditedBy: Mapped[str] = mapped_column(String(60), nullable=False, default="system")
    lastEditedAt: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint("name", name="uq_policies_name"),
    )

    def params_dict(self) -> Dict:
        try:
            value = json.loads(self.paramsJSON or "{}")
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        return {}


class WeekSchedule(Base):
    __tablename__ = "week_schedule"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    week_start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, unique=True)
    iso_year: Mapped[int] = mapped_column(Integer, nullable=False)
    iso_week: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(48), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    context_id: Mapped[int | None] = mapped_column(ForeignKey("week_context.id"), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
    )

    week_context: Mapped["WeekContext"] = relationship()
    shifts: Mapped[List["Shift"]] = relationship(
        back_populates="week",
        cascade="all, delete-orphan",
    )


class Shift(Base):
    __tablename__ = "shifts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    week_id: Mapped[int] = mapped_column(ForeignKey("week_schedule.id", ondelete="CASCADE"), nullable=False)
    employee_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    role: Mapped[str] = mapped_column(String(40), nullable=False)
    start: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    location: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    notes: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    labor_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    labor_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
    )

    week: Mapped[WeekSchedule] = relationship(back_populates="shifts")
class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(60), nullable=False)
    action: Mapped[str] = mapped_column(String(60), nullable=False)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False, default="Shift")
    target_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payloadJSON: Mapped[str] = mapped_column(String(2000), nullable=False, default="{}")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc)
    )


employee_engine = create_engine(
    EMPLOYEE_DATABASE_URL,
    echo=False,
    future=True,
)
schedule_engine = create_engine(
    SCHEDULE_DATABASE_URL,
    echo=False,
    future=True,
)
policy_engine = create_engine(
    POLICY_DATABASE_URL,
    echo=False,
    future=True,
)
projections_engine = create_engine(
    PROJECTIONS_DATABASE_URL,
    echo=False,
    future=True,
)
SessionLocal = sessionmaker(bind=schedule_engine, expire_on_commit=False, future=True)
EmployeeSessionLocal = sessionmaker(bind=employee_engine, expire_on_commit=False, future=True)
PolicySessionLocal = sessionmaker(bind=policy_engine, expire_on_commit=False, future=True)
ProjectionSessionLocal = sessionmaker(bind=projections_engine, expire_on_commit=False, future=True)


def init_database() -> None:
    EmployeeBase.metadata.create_all(employee_engine)
    Base.metadata.create_all(schedule_engine)
    PolicyBase.metadata.create_all(policy_engine)
    ProjectionsBase.metadata.create_all(projections_engine)
    with employee_engine.begin() as conn:
        columns = {
            row[1]: True
            for row in conn.execute(text("PRAGMA table_info(employees)"))
        }
        if "start_month" not in columns:
            conn.execute(text("ALTER TABLE employees ADD COLUMN start_month INTEGER"))
        if "start_year" not in columns:
            conn.execute(text("ALTER TABLE employees ADD COLUMN start_year INTEGER"))
        wage_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='employee_role_wages'")
        ).scalar()
        if not wage_exists:
            EmployeeBase.metadata.tables["employee_role_wages"].create(conn)
    with schedule_engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='week_daily_projections'")
        ).scalar()
        if table_exists:
            projection_columns = {
                row[1]: True
                for row in conn.execute(text("PRAGMA table_info(week_daily_projections)"))
            }
            if "projected_sales_amount" not in projection_columns and "projected_labor_hours" in projection_columns:
                conn.execute(
                    text(
                        "ALTER TABLE week_daily_projections "
                        "RENAME COLUMN projected_labor_hours TO projected_sales_amount"
                    )
                )
        policy_exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='policies'")
        ).scalar()
        if policy_exists:
            cols = {row[1]: True for row in conn.execute(text("PRAGMA table_info(policies)"))}
            if "lastEditedBy" not in cols:
                conn.execute(text("ALTER TABLE policies ADD COLUMN lastEditedBy VARCHAR(60) NOT NULL DEFAULT 'system'"))
            if "lastEditedAt" not in cols:
                conn.execute(
                    text(
                        "ALTER TABLE policies ADD COLUMN lastEditedAt DATETIME DEFAULT (datetime('now'))"
                    )
                )
    with policy_engine.begin() as conn:
        cols = {row[1]: True for row in conn.execute(text("PRAGMA table_info(policies)"))}
        if "lastEditedBy" not in cols:
            conn.execute(text("ALTER TABLE policies ADD COLUMN lastEditedBy VARCHAR(60) NOT NULL DEFAULT 'system'"))
        if "lastEditedAt" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE policies ADD COLUMN lastEditedAt DATETIME DEFAULT (datetime('now'))"
                )
            )
    _migrate_legacy_projections()


def _migrate_legacy_projections() -> None:
    """Populate the dedicated projections database from any legacy schedule data."""
    try:
        with projections_engine.connect() as conn:
            existing_rows = conn.execute(text("SELECT COUNT(*) FROM week_daily_projections")).scalar()
            if existing_rows and int(existing_rows) > 0:
                return
    except Exception:
        # If the projections DB is not reachable, silently skip migration.
        return

    with schedule_engine.connect() as legacy_conn:
        legacy_table = legacy_conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='week_daily_projections'")
        ).scalar()
        if not legacy_table:
            return
        columns = {row[1]: True for row in legacy_conn.execute(text("PRAGMA table_info(week_daily_projections)"))}
        if "projected_sales_amount" in columns:
            amount_column = "projected_sales_amount"
        elif "projected_labor_hours" in columns:
            amount_column = "projected_labor_hours"
        else:
            return
        legacy_rows = legacy_conn.execute(
            text(
                f"SELECT week_id, day_of_week, {amount_column} AS amount, "
                "COALESCE(projected_notes, '') AS notes FROM week_daily_projections"
            )
        ).fetchall()
        contexts = {
            row[0]: {"iso_year": row[1], "iso_week": row[2], "label": row[3]}
            for row in legacy_conn.execute(text("SELECT id, iso_year, iso_week, label FROM week_context"))
        }
    if not legacy_rows:
        return

    with ProjectionSessionLocal() as projection_session:
        for week_id, day_of_week, amount, notes in legacy_rows:
            ctx_info = contexts.get(week_id)
            if not ctx_info:
                continue
            projection_context = get_or_create_projection_context(
                projection_session,
                iso_year=int(ctx_info["iso_year"]),
                iso_week=int(ctx_info["iso_week"]),
                label=ctx_info.get("label") or f"{ctx_info['iso_year']} W{ctx_info['iso_week']:02d}",
                schedule_context_id=week_id,
            )
            existing = projection_session.scalars(
                select(WeekDailyProjection).where(
                    WeekDailyProjection.projection_context_id == projection_context.id,
                    WeekDailyProjection.day_of_week == int(day_of_week),
                )
            ).first()
            if existing:
                existing.projected_sales_amount = float(amount or 0.0)
                existing.projected_notes = notes or ""
                continue
            projection_session.add(
                WeekDailyProjection(
                    projection_context_id=projection_context.id,
                    schedule_context_id=week_id,
                    iso_year=projection_context.iso_year,
                    iso_week=projection_context.iso_week,
                    label=projection_context.label,
                    day_of_week=int(day_of_week),
                    projected_sales_amount=float(amount or 0.0),
                    projected_notes=notes or "",
                )
            )
        projection_session.commit()


def _coerce_employee_session(session):
    """Return (employee_session, should_close) ensuring we talk to the employee database."""
    if session is None:
        return EmployeeSessionLocal(), True
    bind = getattr(session, "bind", None)
    if bind is schedule_engine:
        return EmployeeSessionLocal(), True
    return session, False


def _coerce_policy_session(session):
    """Return (policy_session, should_close) ensuring policy data stays in its own database."""
    PolicyBase.metadata.create_all(policy_engine)
    if session is None:
        return PolicySessionLocal(), True
    bind = getattr(session, "bind", None)
    if bind is schedule_engine or bind is employee_engine:
        return PolicySessionLocal(), True
    return session, False


def _coerce_projection_session(session):
    """Return (projection_session, should_close) scoped to the projections database."""
    ProjectionsBase.metadata.create_all(projections_engine)
    if session is None:
        return ProjectionSessionLocal(), True
    bind = getattr(session, "bind", None)
    if bind in {schedule_engine, employee_engine, policy_engine}:
        return ProjectionSessionLocal(), True
    return session, False


def _resolve_week_context(schedule_session, week_identifier) -> WeekContext | None:
    if isinstance(week_identifier, WeekContext):
        return week_identifier
    if schedule_session is None:
        return None
    try:
        return schedule_session.get(WeekContext, week_identifier)
    except Exception:
        stmt = select(WeekContext).where(WeekContext.id == week_identifier)
        return schedule_session.scalars(stmt).first()


def get_or_create_projection_context(
    projection_session,
    *,
    iso_year: int,
    iso_week: int,
    label: str,
    schedule_context_id: int | None = None,
) -> WeekProjectionContext:
    projection_session, close_session = _coerce_projection_session(projection_session)
    try:
        if schedule_context_id is not None:
            existing = projection_session.scalars(
                select(WeekProjectionContext).where(WeekProjectionContext.schedule_context_id == schedule_context_id)
            ).first()
            if existing:
                if label and existing.label != label:
                    existing.label = label
                    projection_session.commit()
                    projection_session.refresh(existing)
                return existing
        stmt = select(WeekProjectionContext).where(
            WeekProjectionContext.iso_year == iso_year,
            WeekProjectionContext.iso_week == iso_week,
        )
        context = projection_session.scalars(stmt).first()
        if context:
            if schedule_context_id is not None and context.schedule_context_id is None:
                context.schedule_context_id = schedule_context_id
            if label and context.label != label:
                context.label = label
            projection_session.commit()
            projection_session.refresh(context)
            return context
        context = WeekProjectionContext(
            schedule_context_id=schedule_context_id,
            iso_year=iso_year,
            iso_week=iso_week,
            label=label or f"{iso_year} W{iso_week:02d}",
        )
        projection_session.add(context)
        projection_session.commit()
        projection_session.refresh(context)
        return context
    finally:
        if close_session:
            projection_session.close()


def get_all_employees(employee_session=None) -> List[Employee]:
    employee_session, close_session = _coerce_employee_session(employee_session)
    try:
        stmt = select(Employee).order_by(Employee.full_name)
        return list(employee_session.scalars(stmt))
    finally:
        if close_session:
            employee_session.close()


def get_all_weeks(session) -> List[WeekContext]:
    stmt = select(WeekContext).order_by(WeekContext.iso_year.desc(), WeekContext.iso_week.desc())
    return list(session.scalars(stmt))


def get_or_create_week_context(session, iso_year: int, iso_week: int, label: str) -> WeekContext:
    stmt = select(WeekContext).where(
        WeekContext.iso_year == iso_year,
        WeekContext.iso_week == iso_week,
    )
    existing = session.scalars(stmt).first()
    if existing:
        return existing
    week = WeekContext(iso_year=iso_year, iso_week=iso_week, label=label)
    session.add(week)
    session.commit()
    session.refresh(week)
    return week


def get_week_daily_projections(schedule_session, week_id: int | WeekContext, *, projection_session=None) -> List[WeekDailyProjection]:
    close_schedule = False
    if schedule_session is None:
        schedule_session = SessionLocal()
        close_schedule = True
    week_context = _resolve_week_context(schedule_session, week_id)
    if not week_context:
        if close_schedule:
            schedule_session.close()
        return []
    projection_session, close_projection = _coerce_projection_session(projection_session)
    projection_context = get_or_create_projection_context(
        projection_session,
        iso_year=week_context.iso_year,
        iso_week=week_context.iso_week,
        label=week_context.label,
        schedule_context_id=week_context.id,
    )
    stmt = select(WeekDailyProjection).where(WeekDailyProjection.projection_context_id == projection_context.id)
    projections = {item.day_of_week: item for item in projection_session.scalars(stmt)}
    created = False
    for day in range(7):
        if day not in projections:
            projection = WeekDailyProjection(
                projection_context_id=projection_context.id,
                schedule_context_id=week_context.id,
                iso_year=projection_context.iso_year,
                iso_week=projection_context.iso_week,
                label=projection_context.label,
                day_of_week=day,
                projected_sales_amount=0.0,
            )
            projection_session.add(projection)
            projections[day] = projection
            created = True
    if created:
        projection_session.commit()
        for projection in projections.values():
            projection_session.refresh(projection)
    result = [projections[day] for day in sorted(projections)]
    if close_projection:
        projection_session.close()
    if close_schedule:
        schedule_session.close()
    return result


def save_week_daily_projection_values(
    schedule_session,
    week_id: int | WeekContext,
    values: Dict[int, Dict[str, float | str]],
    *,
    projection_session=None,
) -> None:
    close_schedule = False
    if schedule_session is None:
        schedule_session = SessionLocal()
        close_schedule = True
    projection_session, close_projection = _coerce_projection_session(projection_session)
    projections = get_week_daily_projections(
        schedule_session,
        week_id,
        projection_session=projection_session,
    )
    mapping = {item.day_of_week: item for item in projections}
    for day, payload in (values or {}).items():
        try:
            day_index = int(day)
        except Exception:
            continue
        projection = mapping.get(day_index)
        if not projection:
            continue
        amount = float(payload.get("projected_sales_amount", projection.projected_sales_amount))
        notes = str(payload.get("projected_notes", projection.projected_notes or ""))
        projection.projected_sales_amount = max(amount, 0.0)
        projection.projected_notes = notes.strip()
    projection_session.commit()
    if close_projection:
        projection_session.close()
    if close_schedule:
        schedule_session.close()


def get_week_modifiers(session, week_id: int) -> List[Modifier]:
    stmt = (
        select(Modifier)
        .where(Modifier.week_id == week_id)
        .order_by(Modifier.day_of_week, Modifier.start_time, Modifier.id)
    )
    return list(session.scalars(stmt))


def list_saved_modifiers(session) -> List[SavedModifier]:
    stmt = select(SavedModifier).order_by(SavedModifier.title.asc(), SavedModifier.id.asc())
    return list(session.scalars(stmt))


def save_modifier_template(
    session,
    *,
    title: str,
    modifier_type: str,
    day_of_week: int,
    start_time: datetime.time,
    end_time: datetime.time,
    pct_change: int,
    notes: str,
    created_by: str,
) -> SavedModifier:
    template = SavedModifier(
        title=title,
        modifier_type=modifier_type if modifier_type in {"increase", "decrease"} else "increase",
        day_of_week=max(0, min(day_of_week, 6)),
        start_time=start_time,
        end_time=end_time,
        pct_change=pct_change,
        notes=notes or "",
        created_by=created_by,
    )
    session.add(template)
    session.commit()
    session.refresh(template)
    return template


def delete_saved_modifier(session, template_id: int) -> None:
    session.query(SavedModifier).filter(SavedModifier.id == template_id).delete()
    session.commit()


def apply_saved_modifier_to_week(
    session,
    template_id: int,
    week_id: int,
    *,
    created_by: str,
) -> Modifier:
    template = session.get(SavedModifier, template_id)
    if not template:
        raise ValueError("Saved modifier not found.")
    modifier = Modifier(
        week_id=week_id,
        title=template.title,
        modifier_type=template.modifier_type,
        day_of_week=template.day_of_week,
        start_time=template.start_time,
        end_time=template.end_time,
        pct_change=template.pct_change,
        notes=template.notes,
        created_by=created_by,
    )
    session.add(modifier)
    session.commit()
    session.refresh(modifier)
    return modifier


def get_policies(session) -> List[Policy]:
    policy_session, close_session = _coerce_policy_session(session)
    try:
        stmt = select(Policy).order_by(Policy.name.asc(), Policy.id.asc())
        return list(policy_session.scalars(stmt))
    finally:
        if close_session:
            policy_session.close()


def upsert_policy(session, name: str, params_dict: Dict, *, edited_by: str = "system") -> Policy:
    policy_session, close_session = _coerce_policy_session(session)
    try:
        existing: Optional[Policy] = policy_session.execute(
            select(Policy).where(Policy.name == name)
        ).scalars().first()
        payload = params_dict if isinstance(params_dict, dict) else {}
        if existing:
            existing.paramsJSON = json.dumps(payload)
            existing.lastEditedBy = edited_by
            existing.lastEditedAt = datetime.datetime.now(datetime.timezone.utc)
            policy_session.commit()
            policy_session.refresh(existing)
            return existing
        policy = Policy(
            name=name,
            paramsJSON=json.dumps(payload),
            lastEditedBy=edited_by,
            lastEditedAt=datetime.datetime.now(datetime.timezone.utc),
        )
        policy_session.add(policy)
        policy_session.commit()
        policy_session.refresh(policy)
        return policy
    finally:
        if close_session:
            policy_session.close()


def delete_policy(session, policy_id: int) -> None:
    policy_session, close_session = _coerce_policy_session(session)
    try:
        policy_session.query(Policy).filter(Policy.id == policy_id).delete()
        policy_session.commit()
    finally:
        if close_session:
            policy_session.close()


def get_active_policy(session) -> Optional[Policy]:
    policy_session, close_session = _coerce_policy_session(session)
    try:
        stmt = select(Policy).order_by(Policy.lastEditedAt.desc(), Policy.id.desc())
        return policy_session.scalars(stmt).first()
    finally:
        if close_session:
            policy_session.close()


OVERNIGHT_CLOSE_CUTOFF_HOUR = 6


def _ensure_aware(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.timezone.utc)
    return value.astimezone(datetime.timezone.utc)


def shift_display_date(start: datetime.datetime, location: Optional[str]) -> datetime.date:
    """Return the canonical day a shift should belong to for reporting."""

    aware_start = _ensure_aware(start)
    day = aware_start.date()
    if (location or "").strip().lower() == "close" and aware_start.hour < OVERNIGHT_CLOSE_CUTOFF_HOUR:
        return day - datetime.timedelta(days=1)
    return day


def _shift_to_dict(shift: Shift, employee: Optional[Employee]) -> Dict[str, Any]:
    return {
        "id": shift.id,
        "week_id": shift.week_id,
        "employee_id": shift.employee_id,
        "employee_name": employee.full_name if employee else None,
        "role": shift.role,
        "start": shift.start,
        "end": shift.end,
        "location": shift.location,
        "notes": shift.notes,
        "status": shift.status,
        "labor_rate": shift.labor_rate,
        "labor_cost": shift.labor_cost,
    }


def get_or_create_week(session, week_start_date: datetime.date) -> WeekSchedule:
    if not isinstance(week_start_date, (datetime.date, datetime.datetime)):
        raise TypeError("week_start_date must be a date or datetime instance.")
    normalized = _normalize_week_start(
        week_start_date.date() if isinstance(week_start_date, datetime.datetime) else week_start_date
    )
    iso_year, iso_week, _ = normalized.isocalendar()
    label = _format_week_label(normalized)
    stmt = select(WeekSchedule).where(WeekSchedule.week_start_date == normalized)
    week = session.scalars(stmt).first()
    if week:
        return week
    context = get_or_create_week_context(session, iso_year, iso_week, label)
    week = WeekSchedule(
        week_start_date=normalized,
        iso_year=iso_year,
        iso_week=iso_week,
        label=label,
        status="draft",
        context_id=context.id if context else None,
    )
    session.add(week)
    session.commit()
    session.refresh(week)
    return week


def list_employees(employee_session=None, only_active: bool = True) -> List[Dict[str, Any]]:
    employee_session, close_session = _coerce_employee_session(employee_session)
    try:
        stmt = select(Employee)
        if only_active:
            stmt = stmt.where(Employee.status == "active")
        stmt = stmt.order_by(Employee.full_name.asc())
        employees = []
        for employee in employee_session.scalars(stmt):
            employees.append(
                {
                    "id": employee.id,
                    "name": employee.full_name,
                    "roles": employee.role_list,
                    "status": employee.status,
                    "desired_hours": employee.desired_hours,
                }
            )
        return employees
    finally:
        if close_session:
            employee_session.close()


def list_roles(schedule_session, employee_session=None) -> List[str]:
    employee_session, close_session = _coerce_employee_session(employee_session)
    roles = set()
    for employee in employee_session.scalars(select(Employee)):
        roles.update({role for role in employee.role_list if not is_manager_role(role)})
    for role in schedule_session.execute(select(Shift.role).distinct()):
        value = role[0]
        if value and not is_manager_role(value):
            roles.add(value)
    if close_session:
        employee_session.close()
    return sorted(roles)


def list_modifiers_for_week(session, week_start_date: datetime.date) -> List[Dict[str, Any]]:
    normalized = _normalize_week_start(week_start_date)
    iso_year, iso_week, _ = normalized.isocalendar()
    context_stmt = select(WeekContext).where(
        WeekContext.iso_year == iso_year,
        WeekContext.iso_week == iso_week,
    )
    week_context = session.scalars(context_stmt).first()
    if not week_context:
        return []
    modifiers = get_week_modifiers(session, week_context.id)
    payload = []
    for modifier in modifiers:
        payload.append(
            {
                "id": modifier.id,
                "title": modifier.title,
                "type": modifier.modifier_type,
                "day_of_week": modifier.day_of_week,
                "start_time": modifier.start_time,
                "end_time": modifier.end_time,
                "pct_change": modifier.pct_change,
            }
        )
    return payload


def get_shifts_for_week(
    session,
    week_start_date: datetime.date,
    *,
    employee_id: Optional[int] = None,
    role: Optional[str] = None,
    status: Optional[str] = None,
    employee_session=None,
) -> List[Dict[str, Any]]:
    week = get_or_create_week(session, week_start_date)
    stmt = (
        select(Shift)
        .where(Shift.week_id == week.id)
        .order_by(Shift.start, Shift.end)
    )
    if employee_id:
        stmt = stmt.where(Shift.employee_id == employee_id)
    if role and role != "All":
        stmt = stmt.where(Shift.role == role)
    if status and status.lower() != "all":
        stmt = stmt.where(Shift.status == status.lower())
    shifts = list(session.scalars(stmt))
    employees: Dict[int, Employee] = {}
    employee_session, close_session = _coerce_employee_session(employee_session)
    if employee_session:
        employees = {emp.id: emp for emp in employee_session.scalars(select(Employee))}
    payload = []
    for shift in shifts:
        if is_manager_role(shift.role):
            continue
        employee = employees.get(shift.employee_id) if employees else None
        payload.append(_shift_to_dict(shift, employee))
    if close_session:
        employee_session.close()
    return payload


def get_employee_role_wages(employee_session=None, employee_ids: Optional[Iterable[int]] = None) -> Dict[int, Dict[str, float]]:
    employee_session, close_session = _coerce_employee_session(employee_session)
    try:
        stmt = select(EmployeeRoleWage)
        ids = list(employee_ids or [])
        if ids:
            stmt = stmt.where(EmployeeRoleWage.employee_id.in_(ids))
        wages: Dict[int, Dict[str, float]] = {}
        for row in employee_session.scalars(stmt):
            wages.setdefault(row.employee_id, {})[row.role] = float(row.wage or 0.0)
        return wages
    finally:
        if close_session:
            employee_session.close()


def save_employee_role_wages(
    employee_session,
    employee_id: int,
    mapping: Dict[str, float],
    *,
    commit: bool = True,
) -> int:
    employee_session, close_session = _coerce_employee_session(employee_session)
    try:
        employee_session.execute(delete(EmployeeRoleWage).where(EmployeeRoleWage.employee_id == employee_id))
        count = 0
        for role, wage in (mapping or {}).items():
            try:
                wage_value = round(float(wage), 2)
            except (TypeError, ValueError):
                continue
            entry = EmployeeRoleWage(
                employee_id=employee_id,
                role=role,
                wage=max(0.0, wage_value),
                confirmed=1,
            )
            employee_session.add(entry)
            count += 1
        if commit or close_session:
            employee_session.commit()
        return count
    finally:
        if close_session:
            employee_session.close()


def get_week_summary(session, week_start_date: datetime.date) -> Dict[str, Any]:
    normalized = _normalize_week_start(week_start_date)
    week = get_or_create_week(session, normalized)
    if not week.context_id:
        context = get_or_create_week_context(session, week.iso_year, week.iso_week, week.label)
        week.context_id = context.id
        session.commit()
    context_id = week.context_id
    start_of_day = normalized
    totals = {}
    total_cost = 0.0
    total_shifts = 0
    for index in range(7):
        day = start_of_day + datetime.timedelta(days=index)
        totals[day] = {"count": 0, "cost": 0.0}
    for shift in session.scalars(select(Shift).where(Shift.week_id == week.id)):
        if is_manager_role(shift.role):
            continue
        shift_day = shift_display_date(shift.start, shift.location)
        info = totals.get(shift_day)
        if info is None:
            continue
        info["count"] += 1
        info["cost"] += float(shift.labor_cost or 0.0)
        total_cost += float(shift.labor_cost or 0.0)
        total_shifts += 1

    projections = get_week_daily_projections(session, context_id)
    modifiers = list_modifiers_for_week(session, normalized)

    modifier_map: Dict[int, List[Dict[str, float]]] = {idx: [] for idx in range(7)}
    for mod in modifiers or []:
        day_idx = int(mod.get("day_of_week", 0))
        start = mod.get("start_time")
        end = mod.get("end_time")
        pct = float(mod.get("pct_change", 0) or 0)
        start_minutes = start.hour * 60 + start.minute if start else 0
        end_minutes = end.hour * 60 + end.minute if end else 24 * 60
        modifier_map.setdefault(day_idx, []).append(
            {
                "start": start_minutes,
                "end": end_minutes,
                "pct": pct,
            }
        )

    def _modifier_multiplier(day_index: int) -> float:
        windows = modifier_map.get(day_index, [])
        if not windows:
            return 1.0
        total = 0.0
        for window in windows:
            span = max(0.0, window["end"] - window["start"])
            fraction = span / (24 * 60)
            total += (window["pct"] / 100.0) * max(fraction, 0.1)
        return max(0.5, 1.0 + total)

    projected_sales_raw = 0.0
    projected_sales_total = 0.0
    for projection in projections:
        sales = float(projection.projected_sales_amount or 0.0)
        projected_sales_raw += sales
        day_idx = int(getattr(projection, "day_of_week", 0) or 0)
        projected_sales_total += sales * _modifier_multiplier(day_idx)
    if projected_sales_total <= 0:
        # Fallback to actual labor spend if projections are missing so the UI can at least show current percentages.
        projected_sales_total = total_cost
    days_payload = [
        {
            "date": day.isoformat(),
            "count": info["count"],
            "cost": round(info["cost"], 2),
        }
        for day, info in sorted(totals.items())
    ]
    return {
        "week_id": week.id,
        "status": week.status,
        "days": days_payload,
        "total_cost": round(total_cost, 2),
        "total_shifts": total_shifts,
        "projected_sales_total": round(projected_sales_total, 2),
        "projected_sales_total_raw": round(projected_sales_raw, 2),
    }


def _apply_week_status(session, week: WeekSchedule, status: str) -> str:
    normalized = (status or "").strip().lower()
    if normalized not in WEEK_STATUS_CHOICES:
        raise ValueError(f"Unsupported week status '{status}'.")
    week.status = normalized
    session.execute(
        update(Shift)
        .where(Shift.week_id == week.id)
        .values(status=normalized)
    )
    return normalized


def set_week_status(session, week_start_date: datetime.date, status: str) -> WeekSchedule:
    normalized_date = _normalize_week_start(week_start_date)
    week = get_or_create_week(session, normalized_date)
    _apply_week_status(session, week, status)
    session.commit()
    session.refresh(week)
    return week


def upsert_shift(session, shift: Dict[str, Any]) -> int:
    shift_id = shift.get("id")
    start = shift.get("start")
    end = shift.get("end")
    if not isinstance(start, datetime.datetime) or not isinstance(end, datetime.datetime):
        raise TypeError("Shift start and end must be datetime instances.")
    start = _ensure_aware(start)
    end = _ensure_aware(end)
    if end <= start:
        raise ValueError("Shift end time must be after start time.")

    week_id = shift.get("week_id")
    week_start = shift.get("week_start_date") or shift.get("week_start")
    if week_id:
        week = session.get(WeekSchedule, week_id)
        if not week:
            raise ValueError(f"WeekSchedule with id {week_id} was not found.")
    else:
        if not week_start:
            week_start = start.date()
        week = get_or_create_week(session, week_start)
        week_id = week.id

    role = shift.get("role")
    if not role:
        raise ValueError("Shift role is required.")
    if is_manager_role(role):
        raise ValueError("Manager roles are not scheduled through this tool.")

    labor_rate = float(shift.get("labor_rate", 0.0) or 0.0)
    labor_cost = shift.get("labor_cost")
    hours = (end - start).total_seconds() / 3600
    if labor_cost is None:
        labor_cost = round(hours * labor_rate, 2)

    if shift_id:
        db_shift = session.get(Shift, shift_id)
        if not db_shift:
            raise ValueError(f"Shift with id {shift_id} was not found.")
    else:
        db_shift = Shift(week_id=week_id)
        session.add(db_shift)

    db_shift.employee_id = shift.get("employee_id")
    db_shift.role = role
    db_shift.start = start
    db_shift.end = end
    db_shift.location = shift.get("location", "") or ""
    db_shift.notes = shift.get("notes", "") or ""
    db_shift.status = (shift.get("status") or "draft").lower()
    db_shift.labor_rate = labor_rate
    db_shift.labor_cost = float(labor_cost)
    _apply_week_status(session, week, "draft")
    session.commit()
    session.refresh(db_shift)
    return db_shift.id


def delete_shift(session, shift_id: int) -> None:
    db_shift = session.get(Shift, shift_id)
    if not db_shift:
        return
    week = db_shift.week
    session.delete(db_shift)
    if week:
        _apply_week_status(session, week, "draft")
    session.commit()


def record_audit_log(
    session,
    user_id: str,
    action: str,
    target_type: str = "Shift",
    target_id: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    log = AuditLog(
        user_id=user_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        payloadJSON=json.dumps(payload or {}),
    )
    session.add(log)
    session.commit()
    return log
