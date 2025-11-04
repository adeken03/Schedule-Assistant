from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.types import Time


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{(DATA_DIR / 'schedule.db').as_posix()}"


class Base(DeclarativeBase):
    pass


class Employee(Base):
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

    @property
    def role_list(self) -> List[str]:
        return [role.strip() for role in self.roles.split(",") if role.strip()]

    @role_list.setter
    def role_list(self, roles: Iterable[str]) -> None:
        self.roles = ", ".join(sorted({role.strip() for role in roles if role.strip()}))

    @property
    def start_date_label(self) -> str:
        if self.start_month and self.start_year:
            month_name = datetime.date(1900, self.start_month, 1).strftime("%b")
            return f"{month_name} {self.start_year}"
        if self.start_year:
            return str(self.start_year)
        return "Not set"


class EmployeeUnavailability(Base):
    __tablename__ = "employee_unavailability"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"))
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 = Monday
    start_time: Mapped[datetime.time] = mapped_column(Time, nullable=False)
    end_time: Mapped[datetime.time] = mapped_column(Time, nullable=False)

    employee: Mapped[Employee] = relationship(back_populates="unavailability")


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

    projections: Mapped[List["WeekDailyProjection"]] = relationship(
        back_populates="week", cascade="all, delete-orphan"
    )
    modifiers: Mapped[List["Modifier"]] = relationship(
        back_populates="week", cascade="all, delete-orphan"
    )


class WeekDailyProjection(Base):
    __tablename__ = "week_daily_projections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    week_id: Mapped[int] = mapped_column(ForeignKey("week_context.id", ondelete="CASCADE"), nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    projected_sales_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    projected_notes: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
    )

    week: Mapped[WeekContext] = relationship(back_populates="projections")

    __table_args__ = (UniqueConstraint("week_id", "day_of_week", name="uq_daily_projection_week_day"),)


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


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    # JSON payload encoded as string; parse at runtime
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


engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_database() -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        columns = {
            row[1]: True
            for row in conn.execute(text("PRAGMA table_info(employees)"))
        }
        if "start_month" not in columns:
            conn.execute(text("ALTER TABLE employees ADD COLUMN start_month INTEGER"))
        if "start_year" not in columns:
            conn.execute(text("ALTER TABLE employees ADD COLUMN start_year INTEGER"))
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
        # Ensure policies table exists and has expected columns if previously created without them
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


def get_all_employees(session) -> List[Employee]:
    stmt = select(Employee).order_by(Employee.full_name)
    return list(session.scalars(stmt))


def get_all_weeks(session) -> List[WeekContext]:
    stmt = select(WeekContext).order_by(WeekContext.iso_year.desc(), WeekContext.iso_week.desc())
    return list(session.scalars(stmt))


def get_or_create_week(session, iso_year: int, iso_week: int, label: str) -> WeekContext:
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


def get_week_daily_projections(session, week_id: int) -> List[WeekDailyProjection]:
    stmt = select(WeekDailyProjection).where(WeekDailyProjection.week_id == week_id)
    projections = {item.day_of_week: item for item in session.scalars(stmt)}
    created = False
    for day in range(7):
        if day not in projections:
            projection = WeekDailyProjection(week_id=week_id, day_of_week=day, projected_sales_amount=0.0)
            session.add(projection)
            projections[day] = projection
            created = True
    if created:
        session.commit()
        for projection in projections.values():
            session.refresh(projection)
    return [projections[day] for day in sorted(projections)]


def save_week_daily_projection_values(
    session,
    week_id: int,
    values: Dict[int, Dict[str, float | str]],
) -> None:
    projections = get_week_daily_projections(session, week_id)
    mapping = {item.day_of_week: item for item in projections}
    for day, payload in values.items():
        projection = mapping.get(day)
        if not projection:
            continue
        amount = float(payload.get("projected_sales_amount", projection.projected_sales_amount))
        notes = str(payload.get("projected_notes", projection.projected_notes or ""))
        projection.projected_sales_amount = max(amount, 0.0)
        projection.projected_notes = notes.strip()
    session.commit()


def get_week_modifiers(session, week_id: int) -> List[Modifier]:
    stmt = (
        select(Modifier)
        .where(Modifier.week_id == week_id)
        .order_by(Modifier.day_of_week, Modifier.start_time, Modifier.id)
    )
    return list(session.scalars(stmt))


# Policy helpers
def get_policies(session) -> List[Policy]:
    stmt = select(Policy).order_by(Policy.name.asc(), Policy.id.asc())
    return list(session.scalars(stmt))


def upsert_policy(session, name: str, params_dict: Dict, *, edited_by: str = "system") -> Policy:
    existing: Optional[Policy] = session.execute(
        select(Policy).where(Policy.name == name)
    ).scalars().first()
    payload = params_dict if isinstance(params_dict, dict) else {}
    if existing:
        existing.paramsJSON = json.dumps(payload)
        existing.lastEditedBy = edited_by
        existing.lastEditedAt = datetime.datetime.now(datetime.timezone.utc)
        session.commit()
        session.refresh(existing)
        return existing
    policy = Policy(
        name=name,
        paramsJSON=json.dumps(payload),
        lastEditedBy=edited_by,
        lastEditedAt=datetime.datetime.now(datetime.timezone.utc),
    )
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def delete_policy(session, policy_id: int) -> None:
    session.query(Policy).filter(Policy.id == policy_id).delete()
    session.commit()


def get_active_policy(session) -> Optional[Policy]:
    # Most recently edited policy is considered active.
    stmt = select(Policy).order_by(Policy.lastEditedAt.desc(), Policy.id.desc())
    return session.scalars(stmt).first()
