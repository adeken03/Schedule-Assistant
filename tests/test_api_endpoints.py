from __future__ import annotations

import datetime
import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import database as db  # noqa: E402
from database import Base, EmployeeBase, ProjectionsBase, WeekDailyProjection, WeekProjectionContext  # noqa: E402


@pytest.fixture()
def api_client(monkeypatch):
    schedule_engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    projection_engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Session = sessionmaker(bind=schedule_engine, expire_on_commit=False, future=True)
    ProjectionSession = sessionmaker(bind=projection_engine, expire_on_commit=False, future=True)

    # Patch database module to point everything at the in-memory engine.
    monkeypatch.setattr(db, "schedule_engine", schedule_engine)
    monkeypatch.setattr(db, "employee_engine", schedule_engine)
    monkeypatch.setattr(db, "policy_engine", schedule_engine)
    monkeypatch.setattr(db, "projections_engine", projection_engine)
    monkeypatch.setattr(db, "SessionLocal", Session)
    monkeypatch.setattr(db, "EmployeeSessionLocal", Session)
    monkeypatch.setattr(db, "PolicySessionLocal", Session)
    monkeypatch.setattr(db, "ProjectionSessionLocal", ProjectionSession)

    Base.metadata.create_all(schedule_engine)
    EmployeeBase.metadata.create_all(schedule_engine)
    ProjectionsBase.metadata.create_all(projection_engine)

    # Reload API after patching database bindings.
    import app.api as api  # type: ignore

    api = importlib.reload(api)

    shared_session = Session()
    shared_employee_session = Session()

    def _get_db():
        try:
            yield shared_session
        finally:
            pass

    def _get_employee_db():
        try:
            yield shared_employee_session
        finally:
            pass

    api.app.dependency_overrides[api.get_db] = _get_db
    api.app.dependency_overrides[api.get_employee_db] = _get_employee_db

    client = TestClient(api.app)
    try:
        yield client, Session, ProjectionSession
    finally:
        shared_session.close()
        shared_employee_session.close()
        schedule_engine.dispose()
        projection_engine.dispose()


def test_health_endpoint(api_client) -> None:
    client, _, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_login_stub_accepts_default_credentials(api_client) -> None:
    client, _, _ = api_client

    ok = client.post("/api/v1/auth/login", json={"username": "it_assistant", "password": "letmein"})
    bad = client.post("/api/v1/auth/login", json={"username": "nope", "password": "bad"})

    assert ok.status_code == 200
    assert ok.json().get("token") == "stub-token"
    assert bad.status_code == 401


def test_projection_endpoint_saves_notes(api_client) -> None:
    client, Session, ProjectionSession = api_client
    week = "2024-04-01"
    payload = {
        "actor": "api-tester",
        "days": [{"day_of_week": 0, "projected_sales_amount": 321.5, "projected_notes": "note"}],
        "modifiers": [],
    }

    resp = client.post(f"/api/v1/weeks/{week}/projection", json=payload)

    assert resp.status_code == 200
    with ProjectionSession() as session:
        ctx_id = session.scalar(select(WeekProjectionContext.id).limit(1))
        rows = session.scalars(
            select(WeekDailyProjection).where(WeekDailyProjection.projection_context_id == ctx_id)
        ).all()
        assert len(rows) == 7  # missing days are auto-created
        monday = next(row for row in rows if row.day_of_week == 0)
        assert monday.projected_sales_amount == pytest.approx(321.5)
        assert monday.projected_notes == "note"


def test_shifts_endpoint_returns_empty_payload(api_client) -> None:
    client, _, _ = api_client
    week = "2024-04-01"

    resp = client.get(f"/api/v1/weeks/{week}/shifts")

    assert resp.status_code == 200
    data = resp.json()
    assert data["week_start"] == week
    assert data["shifts"] == []
